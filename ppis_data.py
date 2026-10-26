import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ppis_config import CONFIG


def normalize_adjacency(matrix: np.ndarray) -> np.ndarray:
    rowsum = np.array(matrix.sum(1))
    inv_sqrt = np.power(rowsum, -0.5).flatten()
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    d_mat = np.diag(inv_sqrt)
    return d_mat @ matrix @ d_mat


def adj_matrix_to_edge_index(adj_matrix: np.ndarray) -> torch.Tensor:
    indices = torch.nonzero(torch.tensor(adj_matrix, dtype=torch.float32), as_tuple=False)
    return indices.t().long()


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def dataset_to_dataframe(dataset: Dict[str, Tuple[str, np.ndarray]]) -> pd.DataFrame:
    ids, sequences, labels = [], [], []
    for protein_id, item in dataset.items():
        ids.append(protein_id)
        sequences.append(item[0])
        labels.append(item[1])
    return pd.DataFrame({"ID": ids, "sequence": sequences, "label": labels})


def load_dataframe(dataset_name: str) -> pd.DataFrame:
    dataset = load_pickle(os.path.join(CONFIG.dataset_dir, dataset_name))
    if dataset_name == CONFIG.train_dataset and "2j3rA" in dataset:
        dataset.pop("2j3rA")
    return dataset_to_dataframe(dataset)


def load_feature(sequence_name: str, subdir: str) -> np.ndarray:
    return np.load(os.path.join(CONFIG.feature_dir, subdir, f"{sequence_name}.npy")).astype(np.float32)


def load_sequence_features(sequence_name: str) -> np.ndarray:
    pssm = load_feature(sequence_name, "pssm")
    hmm = load_feature(sequence_name, "hmm")
    dssp = load_feature(sequence_name, "dssp")
    resaf = load_feature(sequence_name, "resAF")
    base = np.concatenate([pssm, hmm, dssp, resaf], axis=1)
    return base.astype(np.float32)


def load_optional_esm2(sequence_name: str) -> np.ndarray:
    path = os.path.join(CONFIG.esm2_dir, f"{sequence_name}.npy")
    if not CONFIG.use_esm2:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"ESM2 feature not found: {path}")
    esm2 = np.load(path).astype(np.float32)
    if esm2.ndim != 2:
        raise ValueError(f"ESM2 feature must be 2D for {sequence_name}, got shape {esm2.shape}")
    if esm2.shape[1] != CONFIG.esm2_dim:
        raise ValueError(f"ESM2 dim must be {CONFIG.esm2_dim} for {sequence_name}, got {esm2.shape[1]}")
    return esm2


def resolve_geometry_settings(config_override: Dict = None) -> Dict[str, object]:
    if config_override is None:
        return {
            "map_cutoff": CONFIG.map_cutoff,
            "max_spatial_neighbors": CONFIG.max_spatial_neighbors,
            "pos_center_mode": CONFIG.pos_center_mode,
            "edge_attr_mode": CONFIG.edge_attr_mode,
            "edge_distance_rbf_centers": list(CONFIG.edge_distance_rbf_centers),
            "edge_include_sequence_gap": CONFIG.edge_include_sequence_gap,
        }

    # Older checkpoints do not store geometry settings. Keep their legacy feature layout
    # so they remain loadable after the geometry update.
    if "edge_attr_mode" not in config_override and "pos_center_mode" not in config_override:
        return {
            "map_cutoff": 14.0,
            "max_spatial_neighbors": None,
            "pos_center_mode": "first_residue",
            "edge_attr_mode": "legacy",
            "edge_distance_rbf_centers": [],
            "edge_include_sequence_gap": False,
        }

    return {
        "map_cutoff": float(config_override.get("map_cutoff", 14.0)),
        "max_spatial_neighbors": config_override.get("max_spatial_neighbors", None),
        "pos_center_mode": config_override.get("pos_center_mode", CONFIG.pos_center_mode),
        "edge_attr_mode": config_override.get("edge_attr_mode", CONFIG.edge_attr_mode),
        "edge_distance_rbf_centers": list(config_override.get("edge_distance_rbf_centers", CONFIG.edge_distance_rbf_centers)),
        "edge_include_sequence_gap": bool(config_override.get("edge_include_sequence_gap", CONFIG.edge_include_sequence_gap)),
    }


def calculate_edge_index(sequence_name: str, radius: float, max_neighbors: int = None) -> torch.Tensor:
    dist_matrix = load_feature(sequence_name, "distance_map_SC")
    valid_mask = (dist_matrix >= 0) & (dist_matrix <= radius)
    if max_neighbors is None:
        adjacency = valid_mask.astype(np.float32)
    else:
        max_neighbors = max(int(max_neighbors), 0)
        adjacency = np.zeros_like(dist_matrix, dtype=np.float32)
        for residue_index in range(dist_matrix.shape[0]):
            valid_neighbors = np.where(valid_mask[residue_index])[0]
            valid_neighbors = valid_neighbors[valid_neighbors != residue_index]
            if valid_neighbors.size == 0 or max_neighbors == 0:
                continue
            nearest_order = np.argsort(dist_matrix[residue_index, valid_neighbors], kind="stable")
            nearest_neighbors = valid_neighbors[nearest_order[:max_neighbors]]
            adjacency[nearest_neighbors, residue_index] = 1.0
    return adj_matrix_to_edge_index(adjacency)


def center_positions(pos: np.ndarray, center_mode: str) -> torch.Tensor:
    pos_tensor = torch.from_numpy(pos.astype(np.float32))
    if center_mode == "centroid":
        return pos_tensor - pos_tensor.mean(dim=0, keepdim=True)
    if center_mode == "first_residue":
        return pos_tensor - pos_tensor[0]
    raise ValueError(f"Unsupported pos_center_mode: {center_mode}")


def build_rbf_features(distance: torch.Tensor, centers: List[float]) -> torch.Tensor:
    if not centers:
        return distance.new_zeros((distance.size(0), 0))

    centers_tensor = distance.new_tensor(centers).view(1, -1)
    if len(centers) == 1:
        width = max(float(CONFIG.map_cutoff) * 0.25, 1.0)
    else:
        deltas = [float(next_center - center) for center, next_center in zip(centers[:-1], centers[1:])]
        width = max(sum(deltas) / len(deltas), 1e-3)

    return torch.exp(-((distance - centers_tensor) ** 2) / (2.0 * width * width))


def calculate_edge_attr(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    radius: float,
    edge_attr_mode: str,
    edge_distance_rbf_centers: List[float],
    edge_include_sequence_gap: bool,
) -> torch.Tensor:
    src, dst = edge_index
    direction = pos[src] - pos[dst]
    distance = torch.norm(direction, dim=-1, keepdim=True)

    if edge_attr_mode == "legacy":
        distance_norm = distance / radius
        cosine = F.cosine_similarity(pos[src], pos[dst], dim=-1).unsqueeze(-1)
        return torch.cat([distance_norm, (cosine + 1.0) / 2.0], dim=-1)

    if edge_attr_mode != "rbf_seq_gap":
        raise ValueError(f"Unsupported edge_attr_mode: {edge_attr_mode}")

    features = [build_rbf_features(distance, edge_distance_rbf_centers)]
    if edge_include_sequence_gap:
        denom = max(pos.size(0) - 1, 1)
        seq_gap = (src - dst).abs().float().unsqueeze(-1) / float(denom)
        features.append(seq_gap)

    return torch.cat(features, dim=-1)


class ProteinGraphDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, psepos_path: str, geometry_config: Dict = None):
        self.names = dataframe["ID"].values
        self.sequences = dataframe["sequence"].values
        self.labels = dataframe["label"].values
        self.residue_psepos = load_pickle(psepos_path)
        self.geometry_settings = resolve_geometry_settings(geometry_config)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        labels = torch.tensor(np.asarray(self.labels[index]), dtype=torch.float32)

        pos = center_positions(
            self.residue_psepos[sequence_name],
            center_mode=self.geometry_settings["pos_center_mode"],
        )
        radial = torch.sqrt(torch.sum(pos * pos, dim=1, keepdim=True)) / CONFIG.dist_norm

        node_features = torch.from_numpy(load_sequence_features(sequence_name))
        node_features = torch.cat([node_features, radial], dim=-1)

        esm2_feature = load_optional_esm2(sequence_name)
        if esm2_feature is not None:
            if esm2_feature.shape[0] == node_features.size(0) + 1:
                esm2_feature = esm2_feature[:-1, :]
            if esm2_feature.shape[0] != node_features.size(0):
                raise ValueError(
                    f"ESM2 length mismatch for {sequence_name}: "
                    f"got {esm2_feature.shape[0]}, expected {node_features.size(0)}"
                )
            esm2_tensor = torch.from_numpy(esm2_feature)
            esm2_norm = esm2_tensor.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            esm2_tensor = esm2_tensor / esm2_norm
            node_features = torch.cat([node_features, esm2_tensor], dim=-1)

        map_cutoff = float(self.geometry_settings["map_cutoff"])
        max_spatial_neighbors = self.geometry_settings["max_spatial_neighbors"]
        edge_index = calculate_edge_index(
            sequence_name,
            map_cutoff,
            max_neighbors=max_spatial_neighbors,
        )
        edge_attr = calculate_edge_attr(
            edge_index=edge_index,
            pos=pos,
            radius=map_cutoff,
            edge_attr_mode=self.geometry_settings["edge_attr_mode"],
            edge_distance_rbf_centers=self.geometry_settings["edge_distance_rbf_centers"],
            edge_include_sequence_gap=self.geometry_settings["edge_include_sequence_gap"],
        )

        return {
            "name": sequence_name,
            "sequence": sequence,
            "labels": labels,
            "node_features": node_features,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }


def single_item_collate(batch):
    return batch[0]
