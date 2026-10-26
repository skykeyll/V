import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


SEED = 2025
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class PPISConfig:
    dataset_dir: str = "Dataset"
    feature_dir: str = "Feature"
    output_dir: str = "runs_ppis"

    train_dataset: str = "Train_335.pkl"
    test_dataset: str = "Test_60.pkl"
    train_psepos: str = os.path.join("Feature", "psepos", "Train335_psepos_SC.pkl")
    test_psepos: str = os.path.join("Feature", "psepos", "Test60_psepos_SC.pkl")

    validation_ratio: float = 0.2
    split_seed: int = 2024
    train_seeds: List[int] = field(default_factory=lambda: [2026])
    early_stop_patience: int = 10
    split_strategy: str = "auto"
    split_pos_ratio_bins: int = 4
    split_length_bins: int = 4
    split_min_count_per_stratum: int = 2

    map_cutoff: float = 14.0
    max_spatial_neighbors: Optional[int] = None
    dist_norm: float = 15.0
    pos_center_mode: str = "centroid"
    edge_attr_mode: str = "rbf_seq_gap"
    edge_distance_rbf_centers: List[float] = field(default_factory=lambda: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    edge_include_sequence_gap: bool = True
    input_dim: int = 62
    handcrafted_dim: int = 62
    hidden_dim: int = 64
    egnn_layers: int = 4
    dropout: float = 0.4
    encoder_type: str = "egnn"
    residue_num_heads: int = 4
    residue_lmax: int = 2
    residue_sh_cutoff: float = 0.95
    residue_update_sh_neighbors: bool = True
    residue_update_eu_neighbors: bool = False
    residue_use_sh_branch: bool = True
    residue_use_cross_space: bool = True
    residue_cross_space_gate: bool = False
    residue_cross_space_semantic: bool = False
    residue_eu_cutoff: float = 14.0
    residue_max_neighbors: Optional[int] = None
    use_ipr_virtual_nodes: bool = True
    ipr_num_virtual_nodes: int = 4
    ipr_assignment_mode: str = "balanced_sinkhorn"
    ipr_temperature: float = 1.0
    ipr_sinkhorn_iterations: int = 20
    ipr_eval_hard: bool = False
    ipr_scorer_context_layers: int = 1
    ipr_virtual_message_layers: int = 1
    ipr_aux_loss_weight: float = 1.0
    ipr_aux_entropy_weight: float = 0.001
    ipr_aux_balance_weight: float = 0.001
    ipr_aux_compact_weight: float = 0.01
    ipr_aux_smooth_weight: float = 0.01
    ipr_aux_score_l2_weight: float = 0.0005
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 5e-4
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    focal_weight: float = 0.25
    loss_mode: str = "bce_tversky"
    pos_weight_scale: float = 1.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_weight: float = 0.3
    label_smoothing: float = 0.0
    grad_clip_norm: float = 5.0
    use_amp: bool = False
    deterministic: bool = True
    deterministic_warn_only: bool = True

    use_esm2: bool = False
    esm2_dir: str = os.path.join("Feature", "ESM-2")
    esm2_dim: int = 1280

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    save_train_best: bool = False
    save_last: bool = True
    save_valid_best: bool = True


CONFIG = PPISConfig()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if CONFIG.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=CONFIG.deterministic_warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def build_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
