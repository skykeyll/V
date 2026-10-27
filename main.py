"""Float64 test inference with the fixed 20261020-120930 models and fitted HGB.

Numerical policy matches diagnose_double_precision_rotation.py: load the
original float32 model first, then call double(); retain preprocessed feature
and centered-coordinate values, and recompute edge attributes in float64.
No training, HGB refitting, float32 inference, or rotation tests are performed.
This entry point uses the ppis_* core modules without importing test.py or train.py.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import torch

from ppis_config import CONFIG, set_seed
from ppis_data import (
    ProteinGraphDataset,
    build_rbf_features,
    calculate_edge_attr,
    load_dataframe,
    resolve_geometry_settings,
)
from ppis_losses import compute_metrics
from ppis_model import PPISModel
from ppis_repro import sha256_file


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR / "runs" / "20261020-120930"
SUMMARY_PATH = RUN_DIR / "train_4l_4vn_sh095_summary.json"
META_PATH = RUN_DIR / "refit_4l4vn_hgb_3p533_onepass2.joblib"
DATASETS = ("Test_60.pkl", "UBtest_31-6.pkl", "Test_315-28.pkl")
DATASET_PSEPOS_MAP = {
    "Test_60.pkl": os.path.join("Feature", "psepos", "Test60_psepos_SC.pkl"),
    "UBtest_31-6.pkl": os.path.join("Feature", "psepos", "UBtest31-6_psepos_SC.pkl"),
    "Test_315-28.pkl": os.path.join("Feature", "psepos", "Test315-28_psepos_SC.pkl"),
}
METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc")
THRESHOLD = 0.5


def resolve_test_specs(dataset: str = None, psepos: str = None):
    if dataset is None:
        if psepos is not None:
            raise ValueError("A psepos override requires a dataset")
        return [{"dataset": name, "psepos": DATASET_PSEPOS_MAP[name]} for name in DATASETS]
    resolved_psepos = psepos or DATASET_PSEPOS_MAP.get(dataset)
    if resolved_psepos is None:
        raise ValueError(f"No default psepos mapping for {dataset}; provide psepos explicitly")
    return [{"dataset": dataset, "psepos": resolved_psepos}]


def move_batch_to_device(batch):
    return {
        key: value.to(CONFIG.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def select_features(node_features: torch.Tensor, feature_mode: str, esm_chunks: int) -> torch.Tensor:
    # Keep reference float32 preprocessing before the caller promotes to float64.
    node_features = node_features.to(CONFIG.device).float()
    if node_features.dim() == 3:
        node_features = node_features.squeeze(0)

    feature_dim = node_features.size(-1)
    handcrafted_dim = CONFIG.handcrafted_dim
    full_dim = CONFIG.handcrafted_dim + CONFIG.esm2_dim
    if feature_dim == handcrafted_dim:
        handcrafted = node_features
        esm2 = None
    elif feature_dim == full_dim:
        handcrafted = node_features[:, :handcrafted_dim]
        esm2 = node_features[:, handcrafted_dim:]
    else:
        raise ValueError(f"Unexpected node feature dim: {feature_dim}; expected {handcrafted_dim} or {full_dim}")

    if feature_mode == "handcrafted":
        return handcrafted
    if esm2 is None:
        raise ValueError(f"{feature_mode} requires CONFIG.use_esm2=True and residue language-model node features")
    if feature_mode == "all_features":
        return torch.cat([handcrafted, esm2], dim=-1)
    for prefix in ("sap_chunk", "saprot_chunk", "esm_chunk"):
        if feature_mode.startswith(prefix):
            chunk_index = int(feature_mode[len(prefix):])
            return torch.chunk(esm2, chunks=esm_chunks, dim=-1)[chunk_index]
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def load_base_model(checkpoint_path: str, sample) -> tuple:
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG.device)
    checkpoint_config = checkpoint.get("config", {})
    model = PPISModel(
        input_dim=checkpoint_config.get("input_dim", sample["node_features"].size(-1)),
        edge_attr_dim=checkpoint_config.get("edge_attr_dim", sample["edge_attr"].size(-1)),
        hidden_dim=checkpoint_config.get("hidden_dim", CONFIG.hidden_dim),
        egnn_layers=checkpoint_config.get("egnn_layers", CONFIG.egnn_layers),
        dropout=checkpoint_config.get("dropout", CONFIG.dropout),
        use_ipr_virtual_nodes=checkpoint_config.get("use_ipr_virtual_nodes", CONFIG.use_ipr_virtual_nodes),
        ipr_num_virtual_nodes=checkpoint_config.get("ipr_num_virtual_nodes", CONFIG.ipr_num_virtual_nodes),
        ipr_temperature=checkpoint_config.get("ipr_temperature", CONFIG.ipr_temperature),
        ipr_sinkhorn_iterations=checkpoint_config.get("ipr_sinkhorn_iterations", CONFIG.ipr_sinkhorn_iterations),
        ipr_eval_hard=checkpoint_config.get("ipr_eval_hard", CONFIG.ipr_eval_hard),
        ipr_scorer_context_layers=checkpoint_config.get("ipr_scorer_context_layers", CONFIG.ipr_scorer_context_layers),
        ipr_virtual_message_layers=checkpoint_config.get("ipr_virtual_message_layers", CONFIG.ipr_virtual_message_layers),
        ipr_aux_entropy_weight=checkpoint_config.get("ipr_aux_entropy_weight", CONFIG.ipr_aux_entropy_weight),
        ipr_aux_balance_weight=checkpoint_config.get("ipr_aux_balance_weight", CONFIG.ipr_aux_balance_weight),
        ipr_aux_compact_weight=checkpoint_config.get("ipr_aux_compact_weight", CONFIG.ipr_aux_compact_weight),
        ipr_aux_smooth_weight=checkpoint_config.get("ipr_aux_smooth_weight", CONFIG.ipr_aux_smooth_weight),
        ipr_aux_score_l2_weight=checkpoint_config.get("ipr_aux_score_l2_weight", CONFIG.ipr_aux_score_l2_weight),
        encoder_type=checkpoint_config.get("encoder_type", CONFIG.encoder_type),
        residue_num_heads=checkpoint_config.get("residue_num_heads", CONFIG.residue_num_heads),
        residue_lmax=checkpoint_config.get("residue_lmax", CONFIG.residue_lmax),
        residue_sh_cutoff=checkpoint_config.get("residue_sh_cutoff", CONFIG.residue_sh_cutoff),
        residue_update_sh_neighbors=checkpoint_config.get("residue_update_sh_neighbors", CONFIG.residue_update_sh_neighbors),
        residue_update_eu_neighbors=checkpoint_config.get("residue_update_eu_neighbors", CONFIG.residue_update_eu_neighbors),
        residue_use_sh_branch=checkpoint_config.get("residue_use_sh_branch", CONFIG.residue_use_sh_branch),
        residue_use_cross_space=checkpoint_config.get("residue_use_cross_space", CONFIG.residue_use_cross_space),
        residue_cross_space_gate=checkpoint_config.get("residue_cross_space_gate", CONFIG.residue_cross_space_gate),
        residue_cross_space_semantic=checkpoint_config.get("residue_cross_space_semantic", CONFIG.residue_cross_space_semantic),
        residue_eu_cutoff=checkpoint_config.get("residue_eu_cutoff", CONFIG.residue_eu_cutoff),
        residue_max_neighbors=checkpoint_config.get("residue_max_neighbors", CONFIG.residue_max_neighbors),
    ).to(CONFIG.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    egnn_layers = int(checkpoint_config.get("egnn_layers", -1))
    virtual_nodes = int(checkpoint_config.get("ipr_num_virtual_nodes", -1))
    uses_virtual_nodes = bool(checkpoint_config.get("use_ipr_virtual_nodes", False))
    sh_cutoff = float(checkpoint_config.get("residue_sh_cutoff", float("nan")))
    if egnn_layers != 4 or not uses_virtual_nodes or virtual_nodes != 4 or abs(sh_cutoff - 0.95) > 1e-12:
        raise ValueError(
            "Checkpoint is not a 4L/4VN/SH0.95 model: "
            f"layers={egnn_layers}, virtual_nodes={virtual_nodes}, "
            f"use_virtual_nodes={uses_virtual_nodes}, sh_cutoff={sh_cutoff}: {checkpoint_path}"
        )
    model.eval()
    return model, checkpoint, checkpoint_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", choices=DATASETS,
        help="Repeat to select datasets; by default evaluate all three.",
    )
    parser.add_argument("--device", default=CONFIG.device)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", default="analysis_outputs/test_float64")
    return parser.parse_args()


def geometry_attrs_float64(batch, pos, geometry):
    """Use the corrected diagnostic's geometry formula without float32 gaps."""
    if geometry["edge_attr_mode"] == "legacy":
        return calculate_edge_attr(
            batch["edge_index"], pos,
            radius=float(geometry["map_cutoff"]),
            edge_attr_mode="legacy",
            edge_distance_rbf_centers=list(geometry["edge_distance_rbf_centers"]),
            edge_include_sequence_gap=bool(geometry["edge_include_sequence_gap"]),
        )
    if geometry["edge_attr_mode"] != "rbf_seq_gap":
        raise ValueError(f"Unsupported edge attributes: {geometry['edge_attr_mode']}")
    source, target = batch["edge_index"]
    distance = torch.norm(pos[source] - pos[target], dim=-1, keepdim=True)
    parts = [build_rbf_features(distance, list(geometry["edge_distance_rbf_centers"]))]
    if geometry["edge_include_sequence_gap"]:
        parts.append(
            (source - target).abs().to(pos.dtype).unsqueeze(-1)
            / max(pos.size(0) - 1, 1)
        )
    return torch.cat(parts, dim=-1)


def load_models(summary, sample):
    models = []
    for base_config in summary["base_model_configs"]:
        # Resolve within this fixed run, ignoring stale absolute paths in JSON.
        checkpoint_path = RUN_DIR / "base_models" / base_config["filename"]
        model, _, config = load_base_model(str(checkpoint_path), sample)
        model = model.double().eval()
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            if value.is_floating_point() and value.dtype != torch.float64:
                raise TypeError(f"Non-float64 model tensor: {name}: {value.dtype}")
        models.append({
            "name": config.get("base_model_name", base_config["name"]),
            "feature_mode": config.get("feature_mode", base_config["feature_mode"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model": model,
        })
    if len(models) != 6:
        raise ValueError(f"Expected six base models, found {len(models)}")
    return models


def evaluate_dataset(dataset_name, config, geometry, models, meta_model, chunks, output_dir):
    spec = resolve_test_specs(dataset_name)[0]
    dataset = ProteinGraphDataset(
        load_dataframe(dataset_name), spec["psepos"], geometry_config=config,
    )
    logits_all, labels_all, protein_ids, protein_sizes = [], [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = move_batch_to_device(dataset[index])
            pos = batch["pos"].double()
            attrs = geometry_attrs_float64(batch, pos, geometry)
            protein_logits = []
            for item in models:
                # Preserve the diagnostic's feature preprocessing, then promote.
                features = select_features(
                    batch["node_features"], item["feature_mode"], chunks,
                ).double()
                logits = item["model"](
                    features, pos, batch["edge_index"], attrs,
                ).view(-1)
                if logits.dtype != torch.float64 or not torch.isfinite(logits).all():
                    raise RuntimeError(f"Invalid float64 logits: {batch['name']}")
                protein_logits.append(logits.cpu().numpy())
            logits_all.append(np.column_stack(protein_logits))
            labels_all.append(batch["labels"].cpu().numpy().astype(np.int64))
            protein_ids.append(str(batch["name"]))
            protein_sizes.append(pos.size(0))
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"  {dataset_name}: {index + 1}/{len(dataset)} chains", flush=True)

    meta_features = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all)
    probabilities = meta_model.predict_proba(meta_features)[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"Non-finite HGB probabilities: {dataset_name}")
    metrics = compute_metrics(labels.tolist(), probabilities.tolist(), threshold=THRESHOLD)
    prediction_path = output_dir / f"{Path(dataset_name).stem}_predictions.npz"
    np.savez_compressed(
        prediction_path, meta_features=meta_features, labels=labels,
        probabilities=probabilities, predictions=(probabilities >= THRESHOLD).astype(np.int64),
        protein_ids=np.asarray(protein_ids), protein_sizes=np.asarray(protein_sizes),
    )
    print(f"\n{dataset_name} | chains={len(dataset)} | residues={labels.size} | threshold=0.5")
    for name in METRICS:
        print(f"  {name}: {metrics[name]:.4f}  ({metrics[name]:.12f})")
    return {
        "dataset": dataset_name, "num_chains": len(dataset),
        "num_residues": int(labels.size), "metrics_at_threshold_0p5": metrics,
        "predictions": str(prediction_path), "seconds": time.perf_counter() - started,
    }


def main():
    args = parse_args()
    # Existing feature paths are relative to the project, as in the diagnostic.
    os.chdir(SCRIPT_DIR)
    CONFIG.device = args.device
    CONFIG.use_esm2 = True
    CONFIG.use_amp = False
    set_seed(args.seed)
    torch.set_num_threads(1)
    with SUMMARY_PATH.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    base_configs = summary["base_model_configs"]
    first_path = RUN_DIR / "base_models" / base_configs[0]["filename"]
    config = torch.load(first_path, map_location=CONFIG.device)["config"]
    geometry = resolve_geometry_settings(config)
    summary_args = summary.get("args", {})
    CONFIG.esm2_dir = summary_args.get("esm2_dir") or config.get("esm2_dir", CONFIG.esm2_dir)
    CONFIG.esm2_dim = int(summary_args.get("esm2_dim") or config.get("esm2_dim", CONFIG.esm2_dim))
    CONFIG.handcrafted_dim = int(config.get("handcrafted_dim", CONFIG.handcrafted_dim))
    chunks = int(summary_args.get("saprot_chunks") or summary_args.get("esm_chunks", 4))
    # Use the same construction sample as the reference diagnostic.
    sample_frame = load_dataframe("Test_60.pkl")
    sample_frame = sample_frame[sample_frame["ID"].astype(str) == "7ceiB"].reset_index(drop=True)
    sample_spec = resolve_test_specs("Test_60.pkl")[0]
    sample = move_batch_to_device(
        ProteinGraphDataset(sample_frame, sample_spec["psepos"], geometry_config=config)[0]
    )
    models = load_models(summary, sample)
    meta_hash = sha256_file(META_PATH)
    meta_model = joblib.load(str(META_PATH))
    output_dir = Path(args.output_dir).resolve() / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Fixed training run: {RUN_DIR}")
    print(f"Existing HGB (no refit): {META_PATH}")
    print(f"Float64 test inference; outputs: {output_dir}", flush=True)
    results = []
    for dataset_name in args.dataset or DATASETS:
        results.append(evaluate_dataset(
            dataset_name, config, geometry, models, meta_model, chunks, output_dir,
        ))
    if sha256_file(META_PATH) != meta_hash or any(
        sha256_file(item["checkpoint"]) != item["checkpoint_sha256"] for item in models
    ):
        raise RuntimeError("A source checkpoint or HGB file changed during testing")
    report = {
        "run_dir": str(RUN_DIR), "training_summary": str(SUMMARY_PATH),
        "meta_model": str(META_PATH), "meta_model_sha256": meta_hash,
        "inference_precision": "float64", "threshold": THRESHOLD,
        "hgb_refitted": False, "seed": args.seed, "device": args.device,
        "input_policy": "retain_reference_preprocessed_values_then_promote",
        "model_policy": "load_original_model_then_double_as_in_diagnostic",
        "geometry": geometry,
        "base_models": [{k: v for k, v in item.items() if k != "model"} for item in models],
        "results": results,
    }
    summary_path = output_dir / "test_float64_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([
        {"dataset": result["dataset"], **result["metrics_at_threshold_0p5"]}
        for result in results
    ]).to_csv(output_dir / "metrics.csv", index=False)
    print(f"\nSaved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
