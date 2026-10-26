"""Evaluate the reproducible VDG-PPIS 4L/4VN/SH0.95 checkpoint ensemble."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader

from ppis_config import CONFIG
from ppis_data import ProteinGraphDataset, load_dataframe, single_item_collate
from ppis_losses import compute_metrics, sigmoid_probs
from ppis_model import PPISModel

try:
    import joblib
except ImportError:
    joblib = None

SCRIPT_DIR = Path(__file__).resolve().parent

BASE_MODEL_DIRNAME = "base_models"
SUMMARY_FILENAME = "train_4l_4vn_sh095_summary.json"
LEGACY_SUMMARY_FILENAME = SUMMARY_FILENAME
OOF_FEATURE_FILENAME = "oof_meta_features.pt"
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "20260725-110935"
DEFAULT_DATASETS = ["Test_60.pkl", "UBtest_31-6.pkl", "Test_315-28.pkl"]
DEFAULT_TEST_SPECS = [
    {"dataset": "Test_60.pkl", "psepos": os.path.join("Feature", "psepos", "Test60_psepos_SC.pkl")},
    {"dataset": "UBtest_31-6.pkl", "psepos": os.path.join("Feature", "psepos", "UBtest31-6_psepos_SC.pkl")},
    {"dataset": "Test_315-28.pkl", "psepos": os.path.join("Feature", "psepos", "Test315-28_psepos_SC.pkl")},
]
DATASET_PSEPOS_MAP = {spec["dataset"]: spec["psepos"] for spec in DEFAULT_TEST_SPECS}
MODEL_BASENAME = "refit_4l4vn_hgb_3p533_onepass2"
FIXED_POSITIVE_WEIGHT = 3.533
FIXED_MAX_ITER = 110
FIXED_LEARNING_RATE = 0.00977
FIXED_MAX_LEAF_NODES = 18
FIXED_L2_REGULARIZATION = 117.18
FIXED_MIN_SAMPLES_LEAF = 15
FIXED_MAX_BINS = 64
FIXED_MAX_DEPTH = None
FIXED_THRESHOLD = 0.5

def select_features(node_features: torch.Tensor, feature_mode: str, esm_chunks: int) -> torch.Tensor:
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
    if feature_mode.startswith("sap_chunk"):
        chunk_index = int(feature_mode.replace("sap_chunk", ""))
        return torch.chunk(esm2, chunks=esm_chunks, dim=-1)[chunk_index]
    if feature_mode.startswith("saprot_chunk"):
        chunk_index = int(feature_mode.replace("saprot_chunk", ""))
        return torch.chunk(esm2, chunks=esm_chunks, dim=-1)[chunk_index]
    if feature_mode.startswith("esm_chunk"):
        chunk_index = int(feature_mode.replace("esm_chunk", ""))
        return torch.chunk(esm2, chunks=esm_chunks, dim=-1)[chunk_index]

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")

def load_summary(run_dir: str):
    summary_path = os.path.join(run_dir, SUMMARY_FILENAME)
    if not os.path.exists(summary_path):
        legacy_summary_path = os.path.join(run_dir, LEGACY_SUMMARY_FILENAME)
        if os.path.exists(legacy_summary_path):
            summary_path = legacy_summary_path
        else:
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f), summary_path


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
        ipr_virtual_message_layers=checkpoint_config.get(
            "ipr_virtual_message_layers",
            CONFIG.ipr_virtual_message_layers,
        ),
        ipr_aux_entropy_weight=checkpoint_config.get("ipr_aux_entropy_weight", CONFIG.ipr_aux_entropy_weight),
        ipr_aux_balance_weight=checkpoint_config.get("ipr_aux_balance_weight", CONFIG.ipr_aux_balance_weight),
        ipr_aux_compact_weight=checkpoint_config.get("ipr_aux_compact_weight", CONFIG.ipr_aux_compact_weight),
        ipr_aux_smooth_weight=checkpoint_config.get("ipr_aux_smooth_weight", CONFIG.ipr_aux_smooth_weight),
        ipr_aux_score_l2_weight=checkpoint_config.get("ipr_aux_score_l2_weight", CONFIG.ipr_aux_score_l2_weight),
        encoder_type=checkpoint_config.get("encoder_type", CONFIG.encoder_type),
        residue_num_heads=checkpoint_config.get("residue_num_heads", CONFIG.residue_num_heads),
        residue_lmax=checkpoint_config.get("residue_lmax", CONFIG.residue_lmax),
        residue_sh_cutoff=checkpoint_config.get("residue_sh_cutoff", CONFIG.residue_sh_cutoff),
        residue_update_sh_neighbors=checkpoint_config.get(
            "residue_update_sh_neighbors",
            CONFIG.residue_update_sh_neighbors,
        ),
        residue_update_eu_neighbors=checkpoint_config.get(
            "residue_update_eu_neighbors",
            CONFIG.residue_update_eu_neighbors,
        ),
        residue_use_sh_branch=checkpoint_config.get("residue_use_sh_branch", CONFIG.residue_use_sh_branch),
        residue_use_cross_space=checkpoint_config.get("residue_use_cross_space", CONFIG.residue_use_cross_space),
        residue_cross_space_gate=checkpoint_config.get("residue_cross_space_gate", CONFIG.residue_cross_space_gate),
        residue_cross_space_semantic=checkpoint_config.get(
            "residue_cross_space_semantic",
            CONFIG.residue_cross_space_semantic,
        ),
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


def move_batch_to_device(batch):
    return {key: value.to(CONFIG.device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_ensemble(models, meta_model, loader, saprot_chunks: int):
    all_meta_features = []
    all_labels = []
    residue_scores = {}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch)
            protein_logits = []
            protein_base_probs = {}
            for item in models:
                features = select_features(
                    batch["node_features"],
                    item["feature_mode"],
                    saprot_chunks,
                )
                logits = item["model"](
                    features,
                    batch["pos"],
                    batch["edge_index"],
                    batch["edge_attr"],
                ).view(-1)
                protein_logits.append(logits.detach().cpu().numpy().reshape(-1, 1))
                protein_base_probs[item["name"]] = sigmoid_probs(logits).detach().cpu().tolist()

            meta_features = np.concatenate(protein_logits, axis=1)
            labels = batch["labels"].detach().cpu().int().numpy().reshape(-1)
            probabilities = meta_model.predict_proba(meta_features)[:, 1]

            all_meta_features.append(meta_features)
            all_labels.append(labels)
            residue_scores[batch["name"]] = {
                "labels": labels.tolist(),
                "probabilities": probabilities.tolist(),
                "base_probabilities": protein_base_probs,
            }

    x_meta = np.concatenate(all_meta_features, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_score = meta_model.predict_proba(x_meta)[:, 1]
    return x_meta, y_true, y_score, residue_scores


def resolve_test_specs(dataset: str = None, psepos: str = None):
    if dataset is None:
        if psepos is not None:
            raise ValueError("--psepos can only be used together with --dataset")
        return DEFAULT_TEST_SPECS

    resolved_psepos = psepos or DATASET_PSEPOS_MAP.get(dataset)
    if resolved_psepos is None:
        raise ValueError(f"No default psepos mapping for {dataset}; pass --psepos explicitly")
    return [{"dataset": dataset, "psepos": resolved_psepos}]

def save_meta_model(model, path: str) -> None:
    if joblib is not None:
        joblib.dump(model, path)
        return
    import pickle

    with open(path, "wb") as f:
        pickle.dump(model, f)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run 4-layer 4VN/SH0.95 base models and the fixed HGB meta "
            "learner in one pass."
        )
    )
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DEFAULT_DATASETS,
        default=None,
        help="Dataset pickle name. Repeat to evaluate multiple datasets.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no_save_meta", action="store_true")
    parser.add_argument("--no_save_cache", action="store_true")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--saprot_dir", "--esm2_dir", dest="esm2_dir", default=None)
    parser.add_argument(
        "--saprot_dim", "--esm2_dim", dest="esm2_dim", type=int, default=None
    )
    return parser.parse_args()


def meta_config(seed: int):
    return {
        "type": "HistGradientBoostingClassifier",
        "positive_sample_weight": FIXED_POSITIVE_WEIGHT,
        "max_iter": FIXED_MAX_ITER,
        "learning_rate": FIXED_LEARNING_RATE,
        "max_leaf_nodes": FIXED_MAX_LEAF_NODES,
        "l2_regularization": FIXED_L2_REGULARIZATION,
        "min_samples_leaf": FIXED_MIN_SAMPLES_LEAF,
        "max_bins": FIXED_MAX_BINS,
        "max_depth": FIXED_MAX_DEPTH,
        "early_stopping": False,
        "random_state": seed,
        "posthoc_intercept_shift": 0.0,
        "fixed_threshold": FIXED_THRESHOLD,
    }


def build_meta_learner(seed: int):
    return HistGradientBoostingClassifier(
        max_iter=FIXED_MAX_ITER,
        learning_rate=FIXED_LEARNING_RATE,
        max_leaf_nodes=FIXED_MAX_LEAF_NODES,
        l2_regularization=FIXED_L2_REGULARIZATION,
        min_samples_leaf=FIXED_MIN_SAMPLES_LEAF,
        max_bins=FIXED_MAX_BINS,
        max_depth=FIXED_MAX_DEPTH,
        early_stopping=False,
        random_state=seed,
    )


def resolve_base_model_paths(run_dir: Path, summary: dict):
    configured_dir = summary.get("base_model_dir")
    if configured_dir:
        # Historical summaries were written on Windows and contain backslashes.
        # Normalize both separator styles so the retained run is portable.
        portable_dir = str(configured_dir).replace("\\", os.sep).replace("/", os.sep)
        base_model_dir = Path(portable_dir)
        if not base_model_dir.is_absolute():
            candidates = [
                Path.cwd() / base_model_dir,
                SCRIPT_DIR / base_model_dir,
                run_dir / base_model_dir,
            ]
            base_model_dir = next(
                (candidate for candidate in candidates if candidate.is_dir()),
                candidates[0],
            )
    else:
        base_model_dir = run_dir / BASE_MODEL_DIRNAME

    model_paths = []
    for config in summary["base_model_configs"]:
        path = base_model_dir / config["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing base model {config['name']}: {path}")
        model_paths.append((config, path))
    return model_paths


def fit_meta_learner(run_dir: Path, seed: int):
    oof_path = run_dir / OOF_FEATURE_FILENAME
    if not oof_path.is_file():
        raise FileNotFoundError(f"Missing OOF meta features: {oof_path}")
    oof = torch.load(oof_path, map_location="cpu")
    x_meta = as_numpy(oof["x_meta"])
    y_meta = as_numpy(oof["y_meta"]).astype(np.int64).reshape(-1)
    model = build_meta_learner(seed)
    sample_weight = np.where(y_meta == 1, FIXED_POSITIVE_WEIGHT, 1.0)
    model.fit(x_meta, y_meta, sample_weight=sample_weight)
    return model, oof_path


def load_base_models(run_dir: Path, summary: dict, datasets, args):
    model_paths = resolve_base_model_paths(run_dir, summary)
    first_checkpoint = torch.load(
        model_paths[0][1], map_location=CONFIG.device
    )
    geometry_config = first_checkpoint.get("config", {})
    summary_args = summary.get("args", {})
    CONFIG.esm2_dir = (
        args.esm2_dir
        or summary_args.get("esm2_dir")
        or geometry_config.get("esm2_dir", CONFIG.esm2_dir)
    )
    CONFIG.esm2_dim = int(
        args.esm2_dim
        or summary_args.get("esm2_dim")
        or geometry_config.get("esm2_dim", CONFIG.esm2_dim)
    )

    first_spec = resolve_test_specs(datasets[0], None)[0]
    sample_dataframe = load_dataframe(first_spec["dataset"])
    sample_dataset = ProteinGraphDataset(
        sample_dataframe,
        psepos_path=first_spec["psepos"],
        geometry_config=geometry_config,
    )
    sample = sample_dataset[0]

    models = []
    for base_config, checkpoint_path in model_paths:
        model, checkpoint, checkpoint_config = load_base_model(
            str(checkpoint_path), sample
        )
        models.append(
            {
                "name": checkpoint_config.get(
                    "base_model_name", base_config["name"]
                ),
                "feature_mode": checkpoint_config.get(
                    "feature_mode", base_config["feature_mode"]
                ),
                "checkpoint": str(checkpoint_path),
                "epoch": checkpoint.get("epoch"),
                "model": model,
            }
        )
        print(
            f"Loaded {models[-1]['name']} | epoch={models[-1]['epoch']} | "
            f"{checkpoint_path}"
        )
    return models, geometry_config


def evaluate_dataset(
    run_dir: Path,
    dataset_name: str,
    geometry_config: dict,
    models,
    meta_model,
    saprot_chunks: int,
    num_workers: int,
    save_cache: bool,
):
    spec = resolve_test_specs(dataset_name, None)[0]
    dataframe = load_dataframe(dataset_name)
    dataset = ProteinGraphDataset(
        dataframe,
        psepos_path=spec["psepos"],
        geometry_config=geometry_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=single_item_collate,
    )
    x_meta, labels, probabilities, residue_scores = run_ensemble(
        models, meta_model, loader, saprot_chunks
    )
    result_metrics = compute_metrics(
        labels.tolist(), probabilities.tolist(), threshold=FIXED_THRESHOLD
    )
    predictions = (probabilities >= FIXED_THRESHOLD).astype(np.int64)

    cache_path = None
    if save_cache:
        cache_path = run_dir / f"eval_4l4vn_refit2_{Path(dataset_name).stem}.pt"
        torch.save(
            {
                "metrics_fixed_threshold": result_metrics,
                "metrics": result_metrics,
                "threshold": FIXED_THRESHOLD,
                "test_thresholds": [FIXED_THRESHOLD],
                "meta_features": x_meta,
                "labels": labels,
                "probabilities": probabilities,
                "predictions_fixed_threshold": predictions,
                "predictions": predictions,
                "residue_scores": residue_scores,
                "run_dir": str(run_dir),
                "dataset": dataset_name,
                "psepos": spec["psepos"],
                "meta_config": meta_config(meta_model.random_state),
            },
            cache_path,
        )
    return {
        "dataset": dataset_name,
        "cached_eval_artifact": str(cache_path) if cache_path else None,
        "num_residues": int(labels.size),
        "threshold": FIXED_THRESHOLD,
        "metrics_at_threshold_0p5": result_metrics,
    }


def main():
    args = parse_args()
    CONFIG.egnn_layers = 4
    CONFIG.ipr_num_virtual_nodes = 4
    CONFIG.residue_sh_cutoff = 0.95
    CONFIG.use_esm2 = True

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    summary, summary_path = load_summary(str(run_dir))
    datasets = args.dataset or DEFAULT_DATASETS

    meta_model, oof_path = fit_meta_learner(run_dir, args.seed)
    model_path = None
    if not args.no_save_meta:
        model_path = run_dir / f"{MODEL_BASENAME}.joblib"
        save_meta_model(meta_model, str(model_path))

    models, geometry_config = load_base_models(run_dir, summary, datasets, args)
    summary_args = summary.get("args", {})
    saprot_chunks = int(
        summary_args.get("saprot_chunks") or summary_args.get("esm_chunks", 4)
    )

    print(f"Using run_dir: {run_dir}")
    print(f"Using summary: {summary_path}")
    results = []
    for dataset_name in datasets:
        print(f"\nRunning six base branches: {dataset_name}")
        result = evaluate_dataset(
            run_dir=run_dir,
            dataset_name=dataset_name,
            geometry_config=geometry_config,
            models=models,
            meta_model=meta_model,
            saprot_chunks=saprot_chunks,
            num_workers=args.num_workers,
            save_cache=not args.no_save_cache,
        )
        results.append(result)
        print(f"========== {dataset_name} | threshold=0.50 ==========")
        for name in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "auroc",
            "auprc",
        ]:
            print(f"{name}: {result['metrics_at_threshold_0p5'][name]:.6f}")

    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else run_dir / f"{MODEL_BASENAME}_threshold0p5_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output = {
        "variant": "4l_4vn_sh095_refit2",
        "execution_mode": "one_pass_base_inference_and_refit_meta_prediction",
        "run_dir": str(run_dir),
        "training_summary": summary_path,
        "oof_meta_features": str(oof_path),
        "refit_meta_learner": str(model_path) if model_path else None,
        "meta_config": meta_config(args.seed),
        "fixed_threshold": FIXED_THRESHOLD,
        "datasets": datasets,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_builtin(summary_output),
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved threshold-0.5 summary: {output_path}")

if __name__ == "__main__":
    main()
