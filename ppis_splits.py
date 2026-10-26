import json
import os
from typing import Dict, Iterable, List, Tuple

from sklearn.model_selection import KFold


SPLIT_SCHEMA_VERSION = 1


def dataframe_ids(dataframe) -> List[str]:
    return [str(item) for item in dataframe["ID"].tolist()]


def build_kfold_splits(dataframe, dataset_name: str, n_folds: int, seed: int) -> Dict[str, object]:
    ids = dataframe_ids(dataframe)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    records = []
    for fold, (train_index, valid_index) in enumerate(kfold.split(dataframe), start=1):
        train_indices = [int(index) for index in train_index.tolist()]
        valid_indices = [int(index) for index in valid_index.tolist()]
        records.append(
            {
                "fold": int(fold),
                "train_indices": train_indices,
                "valid_indices": valid_indices,
                "train_ids": [ids[index] for index in train_indices],
                "valid_ids": [ids[index] for index in valid_indices],
            }
        )

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": "KFold(n_splits=n_folds, shuffle=True, random_state=seed)",
        "dataset": dataset_name,
        "n_samples": len(ids),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "ordered_ids": ids,
        "fold_records": records,
    }


def save_splits(splits: Dict[str, object], path: str) -> str:
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    return output_path


def load_splits(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _indices_from_ids(ids: Iterable[str], id_to_index: Dict[str, int], fold: int, split_name: str) -> List[int]:
    indices = []
    missing = []
    for protein_id in ids:
        protein_id = str(protein_id)
        if protein_id not in id_to_index:
            missing.append(protein_id)
        else:
            indices.append(id_to_index[protein_id])
    if missing:
        raise ValueError(f"Fold {fold} {split_name} split contains IDs not present in dataframe: {missing[:5]}")
    return indices


def iter_split_indices(splits: Dict[str, object], dataframe) -> Iterable[Tuple[int, List[int], List[int]]]:
    ids = dataframe_ids(dataframe)
    id_to_index = {protein_id: index for index, protein_id in enumerate(ids)}
    if len(id_to_index) != len(ids):
        raise ValueError("Dataframe contains duplicate protein IDs; fixed split lookup would be ambiguous")

    records = splits.get("fold_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Split file is missing non-empty fold_records")

    for record in records:
        fold = int(record["fold"])
        if "train_ids" in record and "valid_ids" in record:
            train_index = _indices_from_ids(record["train_ids"], id_to_index, fold, "train")
            valid_index = _indices_from_ids(record["valid_ids"], id_to_index, fold, "valid")
        else:
            train_index = [int(index) for index in record["train_indices"]]
            valid_index = [int(index) for index in record["valid_indices"]]
            max_index = max(train_index + valid_index)
            if max_index >= len(ids):
                raise ValueError(f"Fold {fold} contains index {max_index}, but dataframe has only {len(ids)} rows")

        if set(train_index) & set(valid_index):
            raise ValueError(f"Fold {fold} train and valid indices overlap")
        if len(train_index) + len(valid_index) != len(ids):
            raise ValueError(f"Fold {fold} does not cover all proteins exactly once")
        yield fold, train_index, valid_index


def ensure_split_file(
    dataframe,
    dataset_name: str,
    n_folds: int,
    seed: int,
    path: str,
) -> Tuple[Dict[str, object], str, bool]:
    output_path = os.path.abspath(path)
    if os.path.exists(output_path):
        splits = load_splits(output_path)
        records = splits.get("fold_records")
        if not isinstance(records, list) or len(records) != int(n_folds):
            raise ValueError(
                f"Split file {output_path} has {len(records) if isinstance(records, list) else 0} folds, "
                f"but --folds is {n_folds}"
            )
        list(iter_split_indices(splits, dataframe))
        return splits, output_path, False

    splits = build_kfold_splits(dataframe, dataset_name=dataset_name, n_folds=n_folds, seed=seed)
    save_splits(splits, output_path)
    return splits, output_path, True
