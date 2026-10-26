## Environment

The reference Conda environment is named `VDG2` and uses Python 3.10.20. Recreate it with:

```powershell
conda env create -f environment.yml
conda activate VDG2
```

CUDA 12.1-compatible PyTorch and PyG wheels are specified in `requirements.txt`. A supported NVIDIA GPU is strongly recommended for training and full evaluation.

## Data layout

The four dataset pickles and all per-protein inputs required by the two entry points are included:

- `Dataset/`: `Train_335.pkl` and the three evaluation datasets.
- `Feature/{pssm,hmm,dssp,resAF,distance_map_SC,psepos,SaProt-WslFoldseek}/`: features for the 706 unique proteins used across those datasets.Feature can download from: https://pan.baidu.com/s/1xAY58R1z4stsc426TLmh5g?pwd=6a3s
- `ppis_train335_seed2026_folds5.json`: the fixed five-fold split (seed 2026).
- `runs/20260725-110935/`: the six final base checkpoints, OOF meta features, and training summary needed for checkpoint-based evaluation.

Paths are relative to the repository root, so run all commands from that directory.
