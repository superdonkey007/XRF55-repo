# XRF55 mmWave-focused workspace

This workspace is organized around the mmWave-only pipeline while preserving the original multimodal DML code under `legacy/`.

## Main entrypoints

- `mmwave_train.py`: mmWave-only classifier training
- `mmwave_eval.py`: mmWave-only classifier evaluation
- `mmwave_dataset.py`: dedicated mmWave dataset loader
- `XRFDataset.py`: original multimodal dataset definitions kept for legacy DML scripts

## Legacy code

The original multimodal workflow remains directly runnable from `legacy/`:

- `legacy/dml_train.py`
- `legacy/dml_eval.py`
- `legacy/generate_txt.py`
- `legacy/split_train_test.py`
- `legacy/model/`
- `legacy/hardware tutorial/`

## Dataset assumptions

The scripts continue to use the original dataset locations under `/root/autodl-tmp/dataset/`.
