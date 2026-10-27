# EGRL-RPI

EGRL: Edge Generation-Guided Relation-Aware Learning for RNA-Protein Interaction Prediction.

## Structure

- `egrl/` : Common code
- `experiments/` : experiment entry points
- `data/` : Data directory
- `results/` : Output directory

## Installation

`pip install -r requirements.txt`

## Running

```bash
cd experiments
python standard_5fold_cv.py
python coldstart_molecule_holdout.py
python coldstart_cluster_holdout.py
python relation_negative_complexity.py
python soft_edge_quality.py
```
