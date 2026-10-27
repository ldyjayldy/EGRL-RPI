# EGRL-RPI

EGRL: Edge Generation-Guided Relation-Aware Learning for RNA-Protein Interaction Prediction.

## Structure

- `egrl/` : common code
- `experiments/` : experiment entry points
- `data/` : data directory
- `Supplementary Material/` : query-level evaluation records and fold-level predictions

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
python ablations.py
python hyperparameter_1.py
python hyperparameter_2.py
```
