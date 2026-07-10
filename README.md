# IR Carbonyl Graph

This project predicts carbonyl-group IR characteristic peaks from molecular SMILES. The workflow converts carbonyl molecules into PyTorch Geometric graphs, trains a node-environment GNN (NE-GNN), and keeps utilities for model interpretation and tree-model baselines.

## Refactor Overview

The original notebook export, `module/NE-GNN workflow.py`, mixed data cleaning, graph construction, GNN definitions, training, explanation plots, and baseline feature engineering in one large script.

The reusable parts are now split into focused modules under `module/ne_gnn/`. The original exported script is kept as experiment history. Use `module/NE_GNN_workflow.py` as the cleaner command-line training entry point.

## Project Tree

```text
IR-Carbonyl-Graph-main/
+-- readme.md
+-- dataset/
|   +-- carbonyl_group_from_Nist.xlsx
|   +-- CIAC_carbonyl_group.xlsx
|   +-- CIAC_carbonyl_group_extended.xlsx
|   +-- experiment_three_carbonyl_groups.xlsx
|   +-- experiment_two_carbonyl_groups.xlsx
+-- model/
|   +-- one_node_0.pt
|   +-- one_node_1.pt
|   +-- one_node_2.pt
|   +-- one_node_3.pt
|   +-- two_nodes_0.pt
|   +-- two_nodes_1.pt
|   +-- two_nodes_2.pt
|   +-- two_nodes_3.pt
+-- module/
    +-- Extract_carbonyl_data.ipynb
    +-- GNNexplainer.py
    +-- NE-GNN workflow.ipynb
    +-- NE-GNN workflow.py
    +-- NE_GNN_workflow.py
    +-- ne_gnn/
        +-- __init__.py
        +-- constants.py
        +-- explainability.py
        +-- features.py
        +-- graph_dataset.py
        +-- models.py
        +-- preprocessing.py
        +-- splitting.py
        +-- training.py
        +-- tree_features.py
```

## Module Guide

- `module/ne_gnn/preprocessing.py`: loads Excel/CSV data, canonicalizes SMILES, and deduplicates molecules by median IR peak.
- `module/ne_gnn/features.py`: extracts atom, carbonyl-mask, and local environment features with RDKit.
- `module/ne_gnn/graph_dataset.py`: builds `torch_geometric.data.Data` objects through `CarbonylIRDataset`.
- `module/ne_gnn/models.py`: contains the NE-GNN graph encoder, edge-aware convolution variants, and regression head.
- `module/ne_gnn/splitting.py`: provides random and Bemis-Murcko scaffold splits.
- `module/ne_gnn/training.py`: contains training, prediction, and metric helpers.
- `module/ne_gnn/explainability.py`: contains perturbation-based signed node-feature importance utilities.
- `module/ne_gnn/tree_features.py`: contains tabular feature engineering for baseline tree models.
- `module/NE_GNN_workflow.py`: command-line training entry point using the modular package.

## Data Format

The training table is expected to include:

- `SMILES`: molecular structure string.
- `IR_Characteristic_Peak`: target carbonyl IR peak.
- `DOI`: optional metadata field.

During preprocessing, `Canonical_SMILES` is generated automatically.

## Basic Usage

Install the scientific Python stack required by the workflow:

```bash
pip install numpy pandas matplotlib scikit-learn torch torch-geometric rdkit
```

Train a model from the default extended dataset:

```bash
python module/NE_GNN_workflow.py --data dataset/CIAC_carbonyl_group_extended.xlsx --epochs 300
```

Useful options:

```bash
python module/NE_GNN_workflow.py \
  --data dataset/CIAC_carbonyl_group_extended.xlsx \
  --epochs 500 \
  --batch-size 128 \
  --lr 0.001 \
  --device cuda \
  --output model/ne_gnn_latest.pt
```

The checkpoint stores model weights, training-set normalization statistics, test metrics, and run arguments.

standardized parameters:

```
train_dataset.mean , train_dataset.std = 1691.8188449889867 ,  41.20469330934477
```

## Notes

- `model/*.pt` contains existing trained checkpoints from earlier experiments.
- `module/NE-GNN workflow.ipynb` and `module/NE-GNN workflow.py` are notebook-era experiment artifacts and may contain absolute local paths.
- The modular entry point avoids hard-coded external paths and defaults to files inside this repository.
