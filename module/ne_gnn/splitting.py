"""Dataset split helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.9,
    valid_frac: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly split a DataFrame into train, validation, and test sets."""
    indices = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_num = int(len(indices) * train_frac)
    valid_num = int(len(indices) * valid_frac)
    train_index = indices[:train_num]
    valid_index = indices[train_num : train_num + valid_num]
    test_index = indices[train_num + valid_num :]

    return (
        df.iloc[train_index].reset_index(drop=True),
        df.iloc[valid_index].reset_index(drop=True),
        df.iloc[test_index].reset_index(drop=True),
    )


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    """Generate a Bemis-Murcko scaffold SMILES string."""
    if Chem.MolFromSmiles(smiles) is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles,
        includeChirality=include_chirality,
    )


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    frac_train: float = 0.948,
    frac_test: float = 0.052,
    seed: int = 40,
    include_chirality: bool = False,
    scaffold_col: str = "scaffold",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by scaffold so molecules sharing a scaffold stay together."""
    if abs(frac_train + frac_test - 1.0) >= 1e-6:
        raise ValueError("frac_train + frac_test must equal 1.0")

    df = df.copy()
    df[scaffold_col] = df[smiles_col].apply(
        lambda smiles: generate_scaffold(smiles, include_chirality)
    )

    scaffolds = defaultdict(list)
    for idx, scaffold in enumerate(df[scaffold_col]):
        if scaffold is not None:
            scaffolds[scaffold].append(idx)

    scaffold_sets = list(scaffolds.items())
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_sets)

    train_indices = []
    test_indices = []
    train_size = int(frac_train * len(df))
    for _, indices in scaffold_sets:
        if len(train_indices) + len(indices) <= train_size:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True),
    )
