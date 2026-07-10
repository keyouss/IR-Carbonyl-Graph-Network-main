"""Data loading and SMILES preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None when RDKit cannot parse the input."""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def preprocess_carbonyl_table(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "IR_Characteristic_Peak",
) -> pd.DataFrame:
    """Canonicalize SMILES and keep one median target value per molecule."""
    df = df.copy()
    df["Canonical_SMILES"] = df[smiles_col].apply(canonicalize_smiles)
    df = df[df["Canonical_SMILES"].notnull()].copy()
    df[target_col] = df.groupby("Canonical_SMILES")[target_col].transform("median")
    return df.drop_duplicates(subset="Canonical_SMILES", keep="first").reset_index(drop=True)


def load_carbonyl_table(path: str | Path) -> pd.DataFrame:
    """Load a carbonyl dataset from Excel or CSV and apply project preprocessing."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data file type: {path.suffix}")
    return preprocess_carbonyl_table(df)
