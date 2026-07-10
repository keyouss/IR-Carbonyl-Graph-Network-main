"""Feature engineering utilities for baseline tree models."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


POSSIBLE_COLUMNS = [
    "Cl_SINGLE_R1", "Cl_SINGLE_R2", "S_AROMATIC_R1", "S_AROMATIC_R2",
    "S_SINGLE_R1", "S_SINGLE_R2", "F_SINGLE_R1", "F_SINGLE_R2",
    "O_AROMATIC_R1", "O_AROMATIC_R2", "O_DOUBLE_R1", "O_DOUBLE_R2",
    "O_SINGLE_R1", "O_SINGLE_R2", "N_TRIPLE_R2", "N_TRIPLE_R1",
    "N_AROMATIC_R1", "N_AROMATIC_R2", "N_DOUBLE_R1", "N_DOUBLE_R2",
    "N_SINGLE_R1", "N_SINGLE_R2", "C_TRIPLE_R1", "C_TRIPLE_R2",
    "C_AROMATIC_R2", "C_AROMATIC_R1", "C_DOUBLE_R1", "C_DOUBLE_R2",
    "C_SINGLE_R1", "C_SINGLE_R2", "H_SINGLE_R1", "H_SINGLE_R2",
    "P_SINGLE_R1", "P_SINGLE_R2", "Br_SINGLE_R1", "Br_SINGLE_R2",
    "I_SINGLE_R1", "I_SINGLE_R2", "P_DOUBLE_R1", "P_DOUBLE_R2",
    "Si_SINGLE_R1", "Si_SINGLE_R2",
]


def calculate_morgan_fingerprint(smiles: str, n_bits: int = 2048) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return list(fp)


def carbonyl_process_smiles(smiles: str):
    """Extract simple R1/R2 environment descriptors around the first carbonyl group."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None
    mol = Chem.AddHs(mol)

    carbonyl_oxygen = None
    carbonyl_carbon = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if neighbor.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                carbonyl_oxygen = neighbor
                carbonyl_carbon = atom
                break
        if carbonyl_oxygen is not None:
            break

    if carbonyl_oxygen is None or carbonyl_carbon is None:
        return None, None, None, None

    connected_atoms = [
        atom for atom in carbonyl_carbon.GetNeighbors()
        if atom.GetIdx() != carbonyl_oxygen.GetIdx()
    ]
    connections = []
    for atom in connected_atoms:
        neighbors_info = []
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() == carbonyl_carbon.GetIdx():
                continue
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            neighbors_info.append((neighbor.GetSymbol(), str(bond.GetBondType())))
        connections.append(
            {
                "atomic_num": atom.GetAtomicNum(),
                "connections": {atom.GetSymbol(): neighbors_info},
            }
        )

    connections.sort(key=lambda item: item["atomic_num"], reverse=True)
    r1 = connections[0] if connections else None
    r2 = connections[1] if len(connections) > 1 else None
    return r1, r2, r1["atomic_num"] if r1 else None, r2["atomic_num"] if r2 else None


def feature_engineering(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    """Build tabular features used by the baseline tree models."""
    rows = []
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        r1, r2, atomic_number_r1, atomic_number_r2 = carbonyl_process_smiles(smiles)
        counts = {
            "R1": r1,
            "R2": r2,
            "atomic_number_R1": atomic_number_r1,
            "atomic_number_R2": atomic_number_r2,
            "SMILES": smiles,
            "IR_Characteristic_Peak": row.get("IR_Characteristic_Peak"),
            "DOI": row.get("DOI"),
        }
        counts.update({col: 0 for col in POSSIBLE_COLUMNS})

        for i, bit in enumerate(calculate_morgan_fingerprint(smiles)):
            counts[f"Fingerprint_{i}"] = bit

        for suffix, connections in [("R1", r1), ("R2", r2)]:
            if not connections:
                continue
            for connection in connections["connections"].values():
                for atom_symbol, bond_type in connection:
                    col = f"{atom_symbol}_{bond_type}_{suffix}"
                    counts[col] = counts.get(col, 0) + 1
        rows.append(counts)

    return pd.DataFrame(rows).fillna(0)
