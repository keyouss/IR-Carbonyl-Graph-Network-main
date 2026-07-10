"""PyTorch Geometric dataset for carbonyl IR prediction."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset

from .constants import COVALENT_RADIUS, ELECTRONEGATIVITY
from .features import extract_carbonyl_features


class CarbonylIRDataset(Dataset):
    """Convert SMILES and IR peaks into PyTorch Geometric molecular graphs."""

    def __init__(self, smiles_list, ir_values, doi=None):
        super().__init__()
        self.smiles_list = list(smiles_list)
        self.ir_values_raw = np.asarray(ir_values, dtype=float)
        self.mean = float(np.mean(self.ir_values_raw))
        self.std = float(np.std(self.ir_values_raw))
        if self.std == 0:
            raise ValueError("IR target standard deviation is zero.")
        self.ir_values = (self.ir_values_raw - self.mean) / self.std
        self.doi = list(doi) if doi is not None else [None] * len(self.smiles_list)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            atom_features.append(
                [
                    atomic_num,
                    atom.GetDegree(),
                    atom.GetImplicitValence(),
                    int(atom.IsInRing()),
                    atom.GetHybridization().real,
                    ELECTRONEGATIVITY.get(atomic_num, 0.0),
                    COVALENT_RADIUS.get(atomic_num, 0.0),
                ]
            )

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
                bond.GetBoolProp("_IsPolar") if bond.HasProp("_IsPolar") else 0.0,
            ]
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([features, features.copy()])

        carbonyl_mask, carbonyl_env = extract_carbonyl_features(mol)
        x = torch.cat([torch.tensor(atom_features, dtype=torch.float), carbonyl_env], dim=1)

        return Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            carbonyl_mask=carbonyl_mask,
            doi=self.doi[idx],
            smiles=self.smiles_list[idx],
            y=torch.tensor([self.ir_values[idx]], dtype=torch.float),
        )
