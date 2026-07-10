"""RDKit feature extraction for carbonyl molecular graphs."""

from __future__ import annotations

import torch
from rdkit import Chem

from .constants import COVALENT_RADIUS, ELECTRONEGATIVITY


def extract_environment_features(mol: Chem.Mol, center_idx: int, radius: int = 1) -> list[float]:
    """Extract atom environment features around one atom."""
    center_atom = mol.GetAtomWithIdx(center_idx)
    center_atomic_num = center_atom.GetAtomicNum()

    ring_size = 0
    for ring in mol.GetRingInfo().AtomRings():
        if center_idx in ring:
            ring_size = len(ring)
            break

    env_feats = [
        center_atom.GetDegree(),
        center_atom.GetHybridization().real,
        int(center_atom.GetIsAromatic()),
        center_atom.GetFormalCharge(),
        ring_size,
        ELECTRONEGATIVITY.get(center_atomic_num, 0.0),
        COVALENT_RADIUS.get(center_atomic_num, 0.0),
    ]

    neighbor_atoms = set()
    for bond_idx in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx):
        bond = mol.GetBondWithIdx(bond_idx)
        neighbor_atoms.add(bond.GetBeginAtomIdx())
        neighbor_atoms.add(bond.GetEndAtomIdx())

    neighbor_feats = [0.0] * 6
    for idx in neighbor_atoms:
        if idx == center_idx:
            continue
        atom = mol.GetAtomWithIdx(idx)
        neighbor_feats[0] += ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 0.0)
        neighbor_feats[1] += atom.GetDegree()
        neighbor_feats[2] += int(atom.GetIsAromatic())
        neighbor_feats[3] += int(atom.IsInRing())
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                neighbor_feats[4] += 1
            elif bond.GetBondTypeAsDouble() == 3.0:
                neighbor_feats[5] += 1

    env_feats.extend(neighbor_feats)
    return env_feats


def extract_carbonyl_features(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    """Return carbonyl mask and per-atom environment features."""
    carbonyl_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.float)
    carbonyl_bonds = []

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if {begin_atom.GetAtomicNum(), end_atom.GetAtomicNum()} == {6, 8}:
            carbonyl_bonds.append(bond)

    surrounding_atoms = set()
    for bond in carbonyl_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        center_idx = begin_atom.GetIdx() if begin_atom.GetAtomicNum() == 6 else end_atom.GetIdx()
        queue = [(center_idx, 0)]
        visited = set()

        while queue:
            current_idx, current_dist = queue.pop(0)
            if current_idx in visited:
                continue
            visited.add(current_idx)

            if current_dist in {1, 2}:
                surrounding_atoms.add((current_idx, current_dist))
            if current_dist < 2:
                current_atom = mol.GetAtomWithIdx(current_idx)
                queue.extend(
                    (neighbor.GetIdx(), current_dist + 1)
                    for neighbor in current_atom.GetNeighbors()
                    if neighbor.GetIdx() not in visited
                )

    for idx, dist in surrounding_atoms:
        carbonyl_mask[idx] = float(dist)

    for bond in carbonyl_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        oxygen_idx = begin_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else end_atom.GetIdx()
        carbon_idx = end_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else begin_atom.GetIdx()
        carbonyl_mask[oxygen_idx] = 0.8
        carbonyl_mask[carbon_idx] = 0.6

    carbonyl_env = torch.tensor(
        [extract_environment_features(mol, atom.GetIdx()) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )
    return carbonyl_mask, carbonyl_env
