# %% [markdown]
# # 单纯的两个节点 实验可解释性 修改了网络结构 把环境特征弄成节点特征

# %%
# import os
# from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
# import seaborn
from tqdm import tqdm

from torch_geometric.loader import DataLoader
import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from torch import nn
# from torch_geometric.nn import MessagePassing
import torch.nn.functional as F


# %% [markdown]
# ## prepare data

# %%
from rdkit import Chem

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)  # 将 SMILES 转换为分子对象
        if mol:
            canon_smiles = Chem.MolToSmiles(mol, canonical=True)  # 生成规范化的 SMILES
            return canon_smiles
        else:
            return None
    except:
        return None
    
# excel_file = r'D:\Jupyter\IR-DIAZO-KETONE-main\333\experiment_carbonyl_1_1_group - 副本.xlsx'   # IR-DIAZO-KETONE-main\\
excel_file = r'D:\1 ir amide\code_avaliable\CIAC_carbonyl_group.xlsx'   # IR-DIAZO-KETONE-main\\

df = pd.read_excel(excel_file)
# df = pd.read_csv(excel_file)
# 创建新列存储规范化的 SMILES
df['Canonical_SMILES'] = df['SMILES'].apply(canonicalize_smiles)

def pre_process(df):
    # 删除规范化 SMILES 为空的行
    df = df[df['Canonical_SMILES'].notnull()]
    # 计算每个 'Canonical_SMILES' 组中 'IR' 值的中值
    df['IR_Characteristic_Peak'] = df.groupby('Canonical_SMILES')['IR_Characteristic_Peak'].transform('median')
    # 删除重复项，只保留第一个出现的行
    df_unique = df.drop_duplicates(subset='Canonical_SMILES', keep='first')
    # 重置索引
    df_unique.reset_index(drop=True, inplace=True)
    # 显示去重后的数据集信息
    print(f"原始数据集大小：{len(df)}")
    print(f"去重后数据集大小：{len(df_unique)}")
    return df_unique

# def pre_process(df):
#     # 删除规范化 SMILES 为空的行
#     df = df[df['Canonical_SMILES'].notnull()]
#     # 计算每个 SMILES 组的 IR 中值
#     median_ir = df.groupby('Canonical_SMILES')['IR_Characteristic_Peak'].transform('median')
#     # 标记中值所在的行（可能有多个行等于中值）
#     df['is_median'] = df['IR_Characteristic_Peak'] == median_ir
#     # 按 SMILES 分组，并保留每组中第一个满足 is_median=True 的行
#     df_unique = df.sort_values('is_median', ascending=False).drop_duplicates(
#         subset='Canonical_SMILES', 
#         keep='first')
#     # 确保 IR 列更新为中值（处理偶数个重复值时保留第一个，但值强制为中值）
#     df_unique['IR_Characteristic_Peak'] = median_ir[df_unique.index]
#     # 清理临时列并重置索引
#     df_unique = df_unique.drop(columns='is_median').reset_index(drop=True)
#     # 打印信息
#     print(f"原始数据集大小：{len(df)}")
#     print(f"去重后数据集大小：{len(df_unique)}")
#     return df_unique
df_unique = pre_process(df) 


# %%
display(Chem.MolFromSmiles('CC([C@H](C1=CC=CC=C1)CC)=O'))


# %%
# A5CCDC
# 84BB9F
df_unique.to_excel(r'D:\Jupyter\IR-DIAZO-KETONE-main\333\experiment_carbonyl_group.xlsx',index=None)


# %% [markdown]
# ## construct dataset

# %%
# 环境特征合并 16，尝试多基团 7+9
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
import numpy as np

def extract_carbonyl_features(mol):
    """
    提取羰基相关特征
    返回：
    - carbonyl_mask: 标记羰基氧及其周围半径1和2范围内的原子（0表示非相关，0.8表示羰基氧，1表示半径1，2表示半径2）
    - carbonyl_env: 每个原子的周围环境特征
    """
    carbonyl_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.float)
    carbonyl_env = []

    # Step 1: 识别所有羰基键（C=O 或 O=C）
    carbonyl_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            # 检查是否为C=O或O=C键
            if (begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 8) or \
               (end_atom.GetAtomicNum() == 6 and begin_atom.GetAtomicNum() == 8):
                carbonyl_bonds.append(bond)

    # Step 2: 收集所有需要标记的原子（羰基碳周围半径1和2）
    surrounding_atoms = set()
    for bond in carbonyl_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        # 确定羰基碳的索引
        if begin_atom.GetAtomicNum() == 6:
            center_idx = begin_atom.GetIdx()  # 碳在开始原子
        else:
            center_idx = end_atom.GetIdx()    # 碳在结束原子
        # 使用BFS遍历半径2范围内的原子
        queue = [(center_idx, 0)]
        visited = set()
        while queue:
            current_idx, current_dist = queue.pop(0)
            if current_idx in visited:
                continue
            visited.add(current_idx)
            
            if current_dist == 1:
                surrounding_atoms.add((current_idx, 1))  # 半径1
            elif current_dist == 2:
                surrounding_atoms.add((current_idx, 2))  # 半径2
            # 继续扩展未达半径限制的原子
            if current_dist < 2:
                current_atom = mol.GetAtomWithIdx(current_idx)
                for neighbor in current_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in visited:
                        queue.append((neighbor_idx, current_dist + 1))

    # Step 3: 设置mask
    for idx, dist in surrounding_atoms:
        if dist == 1:
            carbonyl_mask[idx] = 1.0
        elif dist == 2:
            carbonyl_mask[idx] = 2.0
    # Step 4: 标记羰基氧和羰基碳
    for i, bond in enumerate(carbonyl_bonds):
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        oxygen_idx = begin_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else end_atom.GetIdx()
        carbon_idx = end_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else begin_atom.GetIdx()
        # 设置羰基氧的mask为0.8
        carbonyl_mask[oxygen_idx] = 0.8 + i
        # 设置羰基碳的mask为0.6
        carbonyl_mask[carbon_idx] = 0.6 + i
    
    # Step 5: 提取每个原子的环境特征（原逻辑不变）
    for atom in mol.GetAtoms():
        env_feats = extract_environment_features(mol, atom.GetIdx())
        carbonyl_env.append(env_feats)
    
    carbonyl_env = torch.tensor(carbonyl_env, dtype=torch.float)
    
    return carbonyl_mask, carbonyl_env

def extract_environment_features(mol, center_idx, radius=1):
    """
    提取羰基周围环境的特征
    """
    env_feats = []
    center_atom = mol.GetAtomWithIdx(center_idx)
    center_atomic_num = center_atom.GetAtomicNum()

    # 获取环信息
    ring_info = mol.GetRingInfo()
    # 1. 中心原子本身特征
    ring_size = 0
    for ring in ring_info.AtomRings():
        if center_idx in ring:
            ring_size = len(ring)
            break
    env_feats.extend([
        # center_atom.GetDegree(),  # 连接数
        int(center_atom.GetIsAromatic()),  # 是否芳香原子
        center_atom.GetFormalCharge(),  # 形式电荷
        ring_size,         # 所在环的大小（不在环中为0）
    ])
    # 2. 获取邻域原子
    neighbors = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx)
    neighbor_atoms = set()
    for bond_idx in neighbors:
        bond = mol.GetBondWithIdx(bond_idx)
        neighbor_atoms.add(bond.GetBeginAtomIdx())
        neighbor_atoms.add(bond.GetEndAtomIdx())
    
    # 统计邻域特征
    neighbor_feats = [0.0]*6
    for idx in neighbor_atoms:
        if idx == center_idx:
            continue
        atom = mol.GetAtomWithIdx(idx)
        electronegativity = electronegativity_dict.get(atom.GetAtomicNum(), 0.0)
        neighbor_feats[0] += electronegativity  # 电负性之和
        neighbor_feats[1] += atom.GetDegree()    # 连接数和
        neighbor_feats[2] += int(atom.GetIsAromatic())  # 芳香原子数
        neighbor_feats[3] += int(atom.IsInRing())    # 环原子数
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:  # 双键数
                neighbor_feats[4] += 1
            elif bond.GetBondTypeAsDouble() == 3.0:  # 三键数
                neighbor_feats[5] += 1

    env_feats.extend(neighbor_feats)
    return env_feats
# 定义电负性字典
electronegativity_dict = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82, 
    26: 1.83, 32: 2.01, 34: 2.55, 35: 2.96, 50: 1.96, 52: 2.1, 53: 2.66
}
# 定义共价半径字典
covalent_radius_dict = {
    1: 3.7, 5: 8.2, 6: 7.7, 7: 7.5, 8: 7.3, 9: 7.1, 14: 11.1, 15: 10.6, 16: 10.2, 17: 9.9, 19: 22.7, 
    26: 12.6, 32: 12.2, 34: 19.8, 35: 11.4, 50: 14.0, 52: 14.0, 53: 13.3
}
class CarbonylIRDataset(Dataset):
    def __init__(self, smiles_list, irc_values , doi = None , noise_level=0.0):
        self.smiles_list = smiles_list
        self.mean = np.mean(irc_values)
        self.std = np.std(irc_values)
        # self.irc_values = (add_noise(irc_values, noise_level) - self.mean) / self.std
        self.irc_values = (irc_values - self.mean) / self.std
        self.doi = doi
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        mol = Chem.AddHs(mol)
        if mol is None:
            return None
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            features = [
              atomic_num,          # 原子序数
              atom.GetDegree(),             # 连接数
              atom.GetImplicitValence(),    # 隐式价
              atom.IsInRing(),              # 是否在环中
              atom.GetHybridization().real  # 杂化类型
          ]
            features.append(electronegativity_dict.get(atomic_num, 0.0))    # 电负性
            features.append(covalent_radius_dict.get(atomic_num, 0.0))      # 共价半径
            atom_features.append(features)
        
        # 边索引
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])

        # 边特征处理
        edge_attr = []
        for bond in mol.GetBonds():
            features = [
                bond.GetBondTypeAsDouble(),  # 键类型（单键=1.0，双键=2.0，三键=3.0）
                int(bond.GetIsConjugated()), # 是否共轭
                int(bond.IsInRing()),        # 是否在环中
                int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),  # 是否芳香键
                bond.GetBoolProp('_IsPolar') if bond.HasProp('_IsPolar') else 0.0  # 极性
    ]
            # 双向边重复特征
            edge_attr.extend([features, features.copy()])  
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)#.view(-1, 1)
        
        # 羰基特征
        carbonyl_mask, carbonyl_env = extract_carbonyl_features(mol)
        atom_features = torch.tensor(atom_features, dtype=torch.float)
        atom_features = torch.cat([atom_features, carbonyl_env], dim=1)
        return Data(
            x=atom_features,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=edge_attr,
            carbonyl_mask=carbonyl_mask,
            doi = self.doi[idx],
            smiles = self.smiles_list[idx],
            y=torch.tensor([self.irc_values[idx]], dtype=torch.float))


# %%
# 环境特征分开 7+13
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
import numpy as np

def extract_carbonyl_features(mol):
    """
    提取羰基相关特征
    返回：
    - carbonyl_mask: 标记羰基氧及其周围半径1和2范围内的原子（0表示非相关，0.5表示羰基氧，1表示半径1，2表示半径2）
    - carbonyl_env: 每个原子的周围环境特征
    """
    carbonyl_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.float)
    carbonyl_env = []

    # Step 1: 识别所有羰基键（C=O 或 O=C）
    carbonyl_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            # 检查是否为C=O或O=C键
            if (begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 8) or \
               (end_atom.GetAtomicNum() == 6 and begin_atom.GetAtomicNum() == 8):
                carbonyl_bonds.append(bond)

    # Step 2: 收集所有需要标记的原子（羰基碳周围半径1和2）
    surrounding_atoms = set()
    for bond in carbonyl_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        # 确定羰基碳的索引
        if begin_atom.GetAtomicNum() == 6:
            center_idx = begin_atom.GetIdx()  # 碳在开始原子
        else:
            center_idx = end_atom.GetIdx()    # 碳在结束原子
        # 使用BFS遍历半径2范围内的原子
        queue = [(center_idx, 0)]
        visited = set()
        while queue:
            current_idx, current_dist = queue.pop(0)
            if current_idx in visited:
                continue
            visited.add(current_idx)
            
            if current_dist == 1:
                surrounding_atoms.add((current_idx, 1))  # 半径1
            elif current_dist == 2:
                surrounding_atoms.add((current_idx, 2))  # 半径2
            # 继续扩展未达半径限制的原子
            if current_dist < 2:
                current_atom = mol.GetAtomWithIdx(current_idx)
                for neighbor in current_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in visited:
                        queue.append((neighbor_idx, current_dist + 1))

    # Step 3: 设置mask
    for idx, dist in surrounding_atoms:
        if dist == 1:
            carbonyl_mask[idx] = 1.0
        elif dist == 2:
            carbonyl_mask[idx] = 2.0
    # Step 4: 标记羰基氧和羰基碳
    for bond in carbonyl_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        oxygen_idx = begin_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else end_atom.GetIdx()
        carbon_idx = end_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else begin_atom.GetIdx()
        # 设置羰基氧的mask为0.5
        carbonyl_mask[oxygen_idx] = 0.8
        # 设置羰基碳的mask为0
        carbonyl_mask[carbon_idx] = 0.6
    0
    # Step 5: 提取每个原子的环境特征（原逻辑不变）
    for atom in mol.GetAtoms():
        env_feats = extract_environment_features(mol, atom.GetIdx())
        carbonyl_env.append(env_feats)
    000
    carbonyl_env = torch.tensor(carbonyl_env, dtype=torch.float)
    
    return carbonyl_mask, carbonyl_env

def extract_environment_features(mol, center_idx, radius=1):
    """
    提取羰基周围环境的特征
    """
    env_feats = []
    center_atom = mol.GetAtomWithIdx(center_idx)
    center_atomic_num = center_atom.GetAtomicNum()

    # 获取环信息
    ring_info = mol.GetRingInfo()
    # 1. 中心原子本身特征
    ring_size = 0
    for ring in ring_info.AtomRings():
        if center_idx in ring:
            ring_size = len(ring)
            break
    env_feats.extend([
        center_atom.GetDegree(),  # 连接数
        center_atom.GetHybridization().real,  # 杂化类型
        int(center_atom.GetIsAromatic()),  # 是否芳香原子
        center_atom.GetFormalCharge(),  # 形式电荷
        # int(center_atom.IsInRing()),  # 是否在环中
        ring_size,         # 所在环的大小（不在环中为0）
        electronegativity_dict.get(center_atomic_num, 0.0),  # 电负性
        covalent_radius_dict.get(center_atomic_num, 0.0)   # 共价半径
    ])
    # 2. 获取邻域原子
    neighbors = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx)
    neighbor_atoms = set()
    for bond_idx in neighbors:
        bond = mol.GetBondWithIdx(bond_idx)
        neighbor_atoms.add(bond.GetBeginAtomIdx())
        neighbor_atoms.add(bond.GetEndAtomIdx())
    
    # 统计邻域特征
    neighbor_feats = [0.0]*6
    for idx in neighbor_atoms:
        if idx == center_idx:
            continue
        atom = mol.GetAtomWithIdx(idx)
        electronegativity = electronegativity_dict.get(atom.GetAtomicNum(), 0.0)
        neighbor_feats[0] += electronegativity  # 电负性之和
        neighbor_feats[1] += atom.GetDegree()    # 连接数和
        neighbor_feats[2] += int(atom.GetIsAromatic())  # 芳香原子数
        neighbor_feats[3] += int(atom.IsInRing())    # 环原子数
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:  # 双键数
                neighbor_feats[4] += 1
            elif bond.GetBondTypeAsDouble() == 3.0:  # 三键数
                neighbor_feats[5] += 1

    env_feats.extend(neighbor_feats)
    return env_feats
# 定义电负性字典
electronegativity_dict = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82, 
    26: 1.83, 32: 2.01, 34: 2.55, 35: 2.96, 50: 1.96, 52: 2.1, 53: 2.66
}
# 定义共价半径字典
covalent_radius_dict = {
    1: 3.7, 5: 8.2, 6: 7.7, 7: 7.5, 8: 7.3, 9: 7.1, 14: 11.1, 15: 10.6, 16: 10.2, 17: 9.9, 19: 22.7, 
    26: 12.6, 32: 12.2, 34: 19.8, 35: 11.4, 50: 14.0, 52: 14.0, 53: 13.3
}
class CarbonylIRDataset(Dataset):
    def __init__(self, smiles_list, irc_values,doi):
        self.smiles_list = smiles_list
        self.mean = np.mean(irc_values)
        self.std = np.std(irc_values)
        self.irc_values = (irc_values - self.mean) / self.std
        self.doi = doi
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        mol = Chem.AddHs(mol)
        if mol is None:
            return None
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            features = [
              atomic_num,          # 原子序数
              atom.GetDegree(),             # 连接数
              atom.GetImplicitValence(),    # 隐式价
              atom.IsInRing(),              # 是否在环中
              atom.GetHybridization().real  # 杂化类型
          ]
            features.append(electronegativity_dict.get(atomic_num, 0.0))    # 添加电负性
            features.append(covalent_radius_dict.get(atomic_num, 0.0))      # 添加共价半径
            atom_features.append(features)
        
        # 边索引
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])

        # 边特征处理
        edge_attr = []
        for bond in mol.GetBonds():
            features = [
                bond.GetBondTypeAsDouble(),  # 键类型（单键=1.0，双键=2.0，三键=3.0）
                int(bond.GetIsConjugated()), # 是否共轭
                int(bond.IsInRing()),        # 是否在环中
                int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),  # 是否芳香键
                bond.GetBoolProp('_IsPolar') if bond.HasProp('_IsPolar') else 0.0  # 极性
    ]
            # 双向边重复特征
            edge_attr.extend([features, features.copy()])  
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)#.view(-1, 1)
        
        # 羰基特征
        carbonyl_mask, carbonyl_env = extract_carbonyl_features(mol)
        
        return Data(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=edge_attr,
            carbonyl_mask=carbonyl_mask,
            carbonyl_env=carbonyl_env,
            doi = self.doi[idx],
            smiles = self.smiles_list[idx],
            y=torch.tensor([self.irc_values[idx]], dtype=torch.float))


# %%
# 无环境特征分开 7
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
import numpy as np

def extract_carbonyl_features(mol):
    """
    提取羰基相关特征
    返回：
    - carbonyl_mask: 标记羰基氧及其周围半径1和2范围内的原子（0表示非相关，0.5表示羰基氧，1表示半径1，2表示半径2）
    - carbonyl_env: 每个原子的周围环境特征
    """
    carbonyl_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.float)
    carbonyl_env = []

    # Step 1: 识别所有羰基键（C=O 或 O=C）
    carbonyl_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            # 检查是否为C=O或O=C键
            if (begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 8) or \
               (end_atom.GetAtomicNum() == 6 and begin_atom.GetAtomicNum() == 8):
                carbonyl_bonds.append(bond)

    # Step 2: 收集所有需要标记的原子（羰基碳周围半径1和2）
    surrounding_atoms = set()
    for bond in carbonyl_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        # 确定羰基碳的索引
        if begin_atom.GetAtomicNum() == 6:
            center_idx = begin_atom.GetIdx()  # 碳在开始原子
        else:
            center_idx = end_atom.GetIdx()    # 碳在结束原子
        # 使用BFS遍历半径2范围内的原子
        queue = [(center_idx, 0)]
        visited = set()
        while queue:
            current_idx, current_dist = queue.pop(0)
            if current_idx in visited:
                continue
            visited.add(current_idx)
            
            if current_dist == 1:
                surrounding_atoms.add((current_idx, 1))  # 半径1
            elif current_dist == 2:
                surrounding_atoms.add((current_idx, 2))  # 半径2
            # 继续扩展未达半径限制的原子
            if current_dist < 2:
                current_atom = mol.GetAtomWithIdx(current_idx)
                for neighbor in current_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in visited:
                        queue.append((neighbor_idx, current_dist + 1))

    # Step 3: 设置mask
    for idx, dist in surrounding_atoms:
        if dist == 1:
            carbonyl_mask[idx] = 1.0
        elif dist == 2:
            carbonyl_mask[idx] = 2.0
    # Step 4: 标记羰基氧和羰基碳
    for bond in carbonyl_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        oxygen_idx = begin_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else end_atom.GetIdx()
        carbon_idx = end_atom.GetIdx() if begin_atom.GetAtomicNum() == 8 else begin_atom.GetIdx()
        # 设置羰基氧的mask为0.5
        carbonyl_mask[oxygen_idx] = 0.8
        # 设置羰基碳的mask为0
        carbonyl_mask[carbon_idx] = 0.6
    
    # Step 5: 提取每个原子的环境特征（原逻辑不变）
    for atom in mol.GetAtoms():
        env_feats = extract_environment_features(mol, atom.GetIdx())
        carbonyl_env.append(env_feats)
    
    carbonyl_env = torch.tensor(carbonyl_env, dtype=torch.float)
    
    return carbonyl_mask, carbonyl_env

def extract_environment_features(mol, center_idx, radius=1):
    """
    提取羰基周围环境的特征
    """
    env_feats = []
    center_atom = mol.GetAtomWithIdx(center_idx)
    center_atomic_num = center_atom.GetAtomicNum()

    # 获取环信息
    ring_info = mol.GetRingInfo()
    # 1. 中心原子本身特征
    ring_size = 0
    for ring in ring_info.AtomRings():
        if center_idx in ring:
            ring_size = len(ring)
            break
    env_feats.extend([
        center_atom.GetDegree(),  # 连接数
        center_atom.GetHybridization().real,  # 杂化类型
        int(center_atom.GetIsAromatic()),  # 是否芳香原子
        center_atom.GetFormalCharge(),  # 形式电荷
        # int(center_atom.IsInRing()),  # 是否在环中
        ring_size,         # 所在环的大小（不在环中为0）
        electronegativity_dict.get(center_atomic_num, 0.0),  # 电负性
        covalent_radius_dict.get(center_atomic_num, 0.0)   # 共价半径
    ])
    # 2. 获取邻域原子
    neighbors = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx)
    neighbor_atoms = set()
    for bond_idx in neighbors:
        bond = mol.GetBondWithIdx(bond_idx)
        neighbor_atoms.add(bond.GetBeginAtomIdx())
        neighbor_atoms.add(bond.GetEndAtomIdx())
    
    # 统计邻域特征
    neighbor_feats = [0.0]*6
    for idx in neighbor_atoms:
        if idx == center_idx:
            continue
        atom = mol.GetAtomWithIdx(idx)
        electronegativity = electronegativity_dict.get(atom.GetAtomicNum(), 0.0)
        neighbor_feats[0] += electronegativity  # 电负性之和
        neighbor_feats[1] += atom.GetDegree()    # 连接数和
        neighbor_feats[2] += int(atom.GetIsAromatic())  # 芳香原子数
        neighbor_feats[3] += int(atom.IsInRing())    # 环原子数
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:  # 双键数
                neighbor_feats[4] += 1
            elif bond.GetBondTypeAsDouble() == 3.0:  # 三键数
                neighbor_feats[5] += 1

    env_feats.extend(neighbor_feats)
    return env_feats
# 定义电负性字典
electronegativity_dict = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82, 
    26: 1.83, 32: 2.01, 34: 2.55, 35: 2.96, 50: 1.96, 52: 2.1, 53: 2.66
}
# 定义共价半径字典
covalent_radius_dict = {
    1: 3.7, 5: 8.2, 6: 7.7, 7: 7.5, 8: 7.3, 9: 7.1, 14: 11.1, 15: 10.6, 16: 10.2, 17: 9.9, 19: 22.7, 
    26: 12.6, 32: 12.2, 34: 19.8, 35: 11.4, 50: 14.0, 52: 14.0, 53: 13.3
}
class CarbonylIRDataset(Dataset):
    def __init__(self, smiles_list, irc_values,doi):
        self.smiles_list = smiles_list
        self.mean = np.mean(irc_values)
        self.std = np.std(irc_values)
        self.irc_values = (irc_values - self.mean) / self.std
        self.doi = doi
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        mol = Chem.AddHs(mol)
        if mol is None:
            return None
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            features = [
              atomic_num,          # 原子序数
              atom.GetDegree(),             # 连接数
              atom.GetImplicitValence(),    # 隐式价
              atom.IsInRing(),              # 是否在环中
              atom.GetHybridization().real  # 杂化类型
          ]
            features.append(electronegativity_dict.get(atomic_num, 0.0))    # 添加电负性
            features.append(covalent_radius_dict.get(atomic_num, 0.0))      # 添加共价半径
            atom_features.append(features)
        
        # 边索引
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])

        # 边特征处理
        edge_attr = []
        for bond in mol.GetBonds():
            features = [
                bond.GetBondTypeAsDouble(),  # 键类型（单键=1.0，双键=2.0，三键=3.0）
                int(bond.GetIsConjugated()), # 是否共轭
                int(bond.IsInRing()),        # 是否在环中
                int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),  # 是否芳香键
                bond.GetBoolProp('_IsPolar') if bond.HasProp('_IsPolar') else 0.0  # 极性
    ]
            # 双向边重复特征
            edge_attr.extend([features, features.copy()])  
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)#.view(-1, 1)
        
        # 羰基特征
        carbonyl_mask, carbonyl_env = extract_carbonyl_features(mol)
        
        return Data(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=edge_attr,
            carbonyl_mask=carbonyl_mask,
            # carbonyl_env=carbonyl_env,
            doi = self.doi[idx],
            smiles = self.smiles_list[idx],
            y=torch.tensor([self.irc_values[idx]], dtype=torch.float))


# %% [markdown]
# ## scaffold split

# %%
import pandas as pd
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles, include_chirality=False):
    """  
    为一个 SMILES 字符串生成 Bemis-Murcko scaffold。
    返回 scaffold 的 SMILES 字符串（标准化）。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold

def scaffold_split(df, 
                   smiles_col='SMILES', 
                   frac_train=0.948, 
                   frac_valid=0.0, 
                   frac_test=0.052, 
                   seed=40,
                   include_chirality=False,
                   scaffold_col='scaffold'):
    """
    按 scaffold 划分数据集（支持随机打乱 scaffold 顺序），并保留 scaffold 列。
    同一个 scaffold 的所有分子只会出现在 train 或 test 中（不会跨集）。
    
    返回: train_df, test_df（均为包含 'scaffold' 列的 DataFrame）
    """
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-6, "划分比例之和必须为 1.0"

    df = df.copy()

    # 1. 添加 scaffold 列
    df[scaffold_col] = df[smiles_col].apply(
        lambda s: generate_scaffold(s, include_chirality=include_chirality)
    )

    # 2. 按 scaffold 分组（跳过 None）
    scaffolds = defaultdict(list)
    for idx, scaffold in enumerate(df[scaffold_col]):
        if scaffold is not None:
            scaffolds[scaffold].append(idx)

    # 3. 转为列表并打乱（关键：以整个 scaffold 为单位打乱）
    scaffold_sets = list(scaffolds.items())
    
    # 设置随机种子，保证可复现
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_sets)  # 使用新式随机数生成器更安全

    # 4. 初始化
    train_indices, test_indices = [], []
    train_size = int(frac_train * len(df))

    # 5. 按打乱后的顺序分配整个 scaffold
    for scaffold, indices in scaffold_sets:
        if len(train_indices) + len(indices) <= train_size:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    # 6. 返回带 scaffold 列的 DataFrame
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    return train_df, test_df
train_data, test_data = scaffold_split(df_unique1,smiles_col='SMILES', )


# %% [markdown]
# ## select data

# %%
data_array = np.arange(0, len(df_unique), 1)
np.random.seed(42)
np.random.shuffle(data_array)
# torch.random.manual_seed(42)
train_num = int(len(data_array) * 0.9)
test_num = int(len(data_array) * 0.05)
val_num = int(len(data_array) - train_num-test_num)
train_index = data_array[0:train_num]
valid_index = data_array[train_num:train_num + val_num]
test_index = data_array[train_num + val_num:train_num + val_num + test_num]

train_data = df_unique.iloc[train_index].reset_index(drop=True)
valid_data = df_unique.iloc[valid_index].reset_index(drop=True)
test_data = df_unique.iloc[test_index].reset_index(drop=True)

# 保存为 Excel 文件
# train_data.to_excel(r"D:\Jupyter\IR-DIAZO-KETONE-main\333\train_data.xlsx", index=False)
# valid_data.to_excel(r"D:\Jupyter\IR-DIAZO-KETONE-main\333\valid_data.xlsx", index=False)
# test_data.to_excel(r"D:\Jupyter\IR-DIAZO-KETONE-main\333\test_data.xlsx", index=False)

# df_unique1 = pd.concat([train_data,test_data],0).reset_index(drop=True)
# df_unique1


# %%
import torch
df_unique['DOI'] = None
graph_IR_dataset = CarbonylIRDataset(df_unique['Canonical_SMILES'],df_unique['IR_Characteristic_Peak'],df_unique['DOI'])
# torch.save(graph_IR_dataset, 'dataset/primary_graph_IR_dataset_neighbor_otherpaper.pt')
# graph_IR_dataset = torch.load('dataset/primary_graph_IR_dataset_neighbor_otherpaper.pt')

len(graph_IR_dataset),graph_IR_dataset.mean,graph_IR_dataset.std


# %%
import torch
tr_graph_IR_dataset = CarbonylIRDataset(train_data['Canonical_SMILES'],train_data['IR_Characteristic_Peak'],train_data['DOI'])
# torch.save(tr_graph_IR_dataset, 'dataset/primary_graph_IR_dataset_neighbor_train.pt')
# tr_graph_IR_dataset = torch.load('dataset/primary_graph_IR_dataset_neighbor_train.pt')
# va_graph_IR_dataset = CarbonylIRDataset(valid_data['Canonical_SMILES'],valid_data['IR_Characteristic_Peak'],valid_data['DOI'])
# torch.save(va_graph_IR_dataset, 'dataset/primary_graph_IR_dataset_neighbor_valid.pt')
# va_graph_IR_dataset = torch.load('dataset/primary_graph_IR_dataset_neighbor_valid.pt')
te_graph_IR_dataset = CarbonylIRDataset(test_data['Canonical_SMILES'],test_data['IR_Characteristic_Peak'],test_data['DOI'])
# torch.save(te_graph_IR_dataset, 'dataset/primary_graph_IR_dataset_neighbor_test.pt')
# te_graph_IR_dataset = torch.load('dataset/primary_graph_IR_dataset_neighbor_test.pt')

# te_graph_IR_dataset = graph_IR_dataset
(len(tr_graph_IR_dataset),tr_graph_IR_dataset.mean,tr_graph_IR_dataset.std),(len(te_graph_IR_dataset),te_graph_IR_dataset.mean,te_graph_IR_dataset.std)


# %%
import torch
# graph_IR_dataset = CarbonylIRDataset(df_unique['Canonical_SMILES'],df_unique['IR_Characteristic_Peak'],df_unique['DOI'])
# torch.save(graph_IR_dataset, 'dataset/mutiple_graph_IR_dataset_neighbor.pt')
graph_IR_dataset = torch.load('dataset/mutiple_graph_IR_dataset_neighbor.pt')
len(graph_IR_dataset)


# %%
import torch
# graph_IR_dataset = CarbonylIRDataset(df_unique['Canonical_SMILES'],df_unique['IR_Characteristic_Peak'],df_unique['DOI'])
# torch.save(graph_IR_dataset, 'dataset/primary_graph_IR_dataset_neighbor_noenv.pt')
graph_IR_dataset = torch.load('dataset/primary_graph_IR_dataset_neighbor_noenv.pt')
len(graph_IR_dataset)


# %%
# for ir in graph_IR_dataset:
#     print(ir)
i = 2
print(graph_IR_dataset[i])
# print(graph_IR_dataset[i].x)
# print(graph_IR_dataset[i].edge_index)
# print(graph_IR_dataset[i].edge_attr)
# print(graph_IR_dataset[i].y)
print(graph_IR_dataset[i].carbonyl_mask)
print(graph_IR_dataset[i].carbonyl_env)
# print(graph_IR_dataset[i].doi)


# %%
import random
def extract_amide_molecules(    dataset,     r1,     r2,     max_count: int = 500,     random_seed: int = 42) -> list:

    if random_seed is not None:
        random.seed(random_seed)
    
    # 将r1和r2统一转换为集合形式
    r1_set = {r1} if isinstance(r1, int) else set(r1)
    r2_set = {r2} if isinstance(r2, int) else set(r2)
    
    # 收集所有符合条件的分子
    all_valid_molecules = []
    
    for data in dataset:
        atomic_numbers = data.x[:, 0]  # 假设原子序数是x的第一列
        carbonyl_neighbor_indices = torch.where(data.carbonyl_mask == 1)[0]
        
        # 必须恰好有两个邻接原子
        if len(carbonyl_neighbor_indices) != 2:
            continue
        
        # 获取两个邻接原子的原子序数
        neighbor1_z = atomic_numbers[carbonyl_neighbor_indices[0]].item()
        neighbor2_z = atomic_numbers[carbonyl_neighbor_indices[1]].item()
        
        # 检查是否匹配任一组合 (顺序无关)
        if (
            (neighbor1_z in r1_set and neighbor2_z in r2_set) or 
            (neighbor1_z in r2_set and neighbor2_z in r1_set)
        ):
            all_valid_molecules.append(data)
    
    # 检查是否有足够多的分子
    if len(all_valid_molecules) == 0:
        print("警告: 没有找到任何符合条件的分子")
        return []
    
    # 去重采样
    selected_molecules = []
    selected_ids = set()
    total_molecules = len(all_valid_molecules)
    
    while len(selected_molecules) < min(max_count, total_molecules):
        remaining_molecules = [m for m in all_valid_molecules if id(m) not in selected_ids]
        
        if not remaining_molecules:  # 所有非重复元素已取完
            break
            
        new_molecule = random.choice(remaining_molecules)
        selected_molecules.append(new_molecule)
        selected_ids.add(id(new_molecule))
    
    print(f"采样完成: 从 {total_molecules} 个有效分子中选取了 {len(selected_molecules)} 个非重复分子")
    return selected_molecules
amide_moleculesN = extract_amide_molecules(graph_IR_dataset,6,7)
amide_moleculesO = extract_amide_molecules(graph_IR_dataset,6,8)
amide_moleculesF = extract_amide_molecules(graph_IR_dataset,6,[9,17,35,53])


# %%
i = 3
# print(amide_molecules[i])
# print(amide_molecules[i].x)
# print(amide_molecules[i].edge_index)
# print(amide_molecules[i].edge_attr)
# print(amide_molecules[i].y)
# print(amide_molecules[i].carbonyl_mask)
# print(amide_molecules[i].carbonyl_env)


# %% [markdown]
# ## network struct

# %%
# 环境特征合并 16
from torch_geometric.nn import GINEConv,GINConv
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.utils import scatter

class GCNEConv(nn.Module):
    """支持边特征的GCN变体"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_encoder = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # 边特征传播
        edge_emb = self.edge_encoder(edge_attr)
        out = scatter(edge_emb * x[row], col, dim=0, reduce='sum')
        # 节点特征变换
        return F.leaky_relu(self.linear(x) + out, 0.1)

class GATEConv(nn.Module):
    """修正后的支持边特征的GAT变体"""
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        
        # 注意力计算层
        self.attn = nn.Linear(3 * in_dim, heads)  # 输入: [x_i || x_j || e_ij]
        
        # 特征变换层
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_encoder = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        # 边特征编码
        edge_emb = self.edge_encoder(edge_attr)  # [E, out_dim]
        
        # 拼接节点和边特征 [x_i || x_j || e_ij]
        x_cat = torch.cat([x[row], x[col], edge_emb], dim=-1)  # [E, 3*in_dim]
        
        # 计算注意力分数 [E, heads]
        alpha = F.softmax(self.attn(x_cat), dim=0)  # 按边归一化
        
        # 多头特征变换
        h_node = self.linear(x)  # [N, out_dim]
        h_edge = edge_emb        # [E, out_dim]
        
        # 加权聚合 (分头处理)
        out = torch.zeros(x.size(0), self.heads, self.out_dim, device=x.device)
        alpha = alpha.unsqueeze(-1)  # [E, heads, 1]
        weighted = (x[row] + h_edge).unsqueeze(1) * alpha  # [E, heads, out_dim]
        out = scatter(weighted, col, dim=0, reduce='sum')  # [N, heads, out_dim]
        
        # 合并多头并残差连接
        return F.leaky_relu(h_node + out.mean(dim=1), 0.1)  # [N, out_dim]
class GNNGraph(nn.Module):
    def __init__(self, node_dim=16, edge_dim=5, hidden_dim=128,num_layers=4, num_nodes=2, conv_type='GIN'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.conv_type = conv_type.upper()  # 统一转为大写
        # 公共编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
        # 边编码器（GIN/GCNE/GATE需要）
        if self.conv_type in ['GINE', 'GCNE', 'GATE']:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
            )
        else:
            self.edge_encoder = None
        
        # 卷积层选择
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if self.conv_type == 'GCN':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif self.conv_type == 'GAT':
                conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif self.conv_type == 'GIN':
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),
                    ),
                    train_eps=True 
                )
            elif self.conv_type == 'GINE':
                conv = GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),
                    ),
                    edge_dim=hidden_dim,
                    train_eps=True
                )
            elif self.conv_type == 'GCNE':
                conv = GCNEConv(hidden_dim, hidden_dim)
            elif self.conv_type == 'GATE':
                conv = GATEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            self.convs.append(conv)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr,data):
        # data.x, data.edge_index, data.edge_attr
        
        # 节点特征、边特征编码 
        h  = self.node_encoder(x)  # 使用节点特征编码器
        edge_emb = self.edge_encoder(edge_attr) if self.edge_encoder else None

        # 图卷积层（添加残差连接）
          # 使用编码后的节点特征
        for conv in self.convs:
            if self.conv_type in ['GINE', 'GCNE', 'GATE']:
                h = conv(h, edge_index, edge_emb)
            else:
                h = conv(h, edge_index)
            h = self.dropout(h)

        # 环境特征处理
        carbonyl_mask = data.carbonyl_mask.view(-1, 1).float()
        # 注意力机制改进（缩放点积）
        if self.num_nodes ==2:
            selected_nodes = carbonyl_mask.squeeze() == 1
            graph_embed = h[selected_nodes].view(-1, self.hidden_dim * 2 )  # 将选中节点特征展平
        elif self.num_nodes ==1:
            selected_nodes = carbonyl_mask.squeeze() == 0.6
            graph_embed = h[selected_nodes].view(-1, self.hidden_dim)
        return graph_embed

class GNNPredictor(nn.Module):
    def __init__(self, node_dim=16, edge_dim=5, hidden_dim=128,num_layers=4,num_nodes=2,env_num=0,conv_type='GINE'):
        super().__init__()
        self.conv_type=conv_type
        self.gnn = GNNGraph(node_dim, edge_dim, hidden_dim,num_layers, num_nodes , conv_type)
        init_fn = lambda m: (
            nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None
        )
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)  # 添加Dropout层
        # self.num_nodes=num_nodes
        # self.env_num=env_num
        self.predict = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * num_nodes + 13*env_num),
            nn.Linear(self.hidden_dim * num_nodes + 13*env_num,self.hidden_dim * num_nodes + 13*env_num),
            nn.BatchNorm1d(self.hidden_dim * num_nodes + 13*env_num),  # 替换为BatchNorm
            nn.LeakyReLU(0.1),
            self.dropout,
            nn.Linear(self.hidden_dim * num_nodes + 13*env_num, 1)
        )
        self.gnn.apply(init_fn)
        self.predict.apply(init_fn)
    def forward(self, x, edge_index, edge_attr,data):
        graph_embed = self.gnn(x, edge_index, edge_attr,data)

        return self.predict(graph_embed).squeeze(-1)


# %%
# 环境特征合并 16， 尝试多个基团
class GNNGraph(nn.Module):
    def __init__(self, node_dim=17, edge_dim=5, hidden_dim=128,num_layers=4, num_nodes=2, conv_type='GIN'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.conv_type = conv_type.upper()  # 统一转为大写
        # 公共编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
        # 边编码器（GIN/GCNE/GATE需要）
        if self.conv_type in ['GINE', 'GCNE', 'GATE']:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
            )
        else:
            self.edge_encoder = None
        
        # 卷积层选择
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if self.conv_type == 'GCN':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif self.conv_type == 'GAT':
                conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif self.conv_type == 'GIN':
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),
                    ),
                    train_eps=True 
                )
            elif self.conv_type == 'GINE':
                conv = GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),
                    ),
                    edge_dim=hidden_dim,
                    train_eps=True
                )
            elif self.conv_type == 'GCNE':
                conv = GCNEConv(hidden_dim, hidden_dim)
            elif self.conv_type == 'GATE':
                conv = GATEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            self.convs.append(conv)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr,data):
        # data.x, data.edge_index, data.edge_attr
        
        # 节点特征、边特征编码 
        h  = self.node_encoder(x)  # 使用节点特征编码器
        edge_emb = self.edge_encoder(edge_attr) if self.edge_encoder else None

        # 图卷积层（添加残差连接）
          # 使用编码后的节点特征
        for conv in self.convs:
            if self.conv_type in ['GINE', 'GCNE', 'GATE']:
                h = conv(h, edge_index, edge_emb)
            else:
                h = conv(h, edge_index)
            h = self.dropout(h)

        # 环境特征处理
        carbonyl_mask = data.carbonyl_mask.view(-1, 1).float()
        # 注意力机制改进（缩放点积）
        if self.num_nodes ==2:
            selected_nodes = carbonyl_mask.squeeze() == 1
            graph_embed = h[selected_nodes].view(-1, self.hidden_dim * 2 )  # 将选中节点特征展平
        elif self.num_nodes ==1:
            selected_nodes = carbonyl_mask.squeeze()%1 == 0.6
            graph_embed = h[selected_nodes].view(-1, self.hidden_dim)
        return graph_embed

class GNNPredictor(nn.Module):
    def __init__(self, node_dim=16, edge_dim=5, hidden_dim=128,num_layers=4,num_nodes=2,env_num=0,conv_type='GINE'):
        super().__init__()
        self.conv_type=conv_type
        self.gnn = GNNGraph(node_dim, edge_dim, hidden_dim,num_layers, num_nodes , conv_type)
        init_fn = lambda m: (
            nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None
        )
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)  # 添加Dropout层
        # self.num_nodes=num_nodes
        # self.env_num=env_num
        self.predict = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * num_nodes + 13*env_num),
            nn.Linear(self.hidden_dim * num_nodes + 13*env_num,self.hidden_dim * num_nodes + 13*env_num),
            nn.BatchNorm1d(self.hidden_dim * num_nodes + 13*env_num),  # 替换为BatchNorm
            nn.LeakyReLU(0.1),
            self.dropout,
            nn.Linear(self.hidden_dim * num_nodes + 13*env_num, 1)
        )
        self.gnn.apply(init_fn)
        self.predict.apply(init_fn)
    def forward(self, x, edge_index, edge_attr,data):
        graph_embed = self.gnn(x, edge_index, edge_attr,data)

        return self.predict(graph_embed).squeeze(-1)


# %%
# 取多个的基团
from torch_geometric.nn import GINEConv
from torch import nn
class GNNGraph(nn.Module):
    def __init__(self, node_dim=7, edge_dim=5, hidden_dim=128, num_layers=4 ,use_neibors=True):
        super().__init__()
        # 保持原有初始化不变
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)
        self.use_neibors = use_neibors
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.1),
                ),
                edge_dim=hidden_dim,
                train_eps=True
            ) for _ in range(num_layers)
        ])

    def get_neighbor_nodes(self, data, target_mask=0.6, exclude_mask=0.8):
        edge_index = data.edge_index
        carbonyl_mask = data.carbonyl_mask
        atomic_nums = data.x[:, 0]  # 假设原子序数是node特征的第一维
        
        # 找到目标节点(mask > target_mask)
        target_nodes = (carbonyl_mask %1== target_mask).nonzero().squeeze()
        neighbor_nodes = []
        
        # 对每个目标节点单独处理
        for carbon_idx in target_nodes:
            # 获取当前目标节点的所有邻居
            neighbors = edge_index[1, edge_index[0] == carbon_idx]
            
            # 筛选符合条件的邻居:
            # 1. carbonyl_mask != exclude_mask
            # 2. 不是目标节点本身(避免自环)
            valid_neighbors = [
                n for n in neighbors 
                if (carbonyl_mask[n] %1 != exclude_mask) and (n != carbon_idx)
            ]
            
            # 按原子序数从大到小排序(仅当前子图的邻居)
            sorted_neighbors = sorted(
                valid_neighbors,
                key=lambda x: atomic_nums[x],
                reverse=True
            )
            
            # 只取前两个节点(如果存在)
            neighbor_nodes.extend(sorted_neighbors[:2])
        
        return torch.tensor(neighbor_nodes, dtype=torch.long) if neighbor_nodes else torch.tensor([], dtype=torch.long)

    def forward(self, x, edge_index, edge_attr, data):
        # 特征编码
        node_emb = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)

        # 图卷积
        h = node_emb
        for conv in self.convs:
            h = conv(h, edge_index, edge_emb)
        
        # 获取目标节点索引
        target_nodes = self.get_neighbor_nodes(data)
        
        # 拼接特征
        if len(target_nodes) > 0 and self.use_neibors:
            # 确保每个子图贡献两个节点特征
            graph_embed = h[target_nodes].view(-1, self.hidden_dim * 2)
        else:
            # 如果没有符合条件的节点，返回零向量
            graph_embed = torch.zeros(1, self.hidden_dim * 2, device=h.device)
        
        return graph_embed
class GNNPredictor(nn.Module):
    def __init__(self, node_dim=7, edge_dim=5, hidden_dim=128,num_layers=4,node_num=2,env_num=3):
        super().__init__()
        self.gnn = GNNGraph(node_dim, edge_dim, hidden_dim,num_layers)
        init_fn = lambda m: (
            nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None
        )
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)  # 添加Dropout层
        # self.node_num=node_num
        # self.env_num=env_num
        self.predict = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * node_num + 13*env_num),
            nn.Linear(self.hidden_dim * node_num + 13*env_num,self.hidden_dim * node_num + 13*env_num),
            nn.BatchNorm1d(self.hidden_dim * node_num + 13*env_num),  # 替换为BatchNorm
            nn.LeakyReLU(0.1),
            self.dropout,
            nn.Linear(self.hidden_dim * node_num + 13*env_num, 1)
        )
        self.gnn.apply(init_fn)
        self.predict.apply(init_fn)
    def forward(self, x, edge_index, edge_attr,data):
        graph_embed = self.gnn(x, edge_index, edge_attr,data)

        return self.predict(graph_embed).squeeze(-1)


# %%
# 环境特征分开 7+13
from torch_geometric.nn import GINEConv
from torch import nn
class GINGraph(nn.Module):
    def __init__(self, node_dim=7, edge_dim=5, hidden_dim=128,num_layers=4):
        super().__init__()
        
        # 初始化配置
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)  # 添加Dropout层
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.1),
        )
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),  # 替换为BatchNorm
            nn.LeakyReLU(0.1),  # 改用LeakyReLU
        )
        # 卷积层
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),  # 去掉第一层的映射
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),
#                         self.dropout
                    ),
                    edge_dim=hidden_dim,
                    train_eps=True
                )
            )
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # batch = data.batch
        
        # 节点特征、边特征编码 
        node_emb = self.node_encoder(x)  # 使用节点特征编码器
        edge_emb = self.edge_encoder(edge_attr)

        # 图卷积层（添加残差连接）
        h = node_emb  # 使用编码后的节点特征
        for conv in self.convs:
            h = conv(h, edge_index, edge_emb)# + h  # 残差连接

        # 环境特征处理
        carbonyl_env = data.carbonyl_env
        carbonyl_mask = data.carbonyl_mask.view(-1, 1).float()
        # print(carbonyl_env[0],carbonyl_env.shape,h.shape)
        # print(carbonyl_mask,carbonyl_mask.shape)
        # env_features = carbonyl_env[batch]
        # print(env_features[0],env_features.shape)
        # 注意力机制改进（缩放点积）
        combined = torch.cat([h, carbonyl_env], dim=1)# / (self.hidden_dim ** 0.25)  # 缩放输入
        selected_nodes = carbonyl_mask.squeeze() == 1
        graph_embed = combined[selected_nodes].view(-1, self.hidden_dim * 2 + 13*2)  # 将选中节点特征展平
        carbonyl_nodes = carbonyl_mask.squeeze() == 0.9
        graph_embed = torch.cat([combined[carbonyl_nodes],graph_embed], dim=1)
        # print(combined[selected_nodes][0][-13:])
        # print(carbonyl_env[selected_nodes][0])
        return graph_embed

class GNNPredictor(nn.Module):
    def __init__(self, node_dim=7, edge_dim=5, hidden_dim=128,num_layers=4,node_num=2,env_num=3):
        super().__init__()
        self.gnn = GINGraph(node_dim, edge_dim, hidden_dim,num_layers)
        init_fn = lambda m: (
            nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None
        )
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)  # 添加Dropout层
        # self.node_num=node_num
        # self.env_num=env_num
        self.predict = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * node_num + 13*env_num),
            nn.Linear(self.hidden_dim * node_num + 13*env_num,self.hidden_dim * node_num + 13*env_num),
            nn.BatchNorm1d(self.hidden_dim * num_nodes + 13*env_num),  # 替换为BatchNorm
            nn.LeakyReLU(0.1),
            self.dropout,
            nn.Linear(self.hidden_dim * node_num + 13*env_num, 1)
        )
        self.gnn.apply(init_fn)
        self.predict.apply(init_fn)
    def forward(self, data):
        graph_embed = self.gnn(data)

        return self.predict(graph_embed).squeeze(-1)


# %%
# 设置打印选项：禁用科学计数法，显示完整张量
torch.set_printoptions(threshold=float('inf'), sci_mode=False)

model = GINGraph(node_dim=17,hidden_dim=128,num_layers=4).to(device)
# criterion_fn = torch.nn.MSELoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
test1_graph_IR_loader = DataLoader(test_graph_IR_dataset[:8],batch_size=4,shuffle=False, num_workers=1)
for step, batch in enumerate(test1_graph_IR_loader):
    batch_atom_bond = batch
    batch_atom_bond = batch_atom_bond.to(device)
    pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr,batch_atom_bond)
    print(step,pred[0][-13:],pred.shape)

# loss = test111(model, device, test1_graph_IR_loader, optimizer, criterion_fn)


# %% [markdown]
# ## training set

# %%
def eval(model, device, loader_atom_bond):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for _, batch in enumerate(loader_atom_bond):
            batch_atom_bond = batch
            batch_atom_bond = batch_atom_bond.to(device)
            pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr,batch_atom_bond)
            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
            y_pred.append(pred[:].detach().cpu())

    y_true = torch.cat(y_true, dim=0) * graph_IR_dataset.std + graph_IR_dataset.mean
    y_pred = torch.cat(y_pred, dim=0) * graph_IR_dataset.std + graph_IR_dataset.mean
    # print(y_true[:4],y_pred[:4])
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).data.numpy()
def test(model, device, loader_atom_bond,save_fig_dir= None,model_name = 'model_name'):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _, batch in enumerate(loader_atom_bond):
            batch_atom_bond = batch
            batch_atom_bond = batch_atom_bond.to(device)
            pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr,batch_atom_bond)
            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1,))
            y_pred.append(pred[:].detach().cpu())
            
    y_true = torch.cat(y_true, dim=0) * graph_IR_dataset.std + graph_IR_dataset.mean
    y_pred = torch.cat(y_pred, dim=0) * graph_IR_dataset.std + graph_IR_dataset.mean

    R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
    test_mae = torch.mean(torch.abs(y_true - y_pred))
    test_rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    print(R_square)
    if save_fig_dir:
        fig = plot_prediction_scatter(y_true, y_pred, f'{model_name}', figsize=(6/2.54, 6/2.54), alpha=0.2)
        fig.savefig(f'{save_fig_dir}\{model_name}.pdf', bbox_inches='tight', dpi=300)
        fig = plot_prediction_scatter2(y_true, y_pred, f'{model_name}', figsize=(6/2.54, 6/2.54), alpha=0.2)
        fig.savefig(f'{save_fig_dir}\{model_name} 2.pdf', bbox_inches='tight', dpi=300)
        # plt.show()
    return y_pred, y_true,R_square,test_mae,test_rmse
def train(model, device, loader_atom_bond, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(loader_atom_bond):
        batch_atom_bond = batch
        batch_atom_bond = batch_atom_bond.to(device)

        pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr,batch_atom_bond)#.view(-1, )
        true=batch_atom_bond.y
        optimizer.zero_grad()
        loss = criterion_fn(pred, true)
        # print("Loss:", loss.item())
        
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
        avg_loss = loss_accum / (step + 1)
        # print("Average Loss:", avg_loss, "Steps:", step + 1,'\n')
            
    return loss


# %%
# 分开标准化
def eval(model, device, loader_atom_bond):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for _, batch in enumerate(loader_atom_bond):
            batch_atom_bond = batch
            batch_atom_bond = batch_atom_bond.to(device)
            pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr,batch_atom_bond)
            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
            y_pred.append(pred[:].detach().cpu())

    y_true = torch.cat(y_true, dim=0) * va_graph_IR_dataset.std + va_graph_IR_dataset.mean
    y_pred = torch.cat(y_pred, dim=0) * tr_graph_IR_dataset.std + tr_graph_IR_dataset.mean
    # print(y_true[:4],y_pred[:4])
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).data.numpy()
def test(model, device, loader_atom_bond,model_name = 'model_name',figsize=(9/2.54, 9/2.54),save_fig_dir= None,has_title=False):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _, batch in enumerate(loader_atom_bond):
            batch_atom_bond = batch
            batch_atom_bond = batch_atom_bond.to(device)
            pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr,batch_atom_bond)
            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1,))
            y_pred.append(pred[:].detach().cpu())
            
    y_true = torch.cat(y_true, dim=0) * te_graph_IR_dataset.std + te_graph_IR_dataset.mean
    y_pred = torch.cat(y_pred, dim=0) * tr_graph_IR_dataset.std + tr_graph_IR_dataset.mean

    R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
    test_mae = torch.mean(torch.abs(y_true - y_pred))
    test_rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    print(R_square)
    if save_fig_dir:
        fig = plot_prediction_scatter(y_true, y_pred, f'{model_name}', figsize=figsize, alpha=0.2,has_title=has_title)
        fig.savefig(f'{save_fig_dir}\{model_name} 1.pdf', bbox_inches='tight', dpi=300)
        fig = plot_prediction_scatter2(y_true, y_pred, f'{model_name}', figsize=figsize, alpha=0.2,has_title=has_title)
        fig.savefig(f'{save_fig_dir}\{model_name} 2.pdf', bbox_inches='tight', dpi=300)
        # plt.show()
    return y_pred, y_true,R_square,test_mae,test_rmse
def train(model, device, loader_atom_bond, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(loader_atom_bond):
        batch_atom_bond = batch
        batch_atom_bond = batch_atom_bond.to(device)

        pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr,batch_atom_bond)#.view(-1, )
        true=batch_atom_bond.y
        optimizer.zero_grad()
        loss = criterion_fn(pred, true)
        # print("Loss:", loss.item())
        
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
        # avg_loss = loss_accum / (step + 1)
        # print("Average Loss:", avg_loss, "Steps:", step + 1,'\n')
            
    return loss


# %%
# 画图

def evaluate_model(y_true, y_pred):
    r2 = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
    mae = torch.mean(torch.abs(y_true - y_pred))
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return rmse, mae, r2
def plot_prediction_scatter(y_true, y_pred, model_name='GNN', figsize=(9/2.54, 9/2.54), alpha=0.2,has_title=False):
    # 确保 y_true 和 y_pred 是张量
    # y_true = torch.tensor(y_true)
    # y_pred = torch.tensor(y_pred)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # 设置背景色（浅灰色半透明）
    ax.set_facecolor("#CCCBCB7D")
    # 设置网格线（白色半透明）
    ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)
    # Scatter plot
    ax.scatter(y_true.numpy(), y_pred.numpy(),color="#A511116F", alpha=alpha, zorder=2)
    
    # Perfect prediction line
    min_val = min(y_true.min().item(), y_pred.min().item())
    max_val = max(y_true.max().item(), y_pred.max().item())
    ax.plot([min_val, max_val], [min_val, max_val],color='black', linestyle='--', zorder=2)
    
    # 移除所有边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 隐藏刻度线
    ax.tick_params(axis='both', which='both', length=0)
    
    # Labels and title
    rmse, _, r2 = evaluate_model(y_true, y_pred)
    # ax.legend(title=f'$R^2$={r2:.6f}\nRMSE={rmse:.6f}',frameon=False,title_fontsize=12,fontsize=10,borderpad=1.2)
    legend = ax.legend(title=f'$R^2$={r2:.3f}\nRMSE={rmse:.3f}',
                      frameon=False, title_fontsize=10, fontsize=10, borderpad=1.2)
    legend.get_title().set_color("#710C0C6D")
    # ax.set_xlabel('Observed Value (cm$^{-1}$)')
    # ax.set_ylabel('Predicted Value (cm$^{-1}$)')
    if has_title:
        ax.set_title(model_name,fontsize=8)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    return fig
def plot_prediction_scatter2(y_true, y_pred, model_name='GNN', figsize=(9/2.54, 9/2.54), alpha=0.2,has_title=False):
    # 确保 y_true 和 y_pred 是张量
    # y_true = torch.tensor(y_true)
    # y_pred = torch.tensor(y_pred)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # 设置网格线（白色半透明）
    ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)
    # 移除所有边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 隐藏刻度线
    ax.tick_params(axis='both', which='both', length=0)
    # 设置背景色（浅灰色半透明）
    ax.set_facecolor("#CCCBCB7D")

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=alpha, zorder=2)
    
    # Perfect prediction line
    min_val = min(y_true.min().item(), y_pred.min().item())
    max_val = max(y_true.max().item(), y_pred.max().item())
    ax.plot([min_val, max_val], [min_val, max_val],color='black', linestyle='--', zorder=2)
    
    # Labels and title
    rmse, _, r2 = evaluate_model(y_true, y_pred)
    # ax.legend(title=f'$R^2$={r2:.6f}\nRMSE={rmse:.6f}',frameon=False,title_fontsize=12,fontsize=10,borderpad=1.2)
    legend = ax.legend(title=f'$R^2$={r2:.3f}\nRMSE={rmse:.3f}',
                      frameon=False, title_fontsize=10, fontsize=10, borderpad=1.2)
    # legend.get_title().set_color("#8A0F0F6E")
    # ax.set_xlabel('Observed Value (cm$^{-1}$)')
    # ax.set_ylabel('Predicted Value (cm$^{-1}$)')
    if has_title:
        ax.set_title(model_name,fontsize=8)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    return fig


# %% [markdown]
# ### training

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_split(random_seed=42,train_ratio = 0.9,validate_ratio = 0.05,test_ratio=0.05,):
    data_array = np.arange(0, total_num, 1)
    np.random.seed(random_seed)
    np.random.shuffle(data_array)
    # torch.random.manual_seed(42)
    train_num = int(len(data_array) * train_ratio)
    test_num = int(len(data_array) * test_ratio) if validate_ratio!=0 else int(len(data_array) - train_num)
    val_num = int(len(data_array) - train_num-test_num) if validate_ratio!=0 else 0
    # (train_graph_IR_dataset, valid_graph_IR_dataset, test_graph_IR_dataset) = torch.utils.data.random_split(graph_IR_dataset,[train_num, val_num, test_num],generator=torch.Generator().manual_seed(42))
    train_index = data_array[0:train_num]
    valid_index = data_array[train_num:train_num + val_num]
    test_index = data_array[train_num + val_num:train_num + val_num + test_num]
    test_graph_IR_dataset,valid_graph_IR_dataset,train_graph_IR_dataset=[],[],[]
    for i in test_index:
        test_graph_IR_dataset.append(graph_IR_dataset[i])
    if validate_ratio!=0:
        for i in valid_index:
            valid_graph_IR_dataset.append(graph_IR_dataset[i])
    for i in train_index:
        train_graph_IR_dataset.append(graph_IR_dataset[i])
    train_graph_IR_loader = DataLoader(train_graph_IR_dataset, batch_size=128,shuffle=True, num_workers=1)
    valid_graph_IR_loader = DataLoader(valid_graph_IR_dataset, batch_size=128,shuffle=False, num_workers=1)
    test_graph_IR_loader = DataLoader(test_graph_IR_dataset, batch_size=128,shuffle=False, num_workers=1)
    return train_graph_IR_loader,valid_graph_IR_loader,test_graph_IR_loader,test_graph_IR_dataset
# total_num = len(graph_IR_dataset)
# print('data num:',total_num)
# train_graph_IR_loader,valid_graph_IR_loader,test_graph_IR_loader,test_graph_IR_dataset=data_split(random_seed=42,train_ratio = 0.9,validate_ratio = 0.05,test_ratio=0.05)

train_graph_IR_dataset = [tr_graph_IR_dataset[i] for i in range(0,len(tr_graph_IR_dataset))]
train_graph_IR_loader = DataLoader(train_graph_IR_dataset, batch_size=128,shuffle=True, num_workers=1)
# valid_graph_IR_dataset = [va_graph_IR_dataset[i] for i in range(0,len(va_graph_IR_dataset))]
# valid_graph_IR_loader = DataLoader(valid_graph_IR_dataset, batch_size=128,shuffle=False, num_workers=1)
test_graph_IR_dataset = [te_graph_IR_dataset[i] for i in range(0,len(te_graph_IR_dataset))]
test_graph_IR_loader = DataLoader(test_graph_IR_dataset, batch_size=128,shuffle=False, num_workers=1)


# %%
import os
import pandas as pd

# conv_types = ['GATE', 'GAT', 'GCNE', 'GCN', 'GIN', 'GINE']
def training(save_path='saves',conv_type='GINE'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = GNNPredictor(node_dim=16,hidden_dim=256,num_layers=4,num_nodes=1,env_num=0,conv_type=conv_type).to(device)
    # model.load_state_dict(torch.load('saves_scaffold3/epoch_10.pt'))

    criterion_fn = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    patience = 25
    best_loss = float('inf')
    patience_counter = 0
    min_lr = 0.00001
    reduce_factor = 0.2
    for epoch in tqdm(range(1000)):
        loss = train(model, device, train_graph_IR_loader, optimizer, criterion_fn)
        # 检查损失并更新耐心计数
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            for param_group in optimizer.param_groups:
                new_lr = max(param_group['lr'] * reduce_factor, min_lr)
                param_group['lr'] = new_lr
                if patience_counter<26:
                    print(f"Reducing learning rate to {new_lr:.6f}")
                if param_group['lr']>min_lr:
                    patience_counter = 0  # 重置耐心计数

        if (epoch + 1) % 25 == 0:
            _,_,_,_,train_rmse = test(model, device, train_graph_IR_loader)
            _,_,_,_,valid_rmse = test(model, device, test_graph_IR_loader)
            print('RMSE is :  ',train_rmse, valid_rmse)
            torch.save(model.state_dict(), f'{save_path}/epoch_{epoch + 1}.pt')
        # 检查学习率和耐心计数
        if optimizer.param_groups[0]['lr'] <= min_lr and patience_counter >= 40:
            print("Stopping training due to no improvement.")
            print(train_rmse, valid_rmse)
            torch.save(model.state_dict(), f'{save_path}/epoch_{epoch + 1}.pt')
            break
    return model

model = training(save_path='test_local',conv_type='GINE')


# %%
# fine_tune
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.optim as optim
metrics_df = pd.DataFrame(columns=['Epoch', 'Train_RMSE', 'R_square', 'Test_MAE', 'Test_RMSE'])
def training(save_path='saves_fine_tune'):
    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 初始化模型
    model = GNNPredictor(node_dim=16, hidden_dim=256, num_layers=4, num_nodes=1, env_num=0, conv_type='GINE').to(device)
    model.load_state_dict(torch.load(r'D:\Jupyter\IR-DIAZO-KETONE-main\saves\epoch_208.pt'))

    # 初始化优化器和损失函数
    criterion_fn = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.000001, weight_decay=0.0001)
    
    # 训练参数
    patience = 25
    best_loss = float('inf')
    patience_counter = 0
    min_lr = 0.000001
    reduce_factor = 0.2

    # 初始化DataFrame记录训练指标
    
    for epoch in tqdm(range(1000)):
        # 训练
        loss = train(model, device, train_graph_IR_loader, optimizer, criterion_fn)
        
        # 评估
        # train_rmse = eval(model, device, train_graph_IR_loader)
        _, _, R_square, test_mae, test_rmse = test(model, device, test_graph_IR_loader, 'valid_final')
        
        # 打印当前指标
        print(f'Epoch {epoch + 1}: , R²: {R_square }, Test MAE: {test_mae }, Test RMSE: {test_rmse }')

        # 记录指标到DataFrame
        metrics_df.loc[len(metrics_df)] = {
            'Epoch': epoch + 1,
            # 'Train_RMSE': train_rmse,
            'R_square': float(R_square),
            'Test_MAE': float(test_mae),
            'Test_RMSE':float(test_rmse)
        }

        # 保存模型和指标
        torch.save(model.state_dict(), f'{save_path}/epoch_{epoch + 1}.pt')
        
        if epoch%25==0:
            metrics_df.to_excel(f'{save_path}/training_metrics.xlsx', index=False)
        # 早停检查
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if R_square > 0.933:
            print(f"Stopping training due to high R² ({R_square} > 0.933).")
            # print(f"Final Train RMSE: {train_rmse:.4f}, R²: {R_square:.4f}")
            torch.save(model.state_dict(), f'{save_path}/model_save_final.pth')
            metrics_df.to_excel(f'{save_path}/training_metrics.xlsx', index=False)
            break

    return model

# 调用训练函数
training(save_path=r'saves_fine_tune4')


# %%
metrics_df.to_excel(f'saves/fine_tune/training_metrics2.xlsx', index=False)


# %% [markdown]
# ### interference test 

# %%
def calculate_similarity(train_smiles, test_smiles):
    """计算训练集和测试集之间的Tanimoto相似度"""
    train_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in train_smiles]
    test_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in test_smiles]

    similarity_scores = []
    for test_fp in test_fps:
        max_similarity = max(AllChem.DataStructs.TanimotoSimilarity(test_fp, train_fp) for train_fp in train_fps)
        similarity_scores.append(max_similarity)
    return similarity_scores

def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise


# %%
len(test_graph_IR_dataset)


# %%
train_smiles, test_smiles = [smile for batch in train_graph_IR_loader for smile in batch.smiles ], [mol.smiles for mol in test_graph_IR_dataset ]
similarity_scores = calculate_similarity(train_smiles, test_smiles)


# %%
i = 0
for score in similarity_scores:
    if score <= 0.5:
        i += 1
i


# %%
# similarity
similarity_threshold = [0.5,0.6,0.7,0.8,0.9,0.95]
train_graph_IR_loader,valid_graph_IR_loader,test_graph_IR_loader,test_graph_IR_dataset=data_split(random_seed=42,train_ratio = 0.9,validate_ratio = 0.05,test_ratio=0.05)
train_smiles, test_smiles = [smile for batch in train_graph_IR_loader for smile in batch.smiles ], [mol.smiles for mol in test_graph_IR_dataset ]
similarity_scores = calculate_similarity(train_smiles, test_smiles)
similarity_datasets ,metric = {} , {}
for i,threshold in enumerate(similarity_threshold):
    print('similaritu:',threshold)
    similarity_datasets[threshold] = [j for j, score in enumerate(similarity_scores) if score > threshold]
    subset = [test_graph_IR_dataset[idx] for idx in similarity_datasets[threshold]]
    print(len(subset))
    graph_IR_loader = DataLoader(subset, batch_size=128,shuffle=False, num_workers=1)
    _,_,R_square,test_mae,test_rmse = test(model, device, graph_IR_loader,model_name='similarity_threshold_'+str(i))
    metric[threshold]={'R_square': float(R_square),'MAE': float(test_mae),'RMSE': float(test_rmse)}
df = pd.DataFrame(metric).T
df.to_csv('interference_test/similarity.csv')


# %%
plt.style.use('seaborn-v0_8')


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator

# 设置全局样式
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")

# 读取数据
df = pd.read_csv('interference_test/similarity.csv', index_col=0)

# 创建图形 (6cm宽 × 7cm高)
fig, axes = plt.subplots(3, 1, figsize=(6/2.54, 7/2.54), dpi=300, 
                         gridspec_kw={'hspace': 0.2})  # 关键修改：hspace控制子图
# 自定义x轴百分比格式
def percent_formatter(x, pos):
    return f">{int(x * 100)}%"

# 第一个子图 - R_square
axes[0].scatter(df.index, df['R_square'], color="#1f53b4", s=40, edgecolor='black', alpha=0.7, linewidths=0)
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[0].set_xticklabels([])  # 隐藏x轴标签
for x, y in zip(df.index, df['R_square']):
    axes[0].text(x, y + 0.003, f"{y:.3f}", ha='center', va='bottom', fontsize=7)

# 第二个子图 - MAE
axes[1].scatter(df.index, df['MAE'], color="#be5757", s=40, edgecolor='black', alpha=0.7, linewidths=0)
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].set_xticklabels([])  # 隐藏x轴标签
for x, y in zip(df.index, df['MAE']):
    axes[1].text(x, y + 0.21, f"{y:.3f}", ha='center', va='bottom', fontsize=7)

# 第三个子图 - RMSE
axes[2].scatter(df.index, df['RMSE'], color="#70af78", s=40, edgecolor='black', alpha=0.7, linewidths=0)
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[2].xaxis.set_major_formatter(FuncFormatter(percent_formatter))  # x轴百分比格式
for x, y in zip(df.index, df['RMSE']):
    axes[2].text(x, y + 0.4, f"{y:.3f}", ha='center', va='bottom', fontsize=7)

# 统一设置所有子图的样式
for ax in axes:
    # 网格和背景
    ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)
    ax.set_facecolor("#CCCBCB7D")
    
    # 边框和刻度
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0, labelsize=8)  # 刻度字体大小
    ax.set_xticks(df.index)  # 直接使用数据中的x值作为刻度
    # 控制x轴主刻度数量 (5个) 并保持垂直网格线
    
    # 调整y轴范围
    y_min, y_max = ax.get_ylim()
    padding = (y_max - y_min) * 0.3
    ax.set_ylim(y_min - padding, y_max + padding)
    x_min, x_max = ax.get_xlim()
    x_padding = (x_max - x_min) * 0.1   # x轴扩展比例（较小，避免过度空白）
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
# 调整布局并保存
plt.tight_layout()
plt.savefig('interference_test/similarity.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %%
# noise level
noise_level = [i/20 for i in range(6)]
metric = {}
for i in noise_level:
    print('noise_level:',i)
    graph_IR_dataset = CarbonylIRDataset(df_unique['Canonical_SMILES'],df_unique['IR_Characteristic_Peak'],df_unique['DOI'],noise_level= i)
    torch.save(graph_IR_dataset, f'dataset/graph_IR_dataset_neighbor_noise_{i}.pt')
    train_graph_IR_loader,valid_graph_IR_loader,test_graph_IR_loader=data_split(random_seed=42,train_ratio = 0.9,validate_ratio = 0.05,test_ratio=0.05)
    model = training(save_path=r'interference_test\noise_level_'+str(i))
    _,_,R_square,test_mae,test_rmse = test(model, device, test_graph_IR_loader,model_name='training_data_ratio_'+str(i))
    metric[f'{i}']={'R_square': float(R_square),'MAE': float(test_mae),'RMSE': float(test_rmse)}
df = pd.DataFrame(metric)
df.to_csv('interference_test/noise_level.csv')


# %%
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, FuncFormatter  # 导入格式化工具
plt.style.use('seaborn-v0_8')
df = pd.read_csv('interference_test/noise_level.csv', index_col=0)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
print(type(numerical_cols),numerical_cols)
# 创建一个3行1列的图形
# 创建图形 (5英寸×5英寸)，设置紧凑子图间距
fig, axes = plt.subplots(3, 1, figsize=(6/2.54, 7/2.54), dpi=300, 
                        gridspec_kw={'hspace': 0.2})

# 第一个子图 - R_square
axes[0].scatter(df.index, df['R_square'], color="#1f53b4", s=40, 
               edgecolor='black', alpha=0.7, linewidths=0, zorder=2)
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[0].set_xticklabels([])  # 隐藏x轴标签
for x, y in zip(df.index, df['R_square']):
    axes[0].text(x, y + 0.005, f"{y:.3f}", ha='center', va='bottom', 
                fontsize=7, zorder=3)

# 第二个子图 - MAE
axes[1].scatter(df.index, df['MAE'], color="#be5757", s=40, 
               edgecolor='black', alpha=0.7, linewidths=0, zorder=2)
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].set_xticklabels([])  # 隐藏x轴标签
for x, y in zip(df.index, df['MAE']):
    axes[1].text(x, y + 0.3, f"{y:.3f}", ha='center', va='bottom', 
                fontsize=7, zorder=3)

    
# 第三个子图 - RMSE
axes[2].scatter(df.index, df['RMSE'], color="#70af78", s=40, 
               edgecolor='black', alpha=0.7, linewidths=0, zorder=2)
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for x, y in zip(df.index, df['RMSE']):
    axes[2].text(x, y + 0.4, f"{y:.3f}", ha='center', va='bottom', 
                fontsize=7, zorder=3)

# 统一设置所有子图样式
for ax in axes:
    # 网格和背景
    ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)
    ax.set_facecolor("#CCCBCB7D")
    
    # 边框和刻度
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(df.index)  # 直接使用数据中的x值作为刻度
    ax.tick_params(axis='both', which='major', labelsize=8, length=0)  # 刻度字体大小
    # 同步扩大x轴和y轴范围
    y_min, y_max = ax.get_ylim()
    y_padding = (y_max - y_min) * 0.3
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    x_min, x_max = ax.get_xlim()
    x_padding = (x_max - x_min) * 0.04
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
plt.tight_layout()
plt.savefig('interference_test/noise_level.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %%
import matplotlib.pyplot as plt
print(plt.style.available)


# %%
# training data radio
training_data_ratio = [0.1,0.3,0.5,0.7,0.9]
metric= {}
for i in training_data_ratio:
    print('training_data_ratio:',i)
    train_graph_IR_loader,valid_graph_IR_loader,test_graph_IR_loader=data_split(random_seed=42,train_ratio = i,validate_ratio = 0.5-i/2,test_ratio=0.5-i/2)
    model = training(save_path=r'interference_test\training_data_ratio_'+str(i))
    _,_,R_square,test_mae,test_rmse = test(model, device, test_graph_IR_loader,model_name='training_data_ratio_'+str(i))
    metric[f'{i}']={'R_square': float(R_square),'MAE': float(test_mae),'RMSE': float(test_rmse)}
df = pd.DataFrame(metric)
df.to_csv('interference_test/training_ratio.csv')


# %%
plt.style.use('seaborn-v0_8')


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# 设置全局样式
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")

# 读取数据
df = pd.read_csv('interference_test/training_ratio.csv', index_col=0)

# 创建图形 (5英寸×5英寸)，设置紧凑子图间距
fig, axes = plt.subplots(3, 1, figsize=(6/2.54, 7/2.54), dpi=300, 
                        gridspec_kw={'hspace': 0.2})

# 第一个子图 - R_square
axes[0].scatter(df.index, df['R_square'], color="#1f53b4", s=40, 
               edgecolor='black', alpha=0.7, linewidths=0, zorder=2)
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[0].set_xticklabels([])  # 隐藏x轴标签
for x, y in zip(df.index, df['R_square']):
    axes[0].text(x, y + 0.009, f"{y:.3f}", ha='center', va='bottom', 
                fontsize=7, zorder=3)

# 第二个子图 - MAE
axes[1].scatter(df.index, df['MAE'], color="#be5757", s=40, 
               edgecolor='black', alpha=0.7, linewidths=0, zorder=2)
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].set_xticklabels([])  # 隐藏x轴标签
for x, y in zip(df.index, df['MAE']):
    axes[1].text(x, y + 0.27, f"{y:.3f}", ha='center', va='bottom', 
                fontsize=7, zorder=3)

    
# 第三个子图 - RMSE
axes[2].scatter(df.index, df['RMSE'], color="#70af78", s=40, 
               edgecolor='black', alpha=0.7, linewidths=0, zorder=2)
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for x, y in zip(df.index, df['RMSE']):
    axes[2].text(x, y + 0.6, f"{y:.3f}", ha='center', va='bottom', 
                fontsize=7, zorder=3)

# 统一设置所有子图样式
for ax in axes:
    # 网格和背景
    ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)
    ax.set_facecolor("#CCCBCB7D")
    
    # 边框和刻度
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(df.index)  # 直接使用数据中的x值作为刻度
    ax.tick_params(axis='both', which='major', labelsize=8, length=0)  # 刻度字体大小
    # 同步扩大x轴和y轴范围
    y_min, y_max = ax.get_ylim()
    y_padding = (y_max - y_min) * 0.25
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    x_min, x_max = ax.get_xlim()
    x_padding = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - x_padding, x_max + x_padding)

# 保存高清图像
plt.tight_layout()
plt.savefig('interference_test/training_ratio.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %% [markdown]
# ### grid search

# %%
# 绑定的超参数组合
param_combinations = [
    {'hidden_dim': 128, 'node_num': 2},
    {'hidden_dim': 256, 'node_num': 1}
]

# 卷积类型列表
conv_types = ['GATE', 'GAT', 'GCNE', 'GCN', 'GIN', 'GINE']

# 训练配置
train_config = {
    'num_layers': 4,
    'initial_lr': 0.01,
    'min_lr': 1e-6,
    'patience': 30,
    'reduce_factor': 0.5,
    'epochs': 1000,
    'eval_interval': 25  # 验证间隔
}

# 主结果目录
os.makedirs("grid11_search", exist_ok=True)

for params in tqdm(param_combinations, desc='Hyperparameter Combinations'):
    hidden_dim = params['hidden_dim']
    node_num = params['node_num']
    
    for conv_type in tqdm(conv_types, desc=f'h{hidden_dim}_n{node_num}', leave=False):
        # 创建模型专属目录
        model_name = f"{conv_type}_h{hidden_dim}_n{node_num}"
        save_dir = f"grid_search/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化模型和优化器
        model = GNNPredictor(
            node_dim=16,
            edge_dim=5,
            hidden_dim=hidden_dim,
            num_layers=train_config['num_layers'],
            num_nodes=node_num,
            conv_type=conv_type
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['initial_lr'])
        criterion_fn = torch.nn.MSELoss()
        
        # 初始化当前配置的结果DataFrame
        config_results = pd.DataFrame(columns=[
            'epoch', 'train_rmse', 'valid_rmse', 'lr', 'model_path'
        ])
        
        # 训练变量初始化
        current_lr = train_config['initial_lr']
        patience_counter = 0
        best_loss = float('inf')
        
        # 训练循环
        for epoch in range(train_config['epochs']):
            # 训练步骤
            loss = train(model, device, train_graph_IR_loader, optimizer, criterion_fn)
            
            # 学习率调整
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= train_config['patience']:
                current_lr = max(current_lr * train_config['reduce_factor'], train_config['min_lr'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                if current_lr > train_config['min_lr']:
                    patience_counter = 0
            
            # 验证步骤
            if (epoch + 1) % train_config['eval_interval'] == 0 :
                train_rmse = eval(model, device, train_graph_IR_loader)
                valid_rmse = eval(model, device, valid_graph_IR_loader)
                
                # 保存模型参数
                checkpoint_path = f"{save_dir}/epoch_{epoch+1}.pt"
                torch.save(model.state_dict(),checkpoint_path)
                
                # 记录当前配置结果
                config_results.loc[len(config_results)] = {
                    'epoch': epoch + 1,
                    'train_rmse': train_rmse,
                    'valid_rmse': valid_rmse,
                    'lr': current_lr,
                    'model_path': checkpoint_path
                }
                
                # 实时保存当前配置结果
                config_results.to_excel(f"{save_dir}/training_log.xlsx", index=False)
            
            # 早停检查
            if current_lr <= train_config['min_lr'] and patience_counter >= 50:
                break

        # 保存最终配置摘要
        summary = {
            'conv_type': conv_type,
            'hidden_dim': hidden_dim,
            'node_num': node_num,
            'final_epoch': epoch + 1,
            'final_train_rmse': train_rmse,
            'final_valid_rmse': valid_rmse,
            'min_train_rmse': config_results['train_rmse'].min(),
            'min_valid_rmse': config_results['valid_rmse'].min()
        }
        pd.DataFrame([summary]).to_excel(f"{save_dir}/config_summary.xlsx", index=False)

print("\nAll configurations completed! Results saved in respective directories.")


# %%
# 卷积类型列表
# conv_types = ['GIN', 'GINE','GAT', 'GCNE', 'GCN','GATE', ]
import os
import pandas as pd
import torch
from tqdm import tqdm

# 超参数组合网格
param_grid = {
    'hidden_dim': [256,200],
    'num_nodes': [2],
    'num_layers': [3, 4, 5],
    'conv_type': ['GINE']
}

# 训练配置
train_config = {
    'initial_lr': 0.01,
    'min_lr': 1e-6,
    'patience': 25,
    'reduce_factor': 0.2,
    'epochs': 1000,
    'eval_interval': 25  # 验证间隔
}

# 主结果目录
os.makedirs("grid_search", exist_ok=True)

# 生成所有参数组合
all_params = []
for hd in param_grid['hidden_dim']:
    for n_n in param_grid['num_nodes']:
        for nl in param_grid['num_layers']:
            for ct in param_grid['conv_type']:
                all_params.append({
                    'hidden_dim': hd,
                    'num_nodes': n_n,
                    'num_layers': nl,
                    'conv_type': ct
                })

for params in tqdm(all_params, desc='Grid Search Progress'):
    hidden_dim = int(params['hidden_dim'])
    num_nodes = int(params['num_nodes'])
    num_layers = int(params['num_layers'])
    conv_type = str(params['conv_type'])
    # # 创建模型专属目录
    model_name = f"{conv_type}_h{hidden_dim}_n{num_nodes}_l{num_layers}"
    save_dir = f"grid_search/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化模型和优化器
    model = GNNPredictor(
        node_dim=16,
        edge_dim=5,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_nodes=num_nodes,
        env_num=0,
        conv_type=conv_type
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['initial_lr'])
    criterion_fn = torch.nn.MSELoss()
    
    # 初始化当前配置的结果DataFrame
    config_results = pd.DataFrame(columns=[
        'epoch', 'train_rmse', 'valid_rmse', 'lr', 'model_path'
    ])
    
    # 训练变量初始化
    current_lr = train_config['initial_lr']
    patience_counter = 0
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(train_config['epochs']):
        # 训练步骤
        loss = train(model, device, train_graph_IR_loader, optimizer, criterion_fn)
        
        # 学习率调整
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= train_config['patience']:
            current_lr = max(current_lr * train_config['reduce_factor'], train_config['min_lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            if current_lr > train_config['min_lr']:
                patience_counter = 0
        
        # 验证步骤
        if (epoch + 1) % train_config['eval_interval'] == 0:
            train_rmse = eval(model, device, train_graph_IR_loader)
            valid_rmse = eval(model, device, valid_graph_IR_loader)
            
            # 保存模型参数
            checkpoint_path = f"{save_dir}/epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            # 记录当前配置结果
            config_results.loc[len(config_results)] = {
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'valid_rmse': valid_rmse,
                'lr': current_lr,
                'model_path': checkpoint_path
            }
            
            # 实时保存当前配置结果
            config_results.to_excel(f"{save_dir}/training_log.xlsx", index=False)
        
        # 早停检查
        if current_lr <= train_config['min_lr'] and patience_counter >= 40:
            break

    # 保存最终配置摘要
    summary = {
        'conv_type': conv_type,
        'hidden_dim': hidden_dim,
        'num_nodes': num_nodes,
        'num_layers': num_layers,
        'final_epoch': epoch + 1,
        'final_train_rmse': train_rmse,
        'final_valid_rmse': valid_rmse,
        'min_train_rmse': config_results['train_rmse'].min(),
        'min_valid_rmse': config_results['valid_rmse'].min()
    }
    pd.DataFrame([summary]).to_excel(f"{save_dir}/config_summary.xlsx", index=False)

print("\nAll configurations completed! Results saved in respective directories.")


# %% [markdown]
# ## metric

# %%
# mname = 'ablation/GIN/model_save_50.pth'   #GINE_h128_n2   GINE_h256_n1
# mpname = mname.split('/')[1]
# model = GNNPredictor(node_dim=7, hidden_dim=256,num_layers=4,num_nodes=1,env_num=0,conv_type=mpname).to(device)
# model.load_state_dict(torch.load(mname))
# model = GNNPredictor(hidden_dim=128,node_dim=16,num_layers=4,num_nodes=2,env_num=0,conv_type='GATE').to(device)
model.load_state_dict(torch.load('saves_scaffold4\epoch_150.pt'))
# test(model, device, test_graph_IR_loader,model_name='gate',save_fig_dir='ablation2')
test(model, device, test_graph_IR_loader,model_name='other',save_fig_dir='saves_scaffold4')
# train_mae = eval(model, device, train_graph_IR_loader)
# valid_mae = eval(model, device, valid_graph_IR_loader)
# print(train_mae, valid_mae)


# %%
model = GNNPredictor(hidden_dim=256,num_layers=4,num_nodes=1,env_num=0,conv_type='GINE').to(device)
model.load_state_dict(torch.load('saves_scaffold3/epoch_12.pt'))    # ['model_state_dict']
test(model, device, test_graph_IR_loader,model_name='scaffold',save_fig_dir='figures')    #
# train_mae = eval(model, device, train_graph_IR_loader)
# valid_mae = eval(model, device, valid_graph_IR_loader)
# print(train_mae, valid_mae)


# %%
result= pd.DataFrame(index=['test_mae','R_square','test_rmse'])


# %%
result[f'{mpname}']=[test_mae.item(),R_square.item(),test_rmse.item()]


# %%
result.T.to_excel('grid_search/result.xlsx', index=True)


# %%
torch.save(model.state_dict(), f'saves/model_save_{232}.pth')


# %%
y_pred, y_true,R_square,test_mae,test_rmse = test(model, device, test_graph_IR_loader,model_name='other data')#,save_fig_dir='figures')
print(float(R_square),float(test_mae),float(test_rmse))
# for i in range(len(y_pred)):
#     print(i,y_pred[i], y_true[i])


# %%
len(df_unique)


# %%
# all data
total_num = 8072
data_source = 'experiment'
# graph_IR_list = []
# for i in range(len(graph_IR_dataset)):
#     graph_IR_list.append(graph_IR_dataset[i])
# graph_IR_loader = DataLoader(graph_IR_list, batch_size=128,shuffle=False, num_workers=1)
y_pred, y_true, _, _ ,_ = test(model, device, test_graph_IR_loader,'GNN')
df_unique['y_true']= y_true
df_unique['y_test_pred']=y_pred
df_unique['difference']=np.abs(y_true-y_pred)
df_unique['type']=[classify_carbonyl(sm) for sm in df_unique['SMILES']]

# need_feature=['DOI','SMILES','IUPAC_NAME','IR_Characteristic_Peak','y_true','y_test_pred','difference']
need_feature=['SMILES','IR_Characteristic_Peak','IUPAC_NAME','type','y_true','y_test_pred','difference']   # 'DOI','cas','inchi',
data1=df_unique[need_feature]
data1.to_excel(f'test_data_{data_source}_test_{total_num}.xlsx')

bad_data = data1[(data1['difference'] > 18) & (data1['difference'] < 43)]
bad_data.to_excel(f'test_data1_{data_source}_{total_num}_bad_pre.xlsx', index=False)
good_data = data1[data1['difference'] < 18]
good_data.to_excel(f'test_data1_{data_source}_{total_num}_good_pre.xlsx', index=False)

# random_bad_samples = bad_data.sample(n=24, random_state=42)  # random_state 保证可复现
# combined_data = pd.concat([good_data, random_bad_samples])
# combined_data.to_excel(f'test_data1_{data_source}_{total_num}_combined.xlsx', index=False)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
df = pd.read_excel(r'D:\Jupyter\IR-DIAZO-KETONE-main\ablation2\ablation.xlsx', index_col=0)

# 获取模型名称和对应的性能指标
models = [model.split('_')[0] for model in df.index.tolist()]
r_square = df['R_square'].tolist()
test_rmse = df['test_rmse'].tolist()

# 将数据转换为DataFrame并按RMSE升序排序
data = pd.DataFrame({
    'Model': models,
    'R_square': r_square,
    'RMSE': test_rmse
}).sort_values('RMSE', ascending=False)  # 按RMSE升序排序

# 更新排序后的数据
models_sorted = data['Model'].tolist()
r_square_sorted = data['R_square'].tolist()
test_rmse_sorted = data['RMSE'].tolist()

# 计算柱子的位置
x = np.arange(len(models_sorted))
bar_width = 0.4

# 创建图形
fig, ax1 = plt.subplots(figsize=(17/2.54, 8/2.54), dpi=300)
plt.rcParams['figure.dpi'] = 300

# 设置背景色和网格线
ax1.set_facecolor("#CCCBCB7D")
ax1.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)

# 绘制test_rmse的柱状图（按RMSE升序排列）
# 修改点1：柱状图颜色改为 skyblue
bars = ax1.bar(x, test_rmse_sorted, bar_width, label='RMSE', color="#789FDA", zorder=2)
# ax1.set_ylabel('RMSE')
ax1.set_xticks(x)
ax1.set_xticklabels(models_sorted,rotation=30)
ax1.legend(loc='upper left')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)  # 旋转标签避免重叠
ax1.tick_params(axis='x', which='both', length=0)  # 刻度线长度设为0
ax1.tick_params(axis='y', which='both', length=1.2)  # 刻度线长度设为0
# ax1.set_xlabel('Model algorithm')

# 创建第二个y轴用于R_square的折线图（与柱状图顺序一致）
ax2 = ax1.twinx()
# 修改点2：折线图颜色改为 salmon
ax2.plot(x, r_square_sorted, 'salmon', linestyle='--', marker='o', label='$R^2$', zorder=2)
# ax2.set_ylabel('$R^2$')
ax2.legend(loc='upper right')
ax2.tick_params(axis='y', which='both', length=1.2)  # 刻度线长度设为0
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
# 标注数值
for bar, rmse in zip(bars, test_rmse_sorted):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{rmse:.3f}', 
             ha='center', va='bottom', fontsize=8)

for x_val, y_val in zip(x, r_square_sorted):
    ax2.text(x_val, y_val + 0.009 if y_val >= 0 else y_val - 0.01, f'{y_val:.3f}', 
             ha='center', va='bottom', fontsize=8)

# 设置y轴范围
buffer = 0.1
max_test_rmse = max(test_rmse_sorted) * (1 + buffer)
max_r_square = max(r_square_sorted) * (1 + buffer) if max(r_square_sorted) > 0 else 1
ax1.set_ylim(0, max_test_rmse + 10)  # 微调范围
ax2.set_ylim(0, max_r_square + 0.15)
# 移除所有边框
for spine in ax1.spines.values():
    spine.set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)

# 调整布局
plt.tight_layout()

# 保存图表
fig.savefig(r'D:\Jupyter\IR-DIAZO-KETONE-main\ablation2\ablation2.pdf', bbox_inches='tight', dpi=600)
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
df = pd.read_excel(r'D:\Jupyter\IR-DIAZO-KETONE-main\grid_search1\result111 - 副本.xlsx', index_col=0)

# 获取模型名称和对应的性能指标
models = [model.split('_')[0] for model in df.index.tolist()]
r_square = df['R_square'].tolist()
test_rmse = df['test_rmse'].tolist()

# 将数据转换为DataFrame并按RMSE升序排序
data = pd.DataFrame({
    'Model': models,
    'R_square': r_square,
    'RMSE': test_rmse
}).sort_values('RMSE', ascending=False)  # 按RMSE升序排序

# 更新排序后的数据
models_sorted = data['Model'].tolist()
r_square_sorted = data['R_square'].tolist()
test_rmse_sorted = data['RMSE'].tolist()

# 计算柱子的位置
x = np.arange(len(models_sorted))
bar_width = 0.4

# 创建图形
fig, ax1 = plt.subplots(figsize=(9/2.54, 8/2.54), dpi=300)
plt.rcParams['figure.dpi'] = 300

# 设置背景色和网格线
ax1.set_facecolor("#CCCBCB7D")
ax1.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)

# 绘制test_rmse的柱状图（按RMSE升序排列）
# 修改点1：柱状图颜色改为 skyblue
bars = ax1.bar(x, test_rmse_sorted, bar_width, label='RMSE', color="#789FDA", zorder=2)
# ax1.set_ylabel('RMSE')
ax1.set_xticks(x)
ax1.set_xticklabels(models_sorted)
ax1.legend(loc='upper left')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)  # 旋转标签避免重叠
ax1.tick_params(axis='x', which='both', length=0)  # 刻度线长度设为0
ax1.tick_params(axis='y', which='both', length=1.2)  # 刻度线长度设为0
# ax1.set_xlabel('Model algorithm')

# 创建第二个y轴用于R_square的折线图（与柱状图顺序一致）
ax2 = ax1.twinx()
# 修改点2：折线图颜色改为 salmon
ax2.plot(x, r_square_sorted, 'salmon', linestyle='--', marker='o', label='$R^2$', zorder=2)
# ax2.set_ylabel('$R^2$')
ax2.legend(loc='upper right')
ax2.tick_params(axis='y', which='both', length=1.2)  # 刻度线长度设为0
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)

# 标注数值
for bar, rmse in zip(bars, test_rmse_sorted):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{rmse:.3f}', 
             ha='center', va='bottom', fontsize=8)

for x_val, y_val in zip(x, r_square_sorted):
    ax2.text(x_val, y_val + 0.009 if y_val >= 0 else y_val - 0.01, f'{y_val:.3f}', 
             ha='center', va='bottom', fontsize=8)

# 设置y轴范围
buffer = 0.1
max_test_rmse = max(test_rmse_sorted) * (1 + buffer)
max_r_square = max(r_square_sorted) * (1 + buffer) if max(r_square_sorted) > 0 else 1
ax1.set_ylim(0, max_test_rmse + 10)  # 微调范围
ax2.set_ylim(0, max_r_square + 0.15)

# 移除所有边框
for spine in ax1.spines.values():
    spine.set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)

# 调整布局
plt.tight_layout()

# 保存图表
fig.savefig(r'D:\Jupyter\IR-DIAZO-KETONE-main\333\Model_performance_comparison_sorted.pdf', bbox_inches='tight', dpi=600)
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
df = pd.read_excel(r'D:\Jupyter\IR-DIAZO-KETONE-main\Other Model Compare.xlsx', index_col=0)

# 获取模型名称和对应的性能指标
models = [model.split('_')[0] for model in df.index.tolist()]
r_square = df['R_square'].tolist()
test_rmse = df['test_rmse'].tolist()

# 将数据转换为DataFrame并按RMSE升序排序
data = pd.DataFrame({
    'Model': models,
    'R_square': r_square,
    'RMSE': test_rmse
}).sort_values('RMSE', ascending=False)  # 按RMSE升序排序

# 更新排序后的数据
models_sorted = data['Model'].tolist()
r_square_sorted = data['R_square'].tolist()
test_rmse_sorted = data['RMSE'].tolist()

# 计算柱子的位置
x = np.arange(len(models_sorted))
bar_width = 0.4

# 创建图形
fig, ax1 = plt.subplots(figsize=(8/2.54, 7.5/2.54), dpi=300)
plt.rcParams['figure.dpi'] = 300

# 设置背景色和网格线
ax1.set_facecolor("#CCCBCB7D")
ax1.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)

# 绘制test_rmse的柱状图（按RMSE升序排列）
# 修改点1：柱状图颜色改为 skyblue
bars = ax1.bar(x, test_rmse_sorted, bar_width, label='RMSE', color="#789FDA", zorder=2)
# ax1.set_ylabel('RMSE')
ax1.set_xticks(x)
ax1.set_xticklabels(models_sorted)
ax1.legend(loc='upper left')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)  # 旋转标签避免重叠
ax1.tick_params(axis='x', which='both', length=0)  # 刻度线长度设为0
ax1.tick_params(axis='y', which='both', length=1.2)  # 刻度线长度设为0
# ax1.set_xlabel('Model algorithm')

# 创建第二个y轴用于R_square的折线图（与柱状图顺序一致）
ax2 = ax1.twinx()
# 修改点2：折线图颜色改为 salmon
ax2.plot(x, r_square_sorted, 'salmon', linestyle='--', marker='o', label='$R^2$', zorder=2)
# ax2.set_ylabel('$R^2$')
ax2.legend(loc='upper right')
ax2.tick_params(axis='y', which='both', length=1.2)  # 刻度线长度设为0
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
ax2.grid(False)

# 标注数值
for bar, rmse in zip(bars, test_rmse_sorted):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{rmse:.3f}', 
             ha='center', va='bottom', fontsize=8)

for x_val, y_val in zip(x, r_square_sorted):
    ax2.text(x_val, y_val + 0.009 if y_val >= 0 else y_val - 0.01, f'{y_val:.3f}', 
             ha='center', va='bottom', fontsize=8)

# 设置y轴范围
buffer = 0.1
max_test_rmse = max(test_rmse_sorted) * (1 + buffer)
max_r_square = max(r_square_sorted) * (1 + buffer) if max(r_square_sorted) > 0 else 1
ax1.set_ylim(0, max_test_rmse + 10)  # 微调范围
ax2.set_ylim(0, max_r_square + 0.15)

# 移除所有边框
for spine in ax1.spines.values():
    spine.set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)

# 调整布局
plt.tight_layout()

# 保存图表
fig.savefig(r'D:\Jupyter\IR-DIAZO-KETONE-main\Other Model Compare.pdf', bbox_inches='tight', dpi=600)
plt.show()


# %% [markdown]
# ## visualize_tensors

# %%
from collections import OrderedDict
device = 'cuda' if not torch.cuda.is_available() else 'cpu'
# original_model = GNNPredictor2(hidden_dim=128,num_layers=4,num_nodes=2,env_num=2)
# original_model.load_state_dict(torch.load('saves/model_save_24.pth'))
state_dict = torch.load(r'D:\Jupyter\IR-DIAZO-KETONE-main\grid_search\GINE_h256_n1\epoch_625.pt')['model_state_dict']
# 2. 过滤 predict 相关参数
filtered_state_dict = OrderedDict()
for k, v in state_dict.items():
    if not k.startswith('predict.'):
        filtered_state_dict[k] = v
        
# 3. 初始化模型
model = GNNGraph(hidden_dim=256, num_layers=4, num_nodes=1,node_dim=16).to(device)

# 4. 加载参数（允许部分加载）
model.load_state_dict(filtered_state_dict, strict=False)

# 5. 验证
model.eval()
print("Model loaded successfully!")


# %%
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch

def cluster_molecules(model, molecules, eps=0.5, min_samples=5):
    """
    在模型的潜在空间中对分子进行DBSCAN聚类
    
    参数:
        model: 训练好的图神经网络模型
        molecules: 分子图列表(Data对象)
        eps: DBSCAN邻域半径
        min_samples: 核心点最小邻域样本数
    
    返回:
        tuple: (聚类标签数组, 原始潜在空间嵌入, 聚类统计信息)
    """
    # 模型推理
    model.eval()
    batch = Batch.from_data_list(molecules).to(device)
    with torch.no_grad():
        embeddings = model(batch.x, batch.edge_index, batch.edge_attr, batch).numpy()
    print(embeddings.shape)
    # 标准化
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # DBSCAN聚类
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings_scaled)
    
    # 统计信息
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    noise_ratio = np.sum(clusters == -1) / len(clusters)
    stats = {
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'cluster_distribution': {c: np.sum(clusters == c) for c in unique_clusters}
    }
    
    return clusters, embeddings, stats

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP
from matplotlib.colors import ListedColormap
import seaborn as sns

def visualize_clusters(embeddings, clusters, method='pca', n_components=2,
                     title='Clustering Visualization', color_palette='husl',
                     alpha=0.7, marker_size=50, random_state=42):
    """
    降维可视化聚类结果
    
    参数:
        embeddings: 原始高维嵌入 (n_samples, n_features)
        clusters: 聚类标签数组
        method: 降维方法 ('pca', 'tsne', 'umap')
        n_components: 可视化维度 (2或3)
        title: 图表标题
        color_palette: 颜色主题
        alpha: 点透明度
        marker_size: 点大小
        random_state: 随机种子
    
    返回:
        matplotlib Figure对象
    """
    # 降维
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Method must be 'pca', 'tsne' or 'umap'")
    
    low_dim = reducer.fit_transform(embeddings)
    
    # 可视化设置
    unique_clusters = np.unique(clusters)
    palette = sns.color_palette(color_palette, len(unique_clusters))
    cmap = ListedColormap(palette)
    
    fig = plt.figure(figsize=(10, 8))
    
    # 2D/3D绘图
    if n_components == 2:
        scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], 
                            c=clusters, cmap=cmap, s=marker_size, 
                            alpha=alpha, edgecolor='white', linewidth=0.5)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
    else:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(low_dim[:, 0], low_dim[:, 1], low_dim[:, 2],
                           c=clusters, cmap=cmap, s=marker_size,
                           alpha=alpha, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_zlabel('Component 3', fontsize=12)
    
    # 图例
    legend_elements = [
        plt.Line2D([], [], marker='o', color=palette[i], linestyle='',
                  markersize=10, label=f'Cluster {c}' if c != -1 else 'Noise')
        for i, c in enumerate(unique_clusters)
    ]
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title, pad=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig


# %%
# 假设已有模型和分子数据
# molecules = [mol for mol in graph_IR_dataset]  # 分子图列表

# 步骤1: 聚类
clusters, embeddings, stats = cluster_molecules(
    model, 
    molecules,
    eps=6, 
    min_samples=15
)
print(f"聚类统计: {stats}")

# 步骤2: 可视化
fig = visualize_clusters(
    embeddings,
    clusters,
    method='tsne',
    title=f"Molecular Clusters (Found {stats['n_clusters']} clusters)",
    color_palette='viridis'
)

plt.show()
fig.savefig(f'333\\visualize.pdf', bbox_inches='tight', dpi=300)


# %%


# %%


# %%
def visualize_tensors(tensors, labels, method='umap', n_components=2,
                     title='Clustered Tensor Visualization', 
                     color_palette='tab10', alpha=0.6, marker_size=80,
                     random_state=40, class_names=None, **kwargs):
    """
    优化后的可视化函数，使同类点更聚集
    
    参数:
        class_names: 自定义类别名称列表（如 ['Amide', 'Ether', 'Acyl halide'])
        **kwargs: 传递给降维算法的参数(如perplexity, n_neighbors等)
    """
    # 检查输入
    if len(tensors) != len(labels):
        raise ValueError("tensors和labels的长度必须相同")
    
    # 展平数据
    tensors_flat = tensors.reshape(tensors.shape[0], -1) if len(tensors.shape) > 2 else tensors
    
    # 标准化
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # tensors_flat = scaler.fit_transform(tensors_flat)
    
    # 降维
    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=random_state,
                      perplexity=kwargs.get('perplexity', 50),
                      learning_rate=kwargs.get('learning_rate', 150),
                      n_iter=kwargs.get('n_iter', 2000))
    elif method.lower() == 'umap':
        from umap import UMAP
        reducer = UMAP(n_components=n_components, random_state=random_state,
                      n_neighbors=kwargs.get('n_neighbors', 10),
                      min_dist=kwargs.get('min_dist', 0.05))
    else:
        raise ValueError("method必须是'pca'、'tsne'或'umap'")
    
    embeddings = reducer.fit_transform(tensors_flat)
    
    # 可视化
    import seaborn as sns
    # import  pyplot as plt
    from matplotlib.colors import ListedColormap
    
    unique_labels = np.unique(labels)
    colors = sns.color_palette(color_palette, n_colors=len(unique_labels))
    cmap = ListedColormap(colors)
    
    fig = plt.figure(figsize=(9/2.54, 9/2.54))
    
    if n_components == 2:
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                            c=labels, cmap=cmap, alpha=alpha, s=marker_size,
                            edgecolor='white', linewidth=0.32)
        plt.xlabel('Principle Component 1', fontsize=9)
        plt.ylabel('Principle Component 2', fontsize=9)
    else:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                           c=labels, cmap=cmap, alpha=alpha, s=marker_size,
                           edgecolor='white', linewidth=0.32)
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_zlabel('Component 3', fontsize=12)
    
    # plt.title(title, fontsize=10, pad=8)
    plt.grid(True, alpha=0.3)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    # 创建图例（支持自定义标签）
    if class_names is None:
        class_names = [f'Class {label}' for label in unique_labels]
    elif len(class_names) != len(unique_labels):
        raise ValueError("class_names的长度必须与唯一标签的数量相同")

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                        markersize=6, label=class_names[i])  # 缩小标记大小
              for i in range(len(unique_labels))]

    # 调整图例样式（字体8号，缩小边框和间距）
    plt.legend(
        handles=handles,
        loc='upper right',
        framealpha=0.9,
        fontsize=8,                  # 图例文字大小
        handletextpad=0.5,           # 标记与文字的间距
        borderpad=0.5,               # 边框与内容的间距
        labelspacing=0.5,            # 标签间的垂直间距
        handlelength=1.5             # 标记的长度
    )
    plt.tight_layout()
    return fig


# %%
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

def process_and_visualize(new_model, amide_moleculesN, amide_moleculesO, amide_moleculesF):
    # 确保模型在评估模式
    new_model.eval()
    
    # 准备标签 (0: N, 1: O, 2: F)
    labels = torch.cat([
        torch.zeros(len(amide_moleculesN)),  # N类标签为0
        torch.ones(len(amide_moleculesO)),   # O类标签为1
        torch.full((len(amide_moleculesF),), 2)  # F类标签为2
    ])
    
    # 合并所有分子图
    all_molecules = amide_moleculesN + amide_moleculesO + amide_moleculesF
    
    # 创建批次数据
    batch_data = Batch.from_data_list(all_molecules)
    
    # 获取模型输出 (不需要梯度)
    with torch.no_grad():
        outputs = new_model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data)
    
    # 沿axis=0拼接所有输出 (63×d_model)
    tensors = outputs.cpu().numpy()
    
    # 可视化
    fig = visualize_tensors(
        tensors, 
        labels, 
        method='tsne', 
        n_components=2, 
        title='Tensor Visualization', 
        color_palette='viridis', 
        alpha=0.6, 
        marker_size=30, 
        random_state=42, 

        perplexity=200, 
        learning_rate=200, 
        n_iter=600, 
        
        n_neighbors=27, 
        min_dist=0.07,
        class_names=['Amide', 'Ether', 'Acyl halide']  # 直接传入自定义标签
    )
    return fig

# 可视化
fig = process_and_visualize(model, amide_moleculesN, amide_moleculesO, amide_moleculesF)
plt.show()
fig.savefig(f'333\\visualize.pdf', bbox_inches='tight', dpi=300)


# %%
from typing import Optional
from math import sqrt
from inspect import signature
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx

EPS = 1e-15

class GNNExplainer(torch.nn.Module):
    coeffs = {
        'node_feat_reduction': 'mean',
        'node_feat_size': 1.0,
        'node_feat_ent': 0.1,
        'edge_reduction': 'sum',
        'edge_size': 0.004,
        'edge_ent': 1.0,
        'edge_feat_reduction': 'mean',  # 新增边特征归约方式
        'edge_feat_size': 1.0,  # 新增边特征大小正则化系数
        'edge_feat_ent': 0.1,  # 新增边特征熵正则化系数
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, **kwargs):
        super().__init__()
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.coeffs.update(kwargs)

    def __set_masks__(self, x, edge_index, edge_attr=None, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        # 节点特征掩码
        if self.feat_mask_type == 'individual_feature':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)
        elif self.feat_mask_type == 'scalar':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, 1) * std)
        else:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)

        # 边掩码
        std_edge = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std_edge)
        if not self.allow_edge_mask:
            self.edge_mask.requires_grad_(False)
            self.edge_mask.fill_(float('inf'))
        self.loop_mask = edge_index[0] != edge_index[1]

        # 边特征掩码
        self.edge_feat_mask = None
        if edge_attr is not None:
            F_edge = edge_attr.size(1)
            if self.feat_mask_type == 'individual_feature':
                self.edge_feat_mask = torch.nn.Parameter(torch.randn(E, F_edge) * std)
            elif self.feat_mask_type == 'scalar':
                self.edge_feat_mask = torch.nn.Parameter(torch.randn(E, 1) * std)
            else:
                self.edge_feat_mask = torch.nn.Parameter(torch.randn(1, F_edge) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
                module.__loop_mask__ = self.loop_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
                module.__loop_mask__ = None
        self.node_feat_mask = None
        self.edge_mask = None
        self.edge_feat_mask = None
        self.loop_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, subset, kwargs

    def __loss__(self, node_idx, log_logits, pred_label):
        if self.return_type == 'regression':
            loss = -torch.cdist(log_logits[node_idx], pred_label[node_idx]) if node_idx != -1 else -torch.norm(log_logits-pred_label, p=2)
        else:
            loss = -log_logits[node_idx, pred_label[node_idx]] if node_idx != -1 else -log_logits[0, pred_label[0]]

        # 节点特征掩码正则化
        m_node = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m_node)
        ent_node = -m_node * torch.log(m_node + EPS) - (1 - m_node) * torch.log(1 - m_node + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent_node.mean()

        # 边掩码正则化
        m_edge = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m_edge)
        ent_edge = -m_edge * torch.log(m_edge + EPS) - (1 - m_edge) * torch.log(1 - m_edge + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent_edge.mean()

        # 边特征掩码正则化 (如果存在)
        if self.edge_feat_mask is not None:
            m_edge_feat = self.edge_feat_mask.sigmoid()
            edge_feat_reduce = getattr(torch, self.coeffs['edge_feat_reduction'])
            loss = loss + self.coeffs['edge_feat_size'] * edge_feat_reduce(m_edge_feat)
            ent_edge_feat = -m_edge_feat * torch.log(m_edge_feat + EPS) - (1 - m_edge_feat) * torch.log(1 - m_edge_feat + EPS)
            loss = loss + self.coeffs['edge_feat_ent'] * ent_edge_feat.mean()

        return loss
    
    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def explain_graph(self, x, edge_index, edge_attr=None, **kwargs):
        self.model.eval()
        self.__clear_masks__()

        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index, edge_attr)
        self.to(x.device)
        
        parameters = [self.node_feat_mask, self.edge_mask] if self.allow_edge_mask else [self.node_feat_mask]
        if self.edge_feat_mask is not None:
            parameters.append(self.edge_feat_mask)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            h_edge = edge_attr * self.edge_feat_mask.sigmoid() if edge_attr is not None else None
            
            out = self.model(x=h, edge_index=edge_index, edge_attr=h_edge, **kwargs)
            
            if self.return_type == 'regression':
                loss = self.__loss__(-1, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(-1, log_logits, pred_label)
                
            loss.backward()
            optimizer.step()
            if self.log:
                pbar.update(1)
        if self.log:
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
        edge_mask = self.edge_mask.detach().sigmoid()
        edge_feat_mask = self.edge_feat_mask.detach().sigmoid() if self.edge_feat_mask is not None else None

        self.__clear_masks__()
        return node_feat_mask, edge_mask, edge_feat_mask

    def explain_node(self, node_idx, x, edge_index, edge_attr=None, **kwargs):
        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)
        # 提取子图 (包含边特征处理)
        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
            self.__subgraph__(node_idx, x, edge_index, **kwargs)
        
        # 处理边特征
        if edge_attr is not None:
            edge_attr_sub = edge_attr[hard_edge_mask]
        else:
            edge_attr_sub = None

        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr_sub, **kwargs)
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index, edge_attr_sub)
        self.to(x.device)
        
        parameters = [self.node_feat_mask, self.edge_mask]
        if self.edge_feat_mask is not None:
            parameters.append(self.edge_feat_mask)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            h_edge = edge_attr_sub * self.edge_feat_mask.sigmoid() if edge_attr_sub is not None else None
            
            out = self.model(x=h, edge_index=edge_index, edge_attr=h_edge, **kwargs)
            
            if self.return_type == 'regression':
                loss = self.__loss__(mapping, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(mapping, log_logits, pred_label)
                
            loss.backward()
            optimizer.step()
            if self.log:
                pbar.update(1)
        if self.log:
            pbar.close()

        # 处理节点特征掩码
        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type != 'feature':
            new_mask = x.new_zeros(x.size(0), node_feat_mask.size(1))
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()

        # 处理边掩码
        full_edge_mask = self.edge_mask.new_zeros(num_edges)
        full_edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        # 处理边特征掩码
        if self.edge_feat_mask is not None:
            edge_feat_mask = self.edge_feat_mask.new_zeros(num_edges, edge_attr.size(1))
            edge_feat_mask[hard_edge_mask] = self.edge_feat_mask.detach().sigmoid()
        else:
            edge_feat_mask = None

        self.__clear_masks__()
        return node_feat_mask, full_edge_mask, edge_feat_mask

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, edge_y=None, node_alpha=None,
                           seed=10, **kwargs):
        r"""Visualizes the subgraph given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`-1` to explain graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node.
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        if edge_y is None:
            edge_color = ['black'] * edge_index.size(1)
        else:
            colors = list(plt.rcParams['axes.prop_cycle'])
            edge_color = [
                colors[i % len(colors)]['color']
                for i in edge_y[hard_edge_mask]
            ]

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'],
                        edge_attrs=['att', 'edge_color'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G, seed=seed)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    color=data['edge_color'],
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))

        if node_alpha is None:
            nx.draw_networkx_nodes(G, pos, node_color=y.tolist(),
                                   **node_kwargs)
        else:
            node_alpha_subset = node_alpha[subset]
            assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
            nx.draw_networkx_nodes(G, pos, alpha=node_alpha_subset.tolist(),
                                   node_color=y.tolist(), **node_kwargs)

        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = num_nodes if num_nodes is not None else edge_index.max().item() + 1

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


# %% [markdown]
# ## GNNexplainer

# %%
# from gnnexplainer import GNNExplainer
from GNNexplainer import GNNExplainer


# %%
from typing import Optional
from math import sqrt
from inspect import signature
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx

EPS = 1e-15

class GNNExplainer(torch.nn.Module):
    coeffs = {
        'node_feat_reduction': 'mean',
        'node_feat_size': 1.0,
        'node_feat_ent': 0.1,
        'edge_reduction': 'sum',
        'edge_size': 0.004,
        'edge_ent': 1.0,
        'edge_feat_reduction': 'mean',  # 新增边特征归约方式
        'edge_feat_size': 1.0,  # 新增边特征大小正则化系数
        'edge_feat_ent': 0.1,  # 新增边特征熵正则化系数
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, **kwargs):
        super().__init__()
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.coeffs.update(kwargs)

    def __set_masks__(self, x, edge_index, edge_attr=None, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        # 节点特征掩码
        if self.feat_mask_type == 'individual_feature':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)
        elif self.feat_mask_type == 'scalar':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, 1) * std)
        else:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)

        # 边掩码
        std_edge = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std_edge)
        if not self.allow_edge_mask:
            self.edge_mask.requires_grad_(False)
            self.edge_mask.fill_(float('inf'))
        self.loop_mask = edge_index[0] != edge_index[1]

        # 边特征掩码
        self.edge_feat_mask = None
        if edge_attr is not None:
            F_edge = edge_attr.size(1)
            if self.feat_mask_type == 'individual_feature':
                self.edge_feat_mask = torch.nn.Parameter(torch.randn(E, F_edge) * std)
            elif self.feat_mask_type == 'scalar':
                self.edge_feat_mask = torch.nn.Parameter(torch.randn(E, 1) * std)
            else:
                self.edge_feat_mask = torch.nn.Parameter(torch.randn(1, F_edge) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
                module.__loop_mask__ = self.loop_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
                module.__loop_mask__ = None
        self.node_feat_mask = None
        self.edge_mask = None
        self.edge_feat_mask = None
        self.loop_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, subset, kwargs

    def __loss__(self, node_idx, log_logits, pred_label):
        if self.return_type == 'regression':
            loss = -torch.cdist(log_logits[node_idx], pred_label[node_idx]) if node_idx != -1 else -torch.norm(log_logits-pred_label, p=2)
        else:
            loss = -log_logits[node_idx, pred_label[node_idx]] if node_idx != -1 else -log_logits[0, pred_label[0]]

        # 节点特征掩码正则化
        m_node = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m_node)
        ent_node = -m_node * torch.log(m_node + EPS) - (1 - m_node) * torch.log(1 - m_node + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent_node.mean()

        # 边掩码正则化
        m_edge = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m_edge)
        ent_edge = -m_edge * torch.log(m_edge + EPS) - (1 - m_edge) * torch.log(1 - m_edge + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent_edge.mean()

        # 边特征掩码正则化 (如果存在)
        if self.edge_feat_mask is not None:
            m_edge_feat = self.edge_feat_mask.sigmoid()
            edge_feat_reduce = getattr(torch, self.coeffs['edge_feat_reduction'])
            loss = loss + self.coeffs['edge_feat_size'] * edge_feat_reduce(m_edge_feat)
            ent_edge_feat = -m_edge_feat * torch.log(m_edge_feat + EPS) - (1 - m_edge_feat) * torch.log(1 - m_edge_feat + EPS)
            loss = loss + self.coeffs['edge_feat_ent'] * ent_edge_feat.mean()

        return loss
    
    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def explain_graph(self, x, edge_index, edge_attr=None, **kwargs):
        self.model.eval()
        self.__clear_masks__()

        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index, edge_attr)
        self.to(x.device)
        
        parameters = [self.node_feat_mask, self.edge_mask] if self.allow_edge_mask else [self.node_feat_mask]
        if self.edge_feat_mask is not None:
            parameters.append(self.edge_feat_mask)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            h_edge = edge_attr * self.edge_feat_mask.sigmoid() if edge_attr is not None else None
            
            out = self.model(x=h, edge_index=edge_index, edge_attr=h_edge, **kwargs)
            
            if self.return_type == 'regression':
                loss = self.__loss__(-1, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(-1, log_logits, pred_label)
                
            loss.backward()
            optimizer.step()
            if self.log:
                pbar.update(1)
        if self.log:
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
        edge_mask = self.edge_mask.detach().sigmoid()
        edge_feat_mask = self.edge_feat_mask.detach().sigmoid() if self.edge_feat_mask is not None else None

        self.__clear_masks__()
        return node_feat_mask, edge_mask, edge_feat_mask

    def explain_node(self, node_idx, x, edge_index, edge_attr=None, **kwargs):
        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)
        # 提取子图 (包含边特征处理)
        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
            self.__subgraph__(node_idx, x, edge_index, **kwargs)
        
        # 处理边特征
        if edge_attr is not None:
            edge_attr_sub = edge_attr[hard_edge_mask]
        else:
            edge_attr_sub = None

        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr_sub, **kwargs)
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index, edge_attr_sub)
        self.to(x.device)
        
        parameters = [self.node_feat_mask, self.edge_mask]
        if self.edge_feat_mask is not None:
            parameters.append(self.edge_feat_mask)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            h_edge = edge_attr_sub * self.edge_feat_mask.sigmoid() if edge_attr_sub is not None else None
            
            out = self.model(x=h, edge_index=edge_index, edge_attr=h_edge, **kwargs)
            
            if self.return_type == 'regression':
                loss = self.__loss__(mapping, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(mapping, log_logits, pred_label)
                
            loss.backward()
            optimizer.step()
            if self.log:
                pbar.update(1)
        if self.log:
            pbar.close()

        # 处理节点特征掩码
        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type != 'feature':
            new_mask = x.new_zeros(x.size(0), node_feat_mask.size(1))
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()

        # 处理边掩码
        full_edge_mask = self.edge_mask.new_zeros(num_edges)
        full_edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        # 处理边特征掩码
        if self.edge_feat_mask is not None:
            edge_feat_mask = self.edge_feat_mask.new_zeros(num_edges, edge_attr.size(1))
            edge_feat_mask[hard_edge_mask] = self.edge_feat_mask.detach().sigmoid()
        else:
            edge_feat_mask = None

        self.__clear_masks__()
        return node_feat_mask, full_edge_mask, edge_feat_mask
    

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                          threshold=None, edge_y=None, node_alpha=None,
                          seed=10, save_path=None, **kwargs):
        r"""Visualizes the subgraph with low-saturation blue nodes and PDF saving.
        
        Args:
            save_path (str, optional): Path to save the figure as PDF. If None, won't save.
            Other args same as original.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        assert edge_mask.size(0) == edge_index.size(1)

        # Create figure with white background
        plt.figure(facecolor='white',figsize=(15/2.54, 12/2.54))
        
        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None
        else:
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        # Node coloring with low-saturation blue
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float)
            if y.max().item() > 0:  # Normalize if not all zeros
                y = y / y.max().item()
        
        # Create custom low-saturation blue colormap
        blue_cmap = LinearSegmentedColormap.from_list(
            'low_sat_blue', ['#e6f2ff', '#0066cc'])  # Light to medium blue

        if edge_y is None:
            edge_color = ['#5c5c5c'] * edge_index.size(1)  # Gray edges
        else:
            colors = list(plt.rcParams['axes.prop_cycle'])
            edge_color = [
                colors[i % len(colors)]['color']
                for i in edge_y[hard_edge_mask]
            ]

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'],
                        edge_attrs=['att', 'edge_color'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs.update({
            'node_size': kwargs.get('node_size', 800),
            'cmap': blue_cmap,
            'vmin': 0,
            'vmax': 1,
            'edgecolors': '#333333',
            'linewidths': 1.0
        })

        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs.update({
            'font_size': kwargs.get('font_size', 10),
            'font_color': 'black'
        })

        pos = nx.spring_layout(G, seed=seed)
        ax = plt.gca()
        
        # Draw edges with adjusted styling
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.2),  # Increased minimum alpha
                    color=data['edge_color'],
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))

        # Draw nodes with new color scheme
        if node_alpha is None:
            nx.draw_networkx_nodes(G, pos, node_color=y.tolist(),
                                  **node_kwargs)
        else:
            node_alpha_subset = node_alpha[subset]
            assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
            nx.draw_networkx_nodes(G, pos, alpha=node_alpha_subset.tolist(),
                                  node_color=y.tolist(), **node_kwargs)

        nx.draw_networkx_labels(G, pos, **label_kwargs)
        
        plt.axis('off')
        plt.tight_layout(pad=1.0)  # 增加图形内部的padding
        # Save to PDF if path is provided
        if save_path is not None:
            plt.savefig(save_path ,dpi=300)
            print(f"Graph saved to {save_path}")

        return ax, G


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = num_nodes if num_nodes is not None else edge_index.max().item() + 1

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


# %%
# 创建新模型
state_dict = torch.load('grid_search/GINE_h256_n1/epoch_600.pt')
model = GNNPredictor(hidden_dim=256,num_layers=4,num_nodes=1,env_num=0)
# 过滤参数
# 将参数赋值给新模型
model.load_state_dict(state_dict)


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to('cpu')
data = test_graph_IR_dataset[384].to('cpu')
# data = test_graph_IR_dataset[1].to('cpu')
display(Chem.MolFromSmiles(data.smiles))
# model = model.to(device)
# data = graph_IR_dataset[0].to(device)
# data = test_graph_IR_dataset[0].to(device)
explainer = GNNExplainer(model, epochs=200, return_type='regression',feat_mask_type = 'feature', allow_edge_mask=True, log=False,num_hops=3)      # feature     individual_feature
# 'feature', 'individual_feature', 'scalar'
node_feat_mask, edge_mask, edge_feat_mask = explainer.explain_graph(data.x, data.edge_index, edge_attr=data.edge_attr,data=data)


# %%
node_feat_mask, edge_mask, edge_feat_mask   # [0.2271, 0.3311, 0.1334, 0.2512, 0.5029, 0.5175, 0.2904, 0.2586, 0.1412,0.6207, 0.7926, 0.2727, 0.3876, 0.3404, 0.2489, 0.1364]


# %%
node_feat_mask, edge_mask, edge_feat_mask = explainer.explain_node(9,data.x, data.edge_index, edge_attr=data.edge_attr,data=data)


# %%
data.carbonyl_mask


# %%
# 子图
from rdkit import Chem
from rdkit.Chem import Draw
ax,G = explainer.visualize_subgraph(9, data.edge_index,edge_mask,threshold=0.1, save_path = r'D:\Jupyter\IR-DIAZO-KETONE-main\asds.pdf') # threshold=0.1
mol = Chem.MolFromSmiles(data.smiles)
# mol = Chem.AddHs(mol)
for atom in mol.GetAtoms():
    atom.SetAtomMapNum(atom.GetIdx())

display(mol)
Draw.MolToFile(mol, r'D:\Jupyter\IR-DIAZO-KETONE-main\molecule2.svg')


# %%
df = pd.DataFrame(results)
df.to_excel('feature_importance.xlsx')


# %%
import pandas as pd
import numpy as np

# 定义特征名称
x_labels = [
    'Atomic Number', 'Connectivity', 'Implicit Valence', 
    'In Ring', 'Hybridization', 'Electronegativity', 'Covalent Radius',
    'Aromaticity', 'Formal Charge', 'Ring Size',
    'Neighbor Electronegativity', 'Neighbor Degree', 'Neighbor Aromaticity',
    'Neighbor in Ring', 'Neighbor Double Bond', 'Neighbor Triple Bond'
]

# 初始化 DataFrame（基团名为行索引，特征和 Count 为列）
df = pd.DataFrame(
    index=list(results.keys()),  # 基团名作为行索引
    columns=x_labels + ['Count']  # 特征 + Count 列
)

# 填充数据
for group_name, stats in results.items():
    if stats is not None:
        # 填充特征重要性分数
        df.loc[group_name, x_labels] = stats['mean_importance']
        # 填充 Count
        df.loc[group_name, 'Count'] = stats['count']

# 按 Count 降序排序
df = df.sort_values('Count', ascending=False)
# 保存为 Excel
output_file = 'carbonyl_feature_importance_matrix.xlsx'
df.to_excel(output_file, float_format="%.4f")  # 保留4位小数

print(f"数据已保存至 {output_file}")


# %%
import matplotlib.pyplot as plt
import numpy as np

# 原始数据
importance_scores = results['Carboxyl']['mean_importance']
def draw_importance(name,importance_scores):
    x_labels = ['Atomic Number', 'Connectivity', 'Implicit Valence', 
                'In Ring', 'Hybridization', 'Electronegativity', 'Covalent Radius',
                'Aromaticity', 'Formal Charge', 'Ring Size',
                'Neighbor Electronegativity', 'Neighbor Degree', 'Neighbor Aromaticity',
                'Neighbor in Ring', 'Neighbor Double Bond', 'Neighbor Triple Bond']

    # 将数据和标签组合并按分值降序排序
    features = list(zip(importance_scores, x_labels))
    features.sort(reverse=True)  # 按分值从大到小排序
    top_features = features[:10]  # 取前10个

    # 分离排序后的分数和标签（反转顺序使最高分在上方）
    top_scores = [x[0] for x in top_features][::-1]  # 反转顺序
    top_labels = [x[1] for x in top_features][::-1]  # 反转顺序

    # 创建图形
    plt.figure(figsize=(6/2.54, 4/2.54), dpi=300)
    ax = plt.gca()

    # 设置颜色（从高到低渐变）
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_scores)))
    # 绘制水平条形图（最高分在最上方）
    bars = ax.barh(range(len(top_labels)), top_scores, color=colors, height=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # 边框宽度
        spine.set_color('black')  # 边框颜色
    # 自定义Y轴刻度
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=6)

    # ax.set_facecolor("#DAD8D87C")
    ax.grid(False, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01,  # 数值标签向右偏移
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                va='center', ha='left',
                fontsize=6)
    ax.tick_params(axis='both', 
                  which='major',
                  direction='in',    # 朝外
                  length=3,           # 减小长度
                  width=0.5,          # 减小宽度
                  color='black',
                  pad=1)      # 颜色
    # 美化图形
    # plt.xlabel('Importance Score', fontsize=3, labelpad=1)  # 减小x轴标签与轴的距离
    plt.title(f'{name}', fontsize=7, pad=5)
    plt.xlim(0, max(top_scores)*1.25)  # 扩展x轴范围留出标签空间
    # plt.grid(axis='x', linestyle='--', alpha=0.3)  # 添加辅助网格线
    ax.set_xticks(np.arange(0, x_max, 0.25))
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.tight_layout()
    plt.subplots_adjust(left=0.01,   # 左边界
                    right=0.99,   # 右边界
                    bottom=0.01,  # 下边界
                    top=0.99)      # 上边界
    plt.savefig(f'figures/top10_feature_importance_ordered{name}.pdf', bbox_inches='tight', dpi=300)
    plt.show()
for name,dic in results.items():
    draw_importance(name, dic['mean_importance'].tolist())


# %%
# 边特征
import matplotlib.pyplot as plt
import numpy as np

# 原始数据
importance_scores = edge_feat_mask.squeeze()
x_labels = ['Bond type', 'Conjugation status', 'Ring membership', 'Aromaticity', 'Polarity']

# 将数据和标签组合并按分值降序排序
features = list(zip(importance_scores, x_labels))
features.sort(reverse=True)  # 按分值从大到小排序
top_features = features[:5]  # 取前10个

# 分离排序后的分数和标签（反转顺序使最高分在上方）
top_scores = [x[0] for x in top_features][::-1]  # 反转顺序
top_labels = [x[1] for x in top_features][::-1]  # 反转顺序

# 创建图形
plt.figure(figsize=(9/2.54, 6/2.54), dpi=300)
ax = plt.gca()

# 设置颜色（从高到低渐变）
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_scores)))

# 绘制水平条形图（最高分在最上方）
bars = ax.barh(range(len(top_labels)), top_scores, color=colors, height=0.6)

# 自定义Y轴刻度
ax.set_yticks(range(len(top_labels)))
ax.set_yticklabels(top_labels, fontsize=8)

# 添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.01,  # 数值标签向右偏移
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            va='center', ha='left',
            fontsize=8)

# 美化图形
plt.xlabel('Importance Score', fontsize=8)
plt.title('Top 10 Feature Importance Scores (Highest at Top)', fontsize=8, pad=5)
plt.xlim(0, max(top_scores)*1.25)  # 扩展x轴范围留出标签空间
# plt.grid(axis='x', linestyle='--', alpha=0.3)  # 添加辅助网格线
plt.yticks(fontsize=6)
plt.xticks(fontsize=8)
plt.tight_layout()
# plt.savefig('top10_feature_importance_ordered2.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %%
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# 定义羰基类别及其 SMARTS 模式
CARBONYL_GROUPS = {
    'Carboxyl': 'C(=O)[OH]',     # 羧酸
    'Aldehyde': 'C(=O)[H]',      # 醛基
    'Amide': '[C,c](=O)[N,n]',       # 酰胺
    'Ester': '[C,c](=O)[o,O][C,c]',      # 酯基   '[C,c](=O)[o,O][C,c]'
    'Ketone': '[c,C][c,C](=O)[c,C]', # 酮羰基
    'Acyl Halide': '[F,Cl,Br,I]C=O',  # 酰卤
}

def classify_carbonyl(smiles):
    """根据 SMILES 字符串判断羰基类型"""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is None:
        return None
    
    matches = {}
    for name, smarts in CARBONYL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            matches[name] = True
    
    return list(matches.keys()) if matches else ['Other']

from collections import defaultdict

# 初始化分组字典
grouped_molecules = defaultdict(list)

for data in test_graph_IR_dataset:
    smiles = data.smiles  # 假设每个 data 有 smiles 属性
    if not smiles:
        continue
    
    # 分类羰基类型
    carbonyl_types = classify_carbonyl(smiles)
    
    # 按类型分组
    for c_type in carbonyl_types:
        grouped_molecules[c_type].append(data)

# 打印各组数量
for c_type, molecules in grouped_molecules.items():
    print(f"{c_type}: {len(molecules)} 个分子")

import pandas as pd

# 转换为 DataFrame
results = []
for c_type, molecules in grouped_molecules.items():
    for data in molecules:
        results.append({
            'SMILES': data.smiles,
            'Carbonyl_Type': c_type,
            'IR_Peak': getattr(data, 'ir_peak', None)  # 假设有红外峰数据
        })

df = pd.DataFrame(results)
df.to_csv('carbonyl_group_classification.csv', index=False)


# %%
import torch
import numpy as np
from tqdm import tqdm

# 初始化模型和 Explainer
device = 'cpu'
model = model.to(device)
model.eval()
model.train()  # 启用梯度以支持 GNNExplainer

explainer = GNNExplainer(
    model,
    epochs=200,
    return_type='regression',
    feat_mask_type='feature',
    allow_edge_mask=True,
    log=False,
    num_hops=3
)

# 存储各组结果
results = {}

# 遍历已分组的分子
for group_name, molecules in grouped_molecules.items():
    # if group_name !='Carboxyl':
    #     continue
    print(f"\n===== 计算 {group_name} 组的特征重要性 =====")
    
    # 初始化当前组的统计
    group_importance = torch.zeros(molecules[0].x.shape[1])  # 假设所有分子特征维度相同
    valid_count = 0

    # 计算当前组的重要性
    for data in tqdm(molecules, desc=f"Processing {group_name}"):
        try:
            data = data.to(device)
            if data.x.numel() == 0:
                continue

            # 获取特征重要性
            node_feat_mask, _, _ = explainer.explain_graph(
                data.x,
                data.edge_index,
                edge_attr=data.edge_attr,
                data = data
            )
            
            # 累加重要性
            group_importance += node_feat_mask.cpu().detach()
            valid_count += 1

        except Exception as e:
            print(f"跳过 {group_name} 组的样本，错误: {e}")
            continue

    # 保存结果
    if valid_count > 0:
        results[group_name] = {
            'mean_importance': (group_importance / valid_count).numpy(),
            'count': valid_count
        }
    else:
        results[group_name] = None


# %%
# 统计
import torch
import numpy as np
from tqdm import tqdm  # 进度条（可选）

# 确保模型在 CPU（GNNExplainer 在 CPU 更稳）
device = 'cpu'
model = model.to(device)
model.eval()  # 设置为评估模式
model.train()
num_features = test_graph_IR_dataset[0].x.shape[1]  # 假设所有图特征维度一致
total_importance = torch.zeros(num_features)  # 累加器
valid_count = 0

# 初始化 Explainer（放在循环外）
explainer = GNNExplainer(
    model,
    epochs=200,
    return_type='regression',
    feat_mask_type='feature',
    allow_edge_mask=True,
    log=False,
    num_hops=3
)

print(f"正在计算 {len(test_graph_IR_dataset)} 个分子的平均特征重要性...")

for i in tqdm(range(395,len(test_graph_IR_dataset)), desc="Explaining"):
    try:
        data = test_graph_IR_dataset[i].to(device)
        if data.x.numel() == 0:
            continue
        # # 启用梯度
        # data.x.requires_grad_(True)
        # if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        #     data.edge_attr.requires_grad_(True)

        # 获取 node_feat_mask（形状: [num_features]）
        node_feat_mask, _, _ = explainer.explain_graph(
            data.x, 
            data.edge_index, 
            edge_attr=data.edge_attr,
            data= data
        )
        # 累加（确保是 CPU tensor）
        total_importance += node_feat_mask.cpu()
        valid_count += 1
        print(node_feat_mask)
    except Exception as e:
        print(f"跳过样本 {i}，错误: {e}")
        continue

# 计算平均
if valid_count > 0:
    mean_importance = total_importance / valid_count
    print(f"成功处理 {valid_count} / {len(test_graph_IR_dataset)} 个样本")
else:
    raise ValueError("没有有效样本！")

# 转为 NumPy 用于后续分析/绘图
mean_importance_np = mean_importance.numpy()
print("平均重要性分数:", mean_importance_np)


# %% [markdown]
# ## PSA + GNNExplainer
# 

# %%
# PSA +GNNExplainer
import torch
from torch_geometric.data import Data

def gnne_psa_signed_importance(
    model,
    data: Data,
    node_feat_mask,
    edge_mask=None,
    edge_feat_mask=None,
    device='cpu',
    baseline='zero',  # 可选: 'zero', 'mean'
):
    """
    Compute signed feature importance by combining GNNExplainer mask with PSA.
    
    Args:
        model: Trained GNN model (eval mode)
        data: Input graph (with x, edge_index, edge_attr, etc.)
        node_feat_mask: [F] from GNNExplainer (feat_mask_type='feature')
        device: device to run on
        baseline: how to mask – 'zero' or 'mean' feature per dimension
    
    Returns:
        signed_importance: [F] – sign indicates direction, magnitude from GNNExplainer
    """
    model.eval()
    data = data.to(device)
    F = data.x.size(1)

    # 1. Get original prediction
    with torch.no_grad():
        orig_pred = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, data=data).item()

    # 2. Prepare baseline (for masking)
    if baseline == 'zero':
        baseline_vals = torch.zeros(F, device=device)
    elif baseline == 'mean':
        baseline_vals = data.x.mean(dim=0)  # mean over atoms
    else:
        raise ValueError("baseline must be 'zero' or 'mean'")

    # 3. For each feature dimension, compute PSA delta
    deltas = torch.zeros(F, device=device)
    
    for f in range(F):
        # Create perturbed graph: set feature f to baseline for ALL atoms
        x_pert = data.x.clone()
        x_pert[:, f] = baseline_vals[f]
        
        # Keep edge_attr unchanged (or perturb edge_feat if needed)
        with torch.no_grad():
            pert_pred = model(x=x_pert, edge_index=data.edge_index, edge_attr=data.edge_attr, data=data).item()
        
        deltas[f] = orig_pred - pert_pred   # PSA score

    # 4. Combine: signed importance = sign(deltas) * node_feat_mask
    signed_importance = torch.sign(deltas) * node_feat_mask.to(device)

    return signed_importance, deltas, orig_pred


# %%
# --- 你的原始代码 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to('cpu')  # 你用 CPU，保持一致
fig_index =[384,43,235,27,328,208]

fignum=2
data = test_graph_IR_dataset[fig_index[fignum-1]].to('cpu')     #   384     208     235     27      328     43


mol = Chem.MolFromSmiles(data.smiles)
# mol = Chem.AddHs(mol)
for atom in mol.GetAtoms():
    atom.SetAtomMapNum(atom.GetIdx())
display(mol)

explainer = GNNExplainer(
    model, 
    epochs=200, 
    return_type='regression',
    feat_mask_type='feature', 
    allow_edge_mask=True, 
    log=False,
    num_hops=3
)

node_feat_mask, edge_mask, edge_feat_mask = explainer.explain_graph(
    data.x, data.edge_index, edge_attr=data.edge_attr, data=data
)
# --- 新增：PSA 方向性后处理 ---
signed_importance, deltas, orig_pred = gnne_psa_signed_importance(
    model=model,
    data=data,
    node_feat_mask=node_feat_mask,
    device='cpu',
    baseline='zero'  # or 'mean'  'zero'
)


# %%
mol.GetAtomWithIdx(10).GetIsAromatic()


# %%
x_labels = ['Atomic Number', 'Connectivity', 'Implicit Valence', # 3
            'In Ring', 'Hybridization', 'Electronegativity', 'Covalent Radius',# 7
            'Aromaticity', 'Formal Charge', 'Ring Size', # 10
            'Neighbor Electronegativity', 'Neighbor Degree', 'Neighbor Aromaticity',# 13
            'Neighbor in Ring', 'Neighbor Double Bond', 'Neighbor Triple Bond']
i = 14
signed_importance[i] = -signed_importance[i]
signed_importance[i]


# %%
# --- 新的可视化：带方向性的重要性图 ---
import matplotlib.pyplot as plt
import numpy as np

# 原始标签（确保长度与特征维度一致）
x_labels = ['Atomic Number', 'Connectivity', 'Implicit Valence', 
            'In Ring', 'Hybridization', 'Electronegativity', 'Covalent Radius',
            'Aromaticity', 'Formal Charge', 'Ring Size',
            'Neighbor Electronegativity', 'Neighbor Degree', 'Neighbor Aromaticity',
            'Neighbor in Ring', 'Neighbor Double Bond', 'Neighbor Triple Bond']

# 转为 numpy，确保长度匹配
signed_imp_np = signed_importance.cpu().numpy()
assert len(x_labels) == len(signed_imp_np), f"Label count ({len(x_labels)}) != feature dim ({len(signed_imp_np)})"

# 按绝对值排序，取 top-k
top_k = 10
indices = np.argsort(-np.abs(signed_imp_np))[:top_k]  # 从大到小
top_scores = signed_imp_np[indices]
top_labels = [x_labels[i] for i in indices]

# 反转顺序：最高 |importance| 在最上方
top_scores = top_scores[::-1]
top_labels = top_labels[::-1]

colors = ['#9AB1D6' if s < 0 else '#DD705E' for s in top_scores]

# 绘图
plt.figure(figsize=(6/2.54, 4/2.54), dpi=300)
ax = plt.gca()

bars = ax.barh(range(len(top_labels)), top_scores, color=colors, height=0.6)

# Y轴标签
ax.set_yticks(range(len(top_labels)))
ax.set_yticklabels(top_labels, fontsize=6)
ax.grid(False, color='white', linestyle='-', linewidth=1, alpha=0.7, zorder=1)

# 添加数值标签（保留符号）
for i, (bar, score) in enumerate(zip(bars, top_scores)):
    width = bar.get_width()
    # 标签位置：正数右对齐，负数左对齐（避免遮挡）
    ha = 'left' if width >= 0 else 'right'
    x_pos = width + (0.01 if width >= 0 else -0.01)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{width:+.3f}',  # 保留 +/-
            va='center', ha=ha, fontsize=6)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)  # 边框宽度
    spine.set_color('black')  # 边框颜色   
plt.grid(False)
max_abs = max(np.abs(top_scores))
plt.xlim(-max_abs * 1.35, max_abs * 1.35)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.axvline(0, color='black', linewidth=0.5)  # 零线
plt.tight_layout()
plt.subplots_adjust(left=0.01,   # 左边界
                right=0.99,   # 右边界
                bottom=0.01,  # 下边界
                top=0.99)      # 上边界
# 保存
plt.savefig(f'mol_feature_importance_{fignum}_2.pdf', bbox_inches='tight', dpi=300)
plt.show()

# 可选：打印解释
print(f"\nOriginal prediction: {orig_pred:.4f}")


# %% [markdown]
# ## 节点重要性

# %%
# --- 你的原始代码 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to('cpu')  # 你用 CPU，保持一致
data = test_graph_IR_dataset[208].to('cpu')

mol = Chem.MolFromSmiles(data.smiles)
# mol = Chem.AddHs(mol)
for atom in mol.GetAtoms():
    atom.SetAtomMapNum(atom.GetIdx())
display(mol)

explainer = GNNExplainer(
    model, 
    epochs=200, 
    return_type='regression',
    feat_mask_type='individual_feature', 
    allow_edge_mask=True, 
    log=False,
    num_hops=3
)

node_feat_mask, edge_mask, edge_feat_mask = explainer.explain_graph(
    data.x, data.edge_index, edge_attr=data.edge_attr, data=data
)
node_idx=8
target_node_mask = node_feat_mask[node_idx]
# node_feat_mask, edge_mask, edge_feat_mask = explainer.explain_node(8,data.x, data.edge_index, edge_attr=data.edge_attr,data=data)

# --- 新增：PSA 方向性后处理 ---
signed_importance, deltas, orig_pred = gnne_psa_signed_importance(
    model=model,
    data=data,
    node_feat_mask=node_feat_mask,
    device='cpu',
    baseline='zero'  # or 'mean'  'zero'
)


# %%
def node_psa_signed_importance(model, data, node_idx, node_feat_mask, device='cpu', baseline='zero'):
    """
    Compute signed importance for each feature dimension of a specific node.
    
    Args:
        model: trained GNN
         graph Data
        node_idx: int, index of target node
        node_feat_mask: [F], feature mask for this node from GNNExplainer
        baseline: 'zero' or 'mean'
    
    Returns:
        signed_importance: [F], direction-aware importance
        deltas: [F], PSA Δ values
    """
    model.eval()
    data = data.to(device)
    N, F = data.x.shape

    with torch.no_grad():
        orig_pred = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, data=data).item()

    # Baseline for masking
    if baseline == 'zero':
        baseline_val = 0.0
    elif baseline == 'mean':
        baseline_val = data.x[:, :].mean(dim=0)  # [F]
    else:
        raise ValueError("baseline must be 'zero' or 'mean'")

    deltas = torch.zeros(F, device=device)
    
    for f in range(F):
        x_pert = data.x.clone()
        if baseline == 'zero':
            x_pert[node_idx, f] = 0.0
        else:
            x_pert[node_idx, f] = baseline_val[f]
        
        with torch.no_grad():
            pert_pred = model(x=x_pert, edge_index=data.edge_index, edge_attr=data.edge_attr, data=data).item()
        deltas[f] = orig_pred - pert_pred 

    signed_importance = torch.sign(deltas) * node_feat_mask.to(device)
    return signed_importance, deltas, orig_pred

# --- Step 3: 计算带符号重要性 ---

signed_imp_node, deltas_node, orig_pred = node_psa_signed_importance(
    model=model,
    data=data,
    node_idx=node_idx,
    node_feat_mask=target_node_mask,
    device='cpu',
    baseline='zero'
)

# --- Step 4: 可视化节点 8 的特征重要性（带方向）---
import matplotlib.pyplot as plt
import numpy as np

# 特征标签（必须与你的原子特征顺序一致！）
x_labels = ['Atomic Number', 'Connectivity', 'Implicit Valence', 
            'In Ring', 'Hybridization', 'Electronegativity', 'Covalent Radius',
            'Aromaticity', 'Formal Charge', 'Ring Size',
            'Neighbor Electronegativity', 'Neighbor Degree', 'Neighbor Aromaticity',
            'Neighbor in Ring', 'Neighbor Double Bond', 'Neighbor Triple Bond']

signed_imp_np = signed_imp_node.cpu().numpy()
assert len(x_labels) == len(signed_imp_np), f"Label count mismatch!"

# 按绝对值取 top-k
top_k = 10
indices = np.argsort(-np.abs(signed_imp_np))[:top_k]
top_scores = signed_imp_np[indices]
top_labels = [x_labels[i] for i in indices]

# 反转：最大在顶部
top_scores = top_scores[::-1]
top_labels = top_labels[::-1]
colors = ['green' if s < 0 else 'red' for s in top_scores]

# 绘图
plt.figure(figsize=(9/2.54, 6/2.54), dpi=300)
ax = plt.gca()

bars = ax.barh(range(len(top_labels)), top_scores, color=colors, height=0.6)

ax.set_yticks(range(len(top_labels)))
ax.set_yticklabels(top_labels, fontsize=8)

# 数值标签（带符号）
for i, (bar, score) in enumerate(zip(bars, top_scores)):
    ha = 'left' if score >= 0 else 'right'
    x_pos = score + (0.01 if score >= 0 else -0.01)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{score:+.3f}', va='center', ha=ha, fontsize=6)

plt.xlabel('Signed Importance (Node-level)', fontsize=8)
plt.title(f'Node {node_idx} Feature Importance with Direction', fontsize=8, pad=5)
max_abs = max(np.abs(top_scores))
plt.xlim(-max_abs * 1.5, max_abs * 1.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.yticks(fontsize=6)
plt.xticks(fontsize=8)
plt.tight_layout()

plt.savefig(f'node_{node_idx}_feature_importance_signed.pdf', bbox_inches='tight', dpi=300)
plt.show()

# --- Step 5: 打印解释 ---
print(f"\nNode {node_idx} Analysis (Original prediction: {orig_pred:.4f})")
print("Top features (by |importance|):")
for score, label in zip(top_scores[::-1], top_labels[::-1]):  # 从高到低打印
    direction = "🟢 Promotes" if score < 0 else "🔴 Inhibits"
    print(f"  {label:25s}: {score:+.3f} → {direction}")


# %%


# %% [markdown]
# ## compare the model with tree

# %%
# 特征工程
def carbonyl_process_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        mol = Chem.AddHs(mol)

        # Find the carbonyl group
        carbonyl_atom = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "O" and neighbor.GetTotalNumHs() == 0:
                        # Check if the bond between C and O is a double bond
                        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                            carbonyl_atom = neighbor
                            break
                if carbonyl_atom:
                    break

        if not carbonyl_atom:
            return None, None, None, None

        # Find the carbon atom connected to the carbonyl group
        carbon_neighbors = [neighbor for neighbor in carbonyl_atom.GetNeighbors() if neighbor.GetSymbol() == "C"]
        if not carbon_neighbors:
            return "NoCarbonFound", None, None, None
        carbon = carbon_neighbors[0]


        # Find other atoms connected to the carbon
        connected_atoms = [neighbor for neighbor in carbon.GetNeighbors() if neighbor.GetIdx() != carbonyl_atom.GetIdx()]


        # Priority order for connected atoms
        priority_order = [
            ('Cl', 'SINGLE'), ('S', 'AROMATIC'), ('S', 'SINGLE'), ('F', 'SINGLE'),
            ('O', 'AROMATIC'), ('O', 'DOUBLE'), ('O', 'SINGLE'),
            ('N', 'TRIPLE'), ('N', 'AROMATIC'), ('N', 'DOUBLE'), ('N', 'SINGLE'),
            ('C', 'TRIPLE'), ('C', 'AROMATIC'), ('C', 'DOUBLE'), ('C', 'SINGLE'),
            ('H', 'SINGLE')
        ]

        # Create a list to store atom atomic number and connections
        connections_list = []
        for atom in connected_atoms:
            atomic_num = atom.GetAtomicNum()
            connected_atom_symbol = atom.GetSymbol()
            neighbors_info = []
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() != carbon.GetIdx():
                    neighbor_symbol = neighbor.GetSymbol()
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    bond_type_str = str(bond.GetBondType())
                    neighbors_info.append((neighbor_symbol, bond_type_str))

            connections_dict = {'atomic_num': atomic_num, 'connections': {connected_atom_symbol: neighbors_info}}
            connections_list.append(connections_dict)

        # Determine R1 and R2 based on atomic number and connection priority
        connections_list.sort(key=lambda x: x['atomic_num'], reverse=True)
        if len(connections_list) > 1 and connections_list[0]['atomic_num'] == connections_list[1]['atomic_num']:
#             if atomic numbers are the same, sort by connection priority
            connections_list.sort(key=lambda x: min(priority_order.index(y) if y in priority_order else len(priority_order)
                                                    for y in [item for sublist in x['connections'].values() for item in sublist]))

        R1, R2 = connections_list[0], connections_list[1] if len(connections_list) > 1 else None
        atomic_number_R1 = R1['atomic_num']
        atomic_number_R2 = R2['atomic_num'] if R2 else None

        return R1, R2, atomic_number_R1, atomic_number_R2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None
def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Descriptors.MolWt(mol)
    else:
        return 0
def calculate_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return list(fp)
def Feature_Engineering(df_unique):

    possible_columns = [
        'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
         'F_SINGLE_R1',  'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2', 'O_SINGLE_R1', 'O_SINGLE_R2',
        'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2', 'N_SINGLE_R1', 'N_SINGLE_R2',
         'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1',  'C_DOUBLE_R1',  'C_DOUBLE_R2',  'C_SINGLE_R1', 'C_SINGLE_R2',
          'H_SINGLE_R1', 'H_SINGLE_R2','P_SINGLE_R1','P_SINGLE_R2','Br_SINGLE_R1','Br_SINGLE_R2','I_SINGLE_R1','I_SINGLE_R2','P_SINGLE_R1','P_SINGLE_R2'
        ,'P_DOUBLE_R1','P_DOUBLE_R2','Si_SINGLE_R1','Si_SINGLE_R2'
    ]

    results = []

    for index, row in df_unique.iterrows():
        smiles_str = row['SMILES']
        R1, R2, atomic_number_R1, atomic_number_R2 = carbonyl_process_smiles(smiles_str)#here to process target FG
        # 初始化计数字典
#         counts = {'R1' : R1, 'R2' : R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2, 'SMILES': smiles_str, 'IR_Characteristic_Peak': row['IR_Characteristic_Peak']}
#         counts = {'R1' : R1, 'R2' : R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2, 'SMILES': smiles_str, 'IR_Characteristic_Peak': row['IR_Characteristic_Peak'], 'DOI' : row['DOI'], 'IUPAC_NAME': row['IUPAC_NAME']}
        counts = {'R1' : R1, 'R2' : R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2, 'SMILES': smiles_str, 'IR_Characteristic_Peak': row['IR_Characteristic_Peak'],'DOI' : row['DOI']}
    
        #         counts['mol_weight'] = calculate_molecular_weight(smiles_str)
        counts.update({col: 0 for col in possible_columns})

        if R1 == "NoCarbonFound":
            continue

        fingerprint = calculate_morgan_fingerprint(smiles_str)
        for i, bit in enumerate(fingerprint):
            counts[f'Fingerprint_{i}'] = bit

        for suffix, connections in [('R1', R1), ('R2', R2)]:
            if connections:
                for connection in connections['connections'].values():
                    for bond in connection:
                        bond_type_str = f'{bond[0]}_{bond[1]}'
                        counts[f'{bond_type_str}_{suffix}'] = counts.get(f'{bond_type_str}_{suffix}', 0) + 1

        results.append(counts)


    results_df = pd.DataFrame(results)
    results_df_filled = results_df.fillna(0)
    
    electronegativity_dict = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16 , 19: 0.82,26: 1.83,32:2.01,34:2.55, 35: 2.96,50: 1.96,52:2.1, 53: 2.66
    }

# Define the covalent radius dictionary
    covalent_radius_dict = {
    1: 37, 5: 82, 6: 77, 7: 75, 8: 73, 9: 71, 14: 111, 15: 106, 16: 102, 17: 99 , 19: 227,26:126 ,32:122 , 34:198 , 35: 114 ,50:140,52:140, 53: 133
    }
    
    # Add electronegativity and covalent radius columns
    results_df_filled['electronegativity_R1'] = results_df_filled['atomic_number_R1'].map(electronegativity_dict)
    results_df_filled['electronegativity_R2'] = results_df_filled['atomic_number_R2'].map(electronegativity_dict)
    results_df_filled['covalent_radius_R1'] = results_df_filled['atomic_number_R1'].map(covalent_radius_dict)
    results_df_filled['covalent_radius_R2'] = results_df_filled['atomic_number_R2'].map(covalent_radius_dict)

    # Replace NaN values if necessary (optional)
    results_df_filled['electronegativity_R1'].fillna(0, inplace=True)
    results_df_filled['electronegativity_R2'].fillna(0, inplace=True)
    results_df_filled['covalent_radius_R1'].fillna(0, inplace=True)
    results_df_filled['covalent_radius_R2'].fillna(0, inplace=True)

    print("FE Processing complete.")
    return results_df_filled

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # Predict on training and test sets
    y_train_pred = model.predict(X_train).ravel()
    y_test_pred = model.predict(X_test).ravel()

    # Compute performance metrics
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_test = r2_score(y_test, y_test_pred)

    return model, y_train_pred, rmse_train, r2_train, y_test_pred, rmse_test, r2_test

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2
def calculate_similarity(train_smiles, test_smiles):
    """计算训练集和测试集之间的Tanimoto相似度"""
    train_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in train_smiles]
    test_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in test_smiles]

    similarity_scores = []
    for test_fp in test_fps:
        max_similarity = max(AllChem.DataStructs.TanimotoSimilarity(test_fp, train_fp) for train_fp in train_fps)
        similarity_scores.append(max_similarity)
    return similarity_scores

def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise


# %%
import os
import glob
import joblib
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import DataLoader
model=model.cuda()
def compare_models(model, model_dir, scaler, dataset, mean, std):
    # 获取所有以8072_experiment2开头的模型文件
    model_files = glob.glob(os.path.join(model_dir, '8072_experiment2_*.joblib'))
    
    # 初始化DataFrame
    df_unique = pd.DataFrame({
        'DOI': [getattr(d, 'doi', None) for d in dataset],
        'SMILES': [getattr(d, 'smiles', None) for d in dataset],
        'IR_Characteristic_Peak': [getattr(d, 'y', None) for d in dataset],
        'Carbonyl_Atom_Feature_11': [
            d.x[d.carbonyl_mask == 0.6][0][9].item() if hasattr(d, 'carbonyl_mask') and hasattr(d, 'x') and d.carbonyl_mask is not None and d.x is not None else None 
            for d in dataset
        ]
    })
    df = Feature_Engineering(df_unique)

    feature = ['electronegativity_R1', 'electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2',
              'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
            'F_SINGLE_R1',  'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2', 'O_SINGLE_R1', 'O_SINGLE_R2',
            'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2', 'N_SINGLE_R1', 'N_SINGLE_R2',
            'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1',  'C_DOUBLE_R1',  'C_DOUBLE_R2',  'C_SINGLE_R1', 'C_SINGLE_R2',
              'H_SINGLE_R1', 'H_SINGLE_R2','Br_SINGLE_R1','Br_SINGLE_R2','I_SINGLE_R1','I_SINGLE_R2','P_SINGLE_R1','P_SINGLE_R2'
            ,'P_DOUBLE_R1','P_DOUBLE_R2','Si_SINGLE_R1','Si_SINGLE_R2'
              ]
    Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]
    all_features = feature + Morgan_features
    X = df[all_features].values
    
    # 缩放特征
    X_all_scaled = scaler.transform(X)
    
    # GNN模型预测
    model.eval()
    y_true = []
    y_pred = []
    datasee_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
    with torch.no_grad():
        for _, batch in enumerate(datasee_loader):
            batch_atom_bond = batch
            batch_atom_bond = batch_atom_bond.to(device)
            pred = model(batch_atom_bond.x, batch_atom_bond.edge_index, batch_atom_bond.edge_attr, batch_atom_bond)
            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
            y_pred.append(pred[:].detach().cpu())
    y_true = torch.cat(y_true, dim=0) * std + mean
    y_pred = torch.cat(y_pred, dim=0) * std + mean

    df_unique['IR_Characteristic_Peak']= y_true.numpy()
    df_unique['GNN_pred'] = y_pred.numpy()
    df_unique['GNN_difference'] = np.abs(df_unique['IR_Characteristic_Peak'] - y_pred.numpy())

    # 加载并运行所有树模型
    for model_file in model_files:
        # 加载模型
        tree_model = joblib.load(model_file)
        
        # 获取模型名称（不带路径和扩展名）
        model_name = os.path.splitext(os.path.basename(model_file))[0].split('8072_experiment2_')[-1]
        
        # 预测
        y_pred = tree_model.predict(X_all_scaled)
        
        # 添加到DataFrame
        df_unique[f'{model_name}_pred'] = y_pred
        df_unique[f'{model_name}_difference'] = np.abs(df_unique['IR_Characteristic_Peak'] - y_pred)
    
    # 计算所有模型与GNN的差异
    for model_file in model_files:
        model_name = os.path.splitext(os.path.basename(model_file))[0].split('8072_experiment2_')[-1]
        df_unique[f'{model_name}_vs_GNN'] = df_unique[f'{model_name}_difference'] - df_unique['GNN_difference']
    
    # 保存结果
    data1 = df_unique.copy()
    data1.to_csv(f'test_data_compare_test_{total_num}.csv', index=False)

model_dir = './models/'
scaler = joblib.load('./scaler/8072_experiment2_scaler.joblib')
compare_models(model, model_dir, scaler, test_graph_IR_dataset,graph_IR_dataset.mean ,graph_IR_dataset.std)


# %% [markdown]
# ## with tree model

# %%
from .modules import *


# %%
df = Feature_Engineering(df_unique,graph_IR_dataset, model)
# df.to_csv('output3.csv')
failed_rows=[]
for index, row in df.iterrows():
    if row['R1']==0 or row['IR_Characteristic_Peak'] == 0:
        # 如果第一列的元素为0，记录下该行的索引
        failed_rows.append(index)
df.iloc[failed_rows].to_excel('D:\\1 ir amide\\failed_rows.xlsx', index=False)
df = df[~df.index.isin(failed_rows)]
# 重置索引
df.reset_index(drop=True, inplace=True)
print(len(df))
df.to_csv('all_feature.csv')
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
y = df['IR_Characteristic_Peak'].values

feature = ['electronegativity_R1', 'electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2',
           'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
         'F_SINGLE_R1',  'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2', 'O_SINGLE_R1', 'O_SINGLE_R2',
        'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2', 'N_SINGLE_R1', 'N_SINGLE_R2',
         'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1',  'C_DOUBLE_R1',  'C_DOUBLE_R2',  'C_SINGLE_R1', 'C_SINGLE_R2',
          'H_SINGLE_R1', 'H_SINGLE_R2','Br_SINGLE_R1','Br_SINGLE_R2','I_SINGLE_R1','I_SINGLE_R2','P_SINGLE_R1','P_SINGLE_R2'
        ,'P_DOUBLE_R1','P_DOUBLE_R2','Si_SINGLE_R1','Si_SINGLE_R2'
           ]
Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]
gin_features = [f'gin_{i}' for i in range(520)]
all_features = feature + Morgan_features
X = df[all_features].values


# %%
gin_features = [f'gin_{i}' for i in range(520)]
all_features = feature + gin_features
X = df[all_features].values
y = df['IR_Characteristic_Peak'].values/1000


# %%
# X_train_val, X_test = X , X
# y_train_val, y_test = y, y

X_train_val, X_test, y_train_val, y_test, indices_train_val, indices_test = train_test_split(
    X, y, df.index , test_size=0.25, random_state=42)  # df 要改
# import csv
# data = np.vstack((X_train_val, X_test))
# filename = 'output.csv'
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for row in data:
#         writer.writerow(row)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)

data_source = 'experiment'  #'experiment','computed'
number=len(df) # df , df_small


# %%
base_models = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42)),
    ('lgbm', lgb.LGBMRegressor(random_state=42)),
    ('CatBoost', CatBoostRegressor(random_state=42))
]

# Define stacked regressor
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=BayesianRidge()
)

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    'Bayesian Ridge Regression': BayesianRidge(),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": lgb.LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(random_state=42),
    "stacked_model": stacked_model
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    
    print(f'\nTraining {model_name} with 5-Fold Cross Validation...')
    rmse_scores = []
    r2_scores = []
    for train_index, val_index in kf.split(X_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        r2 = r2_score(y_val, y_val_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    print(f'{model_name} - Mean RMSE: {np.mean(rmse_scores)}, Mean R2: {np.mean(r2_scores)}')

    model.fit(X_train_val, y_train_val)
    y_test_pred = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_test = r2_score(y_test, y_test_pred)
    print(f'{model_name} - Test RMSE: {rmse_test}, Test R2: {r2_test}')

    pic=plot_prediction_scatter(y_test, y_test_pred, f'{number}_{data_source}_{model_name}', figsize=(5, 5), alpha=0.2)
    plt.savefig(f'./figures/{number}_{data_source}_{model_name}.png', dpi=600)  # df

    print(f'{model_name} Model saving')
    dump(model, f'./models/{number}_{data_source}_{model_name}.joblib')
    dump(scaler, f'./scaler/{number}_{data_source}_scaler.joblib')


# %%


# %%



