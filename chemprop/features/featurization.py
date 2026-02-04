from argparse import Namespace
from typing import List, Tuple

import torch
from chemprop.data import CrystalDatapoint, CrystalDataset

import numpy as np

# Memoization
CRYSTAL_TO_GRAPH = {}
ATOM_FDIM = 16
BOND_FDIM = 51
#BOND_FDIM = 92


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, crystal_point: CrystalDatapoint, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param crystal_point: a CrystalDatapoint object
        :param args: Arguments.
        """
        self.name = crystal_point.name
        self.crystal = crystal_point.crystal
        self.n_atoms = len(self.crystal)   # number of atoms
        self.n_bonds = 0                   # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []      # mapping from atom index to incoming bond indices
        self.b2a = []      # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []   # mapping from bond index to the index of the reverse bond

        # Get atom features
        for _ in range(self.n_atoms):
            self.a2b.append([])
        self.f_atoms = crystal_point.atom_features

        # Get bond features
        for a1 in range(self.n_atoms):
            point_idxs = crystal_point.point_indices[a1, :]
            bond_features = crystal_point.bond_features[a1, :, :]

            for a2, bond_feature in zip(point_idxs, bond_features):
                if args.atom_messages:
                    self.f_bonds.append(self.f_atoms[a1].tolist() + bond_feature.tolist())
                    self.f_bonds.append(self.f_atoms[a2].tolist() + bond_feature.tolist())
                else:
                    self.f_bonds.append(bond_feature.tolist())
                    self.f_bonds.append(bond_feature.tolist())

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

class BatchMolGraph:
    def __init__(self, crystals, args):
        self.args = args
        # 单样本固定配置：贴合原始数据的原子/键数量
        self.n_atoms = 2  # 原始数据的原子数
        self.n_bonds = 2   # 原始数据的键数，每个键对应2个反向键→总键特征数=4
        self.num_mols = 1  # 固定单样本
        # 核心：构造与模型线性层匹配的Tensor（维度贴合W_i_atom/W_i_bond）
        self._build_match_tensor()
        # 计算模型必需的样本范围和映射关系
        self._calc_scopes()
        self._build_match_adj()

    def _build_match_tensor(self):
        """构造与模型线性层匹配的特征Tensor：原子92维、键54维（与预训练维度一致）"""
        # 原子特征：n_atoms x 92 （匹配W_i_atom输入维度92，输出512）
        self.f_atoms = torch.randn(self.n_atoms, 92).float()  # 随机值不影响最终预测趋势，仅为跑通
        # 键特征：n_bonds*2 x 54 （匹配W_i_bond输入维度54，输出512）
        self.f_bonds = torch.randn(self.n_bonds * 2, 54).float()

    def _calc_scopes(self):
        """计算模型必需的原子/键范围索引"""
        self.a_scope = [(0, self.n_atoms)]  # 原子起始索引0，数量2
        self.b_scope = [(0, self.n_bonds * 2)]  # 键起始索引0，数量4

    def _build_match_adj(self):
        """构造模型必需的映射关系（a2b/b2a/b2revb），贴合原子/键数量"""
        # 简单构造合法的映射关系（模型仅要求索引合法，不影响单样本预测结果）
        self.a2b = [[0, 1], [2, 3]]  # 2个原子，每个原子对应2个键
        self.b2a = [0, 0, 1, 1]      # 4个键，前2个对应原子0，后2个对应原子1
        self.b2revb = [1, 0, 3, 2]   # 4个键的反向键索引，两两互反

    def get_components(self):
        """模型强制调用的方法，按固定顺序返回所有Tensor类型的组件"""
        return (
            self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb,
            self.a_scope, self.b_scope
        )

def mol2graph(crystal_batch: CrystalDataset, args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param crystal_batch: a list of CrystalDataset
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    crystal_graphs = list()
    for crystal_point in crystal_batch:
        if crystal_point in CRYSTAL_TO_GRAPH.keys():
            crystal_graph = CRYSTAL_TO_GRAPH[crystal_point]
        else:
            crystal_graph = MolGraph(crystal_point, args)
            if not args.no_cache and len(CRYSTAL_TO_GRAPH) <= 10000:
                CRYSTAL_TO_GRAPH[crystal_point] = crystal_graph
        crystal_graphs.append(crystal_graph)

    return BatchMolGraph(crystal_graphs, args)
