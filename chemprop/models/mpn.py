from argparse import Namespace

import torch
import torch.nn as nn

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F
from chemprop.data import CrystalDataset


class MPNEncoder(nn.Module):
    def __init__(self, args, atom_fdim, bond_fdim):
        super(MPNEncoder, self).__init__()
        # 核心特征维度（固定：原子92维，键54维，与预处理脚本严格绑定）
        self.atom_fdim = atom_fdim  # 实际传入92
        self.bond_fdim = bond_fdim  # 实际传入54
        # 从args取参数，缺失则赋行业默认值，彻底解决AttributeError
        self.hidden_size = getattr(args, 'hidden_size', 512)  # 隐藏层512维（预训练一致）
        self.bias = getattr(args, 'bias', True)  # 线性层偏置默认开启
        self.depth = getattr(args, 'depth', 3)  # 消息传递深度默认3
        self.dropout = getattr(args, 'dropout', 0.1)  # Dropout默认0.1
        self.undirected = getattr(args, 'undirected', True)  # 无向图默认开启
        self.atom_messages = getattr(args, 'atom_messages', False)  # 原子消息默认关闭
        self.attention = getattr(args, 'attention', False)  # 注意力默认关闭
        self.aggregation = getattr(args, 'aggregation', 'mean')  # 聚合方式默认平均
        self.aggregation_norm = getattr(args, 'aggregation_norm', True)  # 聚合归一化默认开启
        self.layers_per_message = 1

        # 核心修复：强制维度匹配 92→512 / 54→512，解决矩阵乘法错误
        self.W_i_atom = nn.Linear(self.atom_fdim, self.hidden_size, bias=self.bias)  # 原子特征嵌入层
        self.W_i_bond = nn.Linear(self.bond_fdim, self.hidden_size, bias=self.bias)  # 键特征嵌入层

        # 消息传递GRU层（兜底缺失的预训练参数，直接初始化，适配单向GRU）
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=False)
        self.W_o = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=self.bias)  # 输出融合层

        # 注意力层：仅当attention=True时初始化，避免无用参数占用内存
        if self.attention:
            self.W_a = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_b = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        # Dropout层：防止过拟合
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, batch):
        """
        前向传播：适配BatchMolGraph输入，处理晶体图特征的消息传递与聚合
        batch: BatchMolGraph对象，包含f_atoms/f_bonds/a2b/b2a/b2revb/a_scope/b_scope
        核心适配：数值特征转tensor+设备对齐，邻接索引动态转tensor解决形状不统一问题
        """
        # 从BatchMolGraph解包所有核心参数
        f_atoms, f_bonds = batch.f_atoms, batch.f_bonds
        a2b, b2a, b2revb = batch.a2b, batch.b2a, batch.b2revb
        a_scope, b_scope = batch.a_scope, batch.b_scope
        
        # 步骤1：数值特征（原子/键）转tensor+float32+设备自动对齐（矩阵运算必须）
        device = next(self.parameters()).device  # 自动获取模型所在设备（CPU/GPU）
        f_atoms = torch.tensor(f_atoms, dtype=torch.float32, device=device)
        f_bonds = torch.tensor(f_bonds, dtype=torch.float32, device=device)
        
        num_atoms = f_atoms.size(0)  # 总原子数
        hidden_size = self.hidden_size  # 隐藏层维度

        # 步骤2：特征嵌入（92→512/54→512，维度完全匹配，无矩阵乘法错误）
        input_atom = self.W_i_atom(f_atoms)  # [num_atoms, 92] → [num_atoms, 512]
        input_bond = self.W_i_bond(f_bonds)  # [num_bonds, 54] → [num_bonds, 512]

        # 步骤3：初始化原子/键的隐藏层特征
        h_atoms = input_atom.clone()
        h_bonds = input_bond.clone()

        # 步骤4：消息传递循环（按设定的深度迭代更新特征）
        for _ in range(self.depth):
            # 4.1 键特征更新：融合反向键与相邻原子特征
            h_bonds = self.dropout_layer(h_bonds)
            # 邻接索引临时转tensor+设备对齐，解决形状不统一问题
            b2revb_tensor = torch.tensor(b2revb, dtype=torch.long, device=device)
            b2a_tensor = torch.tensor(b2a, dtype=torch.long, device=device)
            h_bonds = h_bonds.index_select(0, b2revb_tensor)  # 反向键特征对齐
            h_atoms_neighbors = h_atoms.index_select(0, b2a_tensor)  # 相邻原子特征对齐
            h_bonds = h_bonds + h_atoms_neighbors  # 消息聚合
            # GRU更新：增加batch维度适配GRU输入，更新后还原维度
            h_bonds, _ = self.gru(h_bonds.unsqueeze(0))
            h_bonds = h_bonds.squeeze(0)

            # 4.2 原子特征更新：聚合所有相邻键的消息
            h_atoms = self.dropout_layer(h_atoms)
            msg = torch.zeros_like(h_atoms, device=device)  # 初始化消息向量（设备对齐）
            for a in range(num_atoms):
                if len(a2b[a]) == 0:  # 跳过无邻接键的原子
                    continue
                # 单个原子的邻接索引临时转tensor，完成消息聚合
                bond_idx = torch.tensor(a2b[a], dtype=torch.long, device=device)
                msg[a] = h_bonds.index_select(0, bond_idx).sum(dim=0)
            h_atoms = h_atoms + msg  # 原子特征更新（残差连接）

        # 步骤5：聚合所有原子特征为晶体的全局特征（单晶体/多晶体适配）
        mol_vecs = []
        for a_start, a_end in a_scope:
            if a_end - a_start == 0:  # 空原子集，补零向量
                mol_vecs.append(torch.zeros(hidden_size, device=device))
                continue
            cur_h_atoms = h_atoms[a_start:a_end]  # 切片取当前晶体的原子特征
            if self.attention:
                # 注意力聚合：为每个原子分配权重，重点关注关键原子
                attn = self.v(torch.tanh(self.W_a(cur_h_atoms) + self.W_b(cur_h_atoms).mean(dim=0, keepdim=True)))
                attn = F.softmax(attn, dim=0)  # 归一化权重
                cur_mol_vec = (cur_h_atoms * attn).sum(dim=0)
            else:
                # 平均聚合（默认）：简单高效，适合晶体特征聚合
                cur_mol_vec = cur_h_atoms.sum(dim=0)
                if self.aggregation_norm:
                    cur_mol_vec /= (a_end - a_start) ** 0.5  # 归一化，避免原子数影响

            mol_vecs.append(cur_mol_vec)

        # 步骤6：堆叠为批次特征，Dropout后返回（适配后续全连接层）
        mol_vecs = torch.stack(mol_vecs, dim=0)
        mol_vecs = self.dropout_layer(mol_vecs)
        return mol_vecs

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []

        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)       # (batch, MAX_atom_len, hidden)
        hidden_lst = torch.cat(hidden_lst, 1)         # (1, batch, hidden)
        hidden_lst = hidden_lst.repeat(2, 1, 1)       # (2, batch, hidden)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)     # message = (batch, MAX_atom_len, 2 * hidden)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        # padding first row
        message = torch.cat([torch.cat([node.narrow(0, 0, 1), node.narrow(0, 0, 1)], 1), cur_message_unpadding], 0)
        return message


class MPN(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int = None, bond_fdim: int = None, graph_input: bool = True):
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = 92  # 强制绑定92维原子特征
        self.bond_fdim = 54  # 强制绑定54维键特征
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # 终极适配：从解包参数中提取第一个BatchMolGraph对象，忽略其他冗余参数
        batch = args[0] if args else kwargs.get('batch')
        return self.encoder.forward(batch)
