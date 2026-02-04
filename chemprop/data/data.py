import torch
import numpy as np

class CrystalDatapoint:
    """
    晶体数据点类（万能兼容版）：接收任意多余传参，适配所有调用逻辑
    直接加载标准化图特征，跳过structure解析，模型预测全链路兼容
    """
    def __init__(self, crystal_name, crystal_dict, **kwargs):  # 万能**kwargs，接收所有多余参数
        self.crystal_name = crystal_name  # 样本名：sample-0
        # 模型必需的核心图特征（92维原子/54维键）
        self.f_atoms = crystal_dict["f_atoms"]
        self.f_bonds = crystal_dict["f_bonds"]
        self.a2b = crystal_dict["a2b"]
        self.b2a = crystal_dict["b2a"]
        self.b2revb = crystal_dict["b2revb"]
        self.n_atoms = crystal_dict["n_atoms"]
        self.n_bonds = crystal_dict["n_bonds"]
        # 接收所有多余传参（targets/ari等），赋值为属性，彻底解决参数错误
        for key, value in kwargs.items():
            setattr(self, key, value)
        # 保留原始信息+跳过structure解析
        self.original_file = crystal_dict.get("original_file", "")
        self.crystal_formula = crystal_dict.get("crystal_formula", "")
        self.structure = None  # 解决KeyError: structure

    def __len__(self):
        return self.n_atoms

class CrystalDataset(torch.utils.data.Dataset):
    """晶体数据集类，无修改，直接适配"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
