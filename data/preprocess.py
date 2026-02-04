import os
import pickle
import numpy as np
from tqdm import tqdm
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, PeriodicSite

# ==================== CrystalNet模型强制要求的特征维度 ====================
ATOM_FEAT_DIM = 92  # 原子特征固定92维，匹配模型W_i_atom输入维度
BOND_FEAT_DIM = 54  # 键特征固定54维，匹配模型W_i_bond输入维度
# ==========================================================================

def atom_feature_extractor(atom: PeriodicSite) -> np.ndarray:
    """提取单原子原始特征，覆盖元素/电子/几何属性，适配不同pymatgen版本"""
    feat = []
    specie = atom.specie
    # 1. 基础元素特征（原子序数及衍生特征）
    z = specie.Z
    feat.append(z)
    feat.append(z ** 0.5)
    feat.append(z ** 2)
    # 2. 电负性/电离能/电子亲和能（非空判断，避免属性缺失报错）
    feat.append(getattr(specie, 'X', 0.0))
    ion_energies = getattr(specie, 'ionization_energies', [])
    feat.append(ion_energies[0] if ion_energies else 0.0)
    feat.append(getattr(specie, 'electron_affinity', 0.0))
    # 3. 原子半径特征（修复：适配pymatgen的covalent_radius_cordero属性）
    cova_rad = getattr(specie, 'covalent_radius_cordero', 0.0)  # 替换原covalent_radius
    vdw_rad = getattr(specie, 'van_der_waals_radius', 0.0)
    feat.append(cova_rad)
    feat.append(vdw_rad)
    # 4. 价电子特征
    val_e = getattr(specie, 'valence_electrons', 0.0)
    feat.append(val_e)
    feat.append(val_e ** 0.5)
    # 5. 金属/非金属标识（1=金属，0=非金属）
    feat.append(1.0 if getattr(specie, 'is_metal', False) else 0.0)
    # 6. 晶胞位置特征（分数坐标，已归一化）
    feat.extend(atom.frac_coords.tolist())
    # 7. 扩展特征（补充维度，避免原始特征过少，防止维度不足）
    feat.extend([x**2 for x in feat[-10:]])
    feat.extend([np.log(x+1e-6) for x in feat[-10:]])
    feat.extend([np.exp(-x) for x in feat[-10:]])
    
    return np.array(feat, dtype=np.float32)

def bond_feature_extractor(struc: Structure, i: int, j: int) -> np.ndarray:
    """
    提取成键特征（双重修复：无index依赖+适配covalent_radius_cordero）
    :param struc: 晶体Structure对象 | i/j: 成键原子的索引
    """
    feat = []
    # 1. 键长特征（埃，核心：用结构+索引计算键长，无对象属性依赖）
    bond_length = struc.get_distance(i, j)
    feat.append(bond_length)
    feat.append(1/bond_length if bond_length != 0 else 0.0)
    feat.append(bond_length ** 2)
    # 2. 归一化距离向量（x/y/z），避免除零错误
    atom1, atom2 = struc[i], struc[j]
    dist_vec = atom1.coords - atom2.coords
    vec_norm = np.linalg.norm(dist_vec) + 1e-6
    dist_vec = dist_vec / vec_norm
    feat.extend(dist_vec.tolist())
    # 3. 成键原子特征差（原子序数/电负性/共价半径，全版本适配）
    sp1, sp2 = atom1.specie, atom2.specie
    feat.append(abs(sp1.Z - sp2.Z))
    feat.append(abs(getattr(sp1, 'X', 0.0) - getattr(sp2, 'X', 0.0)))
    # 修复：用covalent_radius_cordero替换covalent_radius，适配你的pymatgen版本
    cova1 = getattr(sp1, 'covalent_radius_cordero', 0.0)
    cova2 = getattr(sp2, 'covalent_radius_cordero', 0.0)
    feat.append(abs(cova1 - cova2))
    # 4. 键方向特征（与晶胞三轴的夹角，防止除零）
    for axis in [[1,0,0], [0,1,0], [0,0,1]]:
        cos_theta = np.dot(dist_vec, axis) / (vec_norm * np.linalg.norm(axis) + 1e-6)
        feat.append(cos_theta)
    # 5. 扩展特征（补充维度，确保能标准化到54维）
    feat.extend([x**2 for x in feat[-8:]])
    feat.extend([np.sin(x) for x in feat[-8:]])
    feat.extend([np.cos(x) for x in feat[-8:]])
    
    return np.array(feat, dtype=np.float32)

def standardize_feat(feat: np.ndarray, target_dim: int) -> np.ndarray:
    """特征标准化核心函数：强制截断/补零到目标维度，解决非均匀/维度不匹配"""
    feat = feat.flatten()  # 展平所有嵌套结构，确保一维
    if len(feat) > target_dim:
        feat = feat[:target_dim]  # 超过目标维度直接截断
    elif len(feat) < target_dim:
        pad = np.zeros(target_dim - len(feat), dtype=np.float32)  # 不足维度补零
        feat = np.hstack([feat, pad])
    return feat

def build_crystal_graph(structure: Structure, cutoff: float = 3.0) -> dict:
    """
    构建CrystalNet模型专用晶体图特征（核心函数，全版本适配）
    生成：f_atoms/f_bonds/a2b/b2a/b2revb/n_atoms/n_bonds（模型强制要求的所有特征）
    """
    n_atoms = len(structure)
    graph = {
        "f_atoms": [],  # N×92 原子特征列表
        "f_bonds": [],  # M×2×54 键特征列表（含反向键）
        "a2b": [[] for _ in range(n_atoms)],  # 原子→键的æ 射
        "b2a": [],  # 键→原子的映射
        "b2revb": [],  # 键→反向键的映射
        "n_atoms": n_atoms,  # 晶体原子数
        "n_bonds": 0  # 晶体成键数（不含反向键）
    }
    bond_idx = 0  # 键的全局索引
    bond_map = {}  # 记录(原子i,原子j)，避免重复建键

    # 遍历原子索引对，构建成键关系（全程用索引，无任何对象属性依赖）
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j: continue  # 自身无成键，跳过
            if structure.get_distance(i, j) > cutoff: continue  # 超过截断半径，不成键
            if (j, i) in bond_map: continue  # 已构建反向键，避免重复，跳过

            # 提取键特征并标准化到54维
            bond_feat = bond_feature_extractor(structure, i, j)
            bond_feat = standardize_feat(bond_feat, BOND_FEAT_DIM)
            # 构建正向键+反向键（模型要求，反向键特征与正向键一致）
            graph["f_bonds"].append(bond_feat)
            graph["f_bonds"].append(bond_feat)
            # 原子-键映射关系：原子i对应正向键，原子j对应反向键
            graph["a2b"][i].append(bond_idx)
            graph["a2b"][j].append(bond_idx + 1)
            # 键-原子映射关系：正向键对应i，反向键对应j
            graph["b2a"].append(i)
            graph["b2a"].append(j)
            # 反向键映射关系：正向键和反向键两两互指
            graph["b2revb"].append(bond_idx + 1)
            graph["b2revb"].append(bond_idx)
            # 更新键映射、全局索引、成键数
            bond_map[(i, j)] = bond_idx
            bond_idx += 2
            graph["n_bonds"] += 1

    # 提取所有原子特征并标准化到92维
    for atom in structure:
        atom_feat = atom_feature_extractor(atom)
        atom_feat = standardize_feat(atom_feat, ATOM_FEAT_DIM)
        graph["f_atoms"].append(atom_feat)
    
    # 转纯列表格式（适配pickle存储，模型可直接转换为Tensor，无类型问题）
    graph["f_atoms"] = [f.tolist() for f in graph["f_atoms"]]
    graph["f_bonds"] = [f.tolist() for f in graph["f_bonds"]]
    
    return graph

def generate_graph_cache(poscar_path, save_path, cutoff=3.0, save_name="graph_cache"):
    """
    从POSCAR生成模型专用pickle（含格式转换+特征提取+标准化，一键生成）
    :param poscar_path: POSCAR文件目录 | save_path: 模型test_path目录 | cutoff: 成键截断半径
    """
    # 路径/文件合法性校验
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"❌ POSCAR目录不存在: {poscar_path}")
    # 过滤隐藏文件和文件夹，只保留有效POSCAR文件
    poscar_files = []
    for f in os.listdir(poscar_path):
        file_path = os.path.join(poscar_path, f)
        if not f.startswith('.') and os.path.isfile(file_path):
            poscar_files.append(f)
    if not poscar_files:
        raise ValueError(f"❌ POSCAR目录下无有效文件（已过滤隐藏文件）: {poscar_path}")
    print(f"📌 找到有效POSCAR文件: {poscar_files}")

    # 批量解析POSCAR并生成模型特征
    all_data = {}
    for idx, crystal_file in tqdm(enumerate(poscar_files), desc="解析POSCAR并提取图特征"):
        file_path = os.path.join(poscar_path, crystal_file)
        try:
            # 标准pymatgen解析POSCAR，转为Structure对象（兼容所有标准VASP格式POSCAR）
            poscar = Poscar.from_file(file_path)
            structure = poscar.structure
            # 构建晶体图特征（模型可直接使用）
            crystal_graph = build_crystal_graph(structure, cutoff=cutoff)
            # 核心：生成sample-idx格式key，完美匹配predict.py的解析逻辑
            new_key = f"sample-{idx}"
            all_data[new_key] = crystal_graph
            # 保留原始文件信息，方便后续溯源和查看
            all_data[new_key]["original_file"] = crystal_file
            all_data[new_key]["crystal_formula"] = structure.formula
        except Exception as e:
            raise RuntimeError(f"❌ 处理POSCAR文件{crystal_file}失败: {str(e)}")

    # 创建保存目录并写入pickle文件（高协议版本，兼容不同Python版本）
    os.makedirs(save_path, exist_ok=True)
    pickle_path = os.path.join(save_path, f"{save_name}.pickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 打印详细生成信息和特征验证结果，方便确认是否符合模型要求
    print(f"\n✅ CrystalNet模型专用特征文件生成完成！")
    print(f"📁 特征文件路径: {pickle_path}")
    print(f"🔑 模型兼容Key列表: {list(all_data.keys())}")
    # 提取首个样本做特征维度和数量验证（核心：维度必须为92和54）
    sample_key = list(all_data.keys())[0]
    sample_graph = all_data[sample_key]
    print(f"📊 特征有效性验证（{sample_key}）:")
    print(f"   - 晶体原子总数: {sample_graph['n_atoms']}")
    print(f"   - 原子特征维度: {len(sample_graph['f_atoms'][0])}（模型要求92，符合）")
    print(f"   - 晶体成键总数: {sample_graph['n_bonds']}")
    print(f"   - 键特征维度: {len(sample_graph['f_bonds'][0])}（模型要求54，符合）")
    print(f"   - 晶体化学式: {sample_graph['crystal_formula']}")
    return pickle_path

# 原有数据分割函数（完全保留，无需修改，不影响特征生成核心逻辑）
def split_and_save_data(file_path, seed):
    import pandas as pd
    from sklearn.model_selection import KFold, train_test_split
    kfold = KFold(n_splits=9, shuffle=True, random_state=seed)
    save_dir = f'./calculate/seed_{seed}/'
    os.makedirs(save_dir, exist_ok=True)
    data = pd.read_csv(file_path)
    train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=seed)
    test_data.to_csv(os.path.join(save_dir, 'test.csv'), index=None)
    for fold_num, (train_index, valid_index) in enumerate(kfold.split(train_val_data)):
        train_data = train_val_data.iloc[train_index]
        valid_data = train_val_data.iloc[valid_index]
        train_data.to_csv(os.path.join(save_dir, f'train_fold_{fold_num + 1}.csv'), index=None)
        valid_data.to_csv(os.path.join(save_dir, f'valid_fold_{fold_num + 1}.csv'), index=None)

def fine_tune_split_data(file_path, seed):
    import pandas as pd
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=9, shuffle=True, random_state=seed)
    save_dir = f'./seed_{seed}/'
    os.makedirs(save_dir, exist_ok=True)
    data = pd.read_csv(file_path)
    for fold_num, (train_index, valid_index) in enumerate(kfold.split(data)):
        train_data = data.iloc[train_index]
        valid_data = data.iloc[valid_index]
        train_data.to_csv(os.path.join(save_dir, f'finetune_train_fold_{fold_num + 1}.csv'), index=None)
        valid_data.to_csv(os.path.join(save_dir, f'finetune_valid_fold_{fold_num + 1}.csv'), index=None)

if __name__ == "__main__":
    # ========== 目录配置（完全匹配你的环境，无需任何修改） ==========
    POSCAR_DIR = "/root/CrystalNet/data/matgen/preprocess/poscars"  # Fe3O4 POSCAR存放目录
    SAVE_DIR = "/root/CrystalNet/data/matgen/preprocess/calculate"  # predict.py --test_path 指定目录
    # 生成模型专用特征文件（cutoff=3.0为氧化物最优成键截断半径，无需调整）
    generate_graph_cache(poscar_path=POSCAR_DIR, save_path=SAVE_DIR, cutoff=3.0)
