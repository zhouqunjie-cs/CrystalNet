import os
import pickle
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from pymatgen.io.vasp import Poscar
from tqdm import tqdm

def generate_graph_cache(poscar_path, save_path, save_name="graph_cache"):
    """
    从POSCAR文件生成格式兼容的graph_cache.pickle
    关键修改：将晶体名转为sample-数字格式，匹配predict.py解析逻辑
    :param poscar_path: POSCAR文件所在文件夹（如./data/matgen/preprocess/poscars）
    :param save_path: pickle文件保存路径（predict.py的test_path：./data/matgen/preprocess/calculate）
    :param save_name: 输出pickle文件名，默认graph_cache
    """
    # 1. 校验输入路径和文件
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"❌ POSCAR文件夹不存在: {poscar_path}")
    # 获取所有POSCAR文件（此处是Fe3O4）
    poscar_files = [f for f in os.listdir(poscar_path) if os.path.isfile(os.path.join(poscar_path, f))]
    if not poscar_files:
        raise ValueError(f"❌ POSCAR文件夹下无有效文件: {poscar_path}")
    print(f"📌 找到POSCAR文件: {poscar_files}")

    # 2. 解析POSCAR并转换key格式为sample-数字
    all_data = {}
    for idx, crystal_name in tqdm(enumerate(poscar_files), desc="解析POSCAR并转换格式"):
        file_path = os.path.join(poscar_path, crystal_name)
        try:
            # 读取Fe3O4的POSCAR文件，转为pymatgen对象
            poscar_obj = Poscar.from_file(file_path)
            # 核心修改：key从Fe3O4 → sample-0（数字从0开始，匹配predict.py排序）
            new_key = f"sample-{idx}"
            all_data[new_key] = poscar_obj.as_dict()
            # 可选：保留原始晶体名到value中，方便后续查看
            all_data[new_key]["original_name"] = crystal_name
        except Exception as e:
            raise RuntimeError(f"❌ 解析POSCAR文件{crystal_name}失败: {str(e)}")

    # 3. 保存为pickle文件（二进制，不可读是正常的）
    os.makedirs(save_path, exist_ok=True)
    pickle_path = os.path.join(save_path, f"{save_name}.pickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 4. 打印生成信息，方便验证
    print(f"✅ 格式兼容的pickle文件生成完成！")
    print(f"📁 文件路径: {pickle_path}")
    print(f"🔑 生成的key（匹配predict.py）: {list(all_data.keys())}")  # 输出['sample-0']
    print(f"📦 包含原始晶体: {[v['original_name'] for v in all_data.values()]}")  # 输出['Fe3O4']
    return pickle_path

# 以下是原有分割数据的函数，无需修改（若用不到可保留）
def split_and_save_data(file_path, seed):
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
    # ========== 核心执行代码：按你的目录结构配置 ==========
    # POSCAR文件所在目录（你的Fe3O4文件位置）
    POSCAR_DIR = "/root/CrystalNet/data/matgen/preprocess/poscars"
    # pickle保存目录（predict.py的--test_path，必须一致！）
    SAVE_DIR = "/root/CrystalNet/data/matgen/preprocess/calculate"
    # 生成格式兼容的graph_cache.pickle
    generate_graph_cache(poscar_path=POSCAR_DIR, save_path=SAVE_DIR)
