import pandas as pd
from ase.io import write
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
import json
import numpy as np
from matminer.datasets import load_dataset
import time 
import os

# 从Matbench数据集中加载数据
# dataset_names = ["matbench_jdft2d","matbench_phonons",
# "matbench_dielectric",
# "matbench_log_gvrh",
# # "matbench_log_kvrh",
# "matbench_perovskites",
# "matbench_mp_gap",
# "matbench_mp_e_form",
# "matbench_mp_is_metal"]
dataset_names = ["matbench_log_kvrh","matbench_mp_gap"]

task_types = ["regression","classification"] # "classification", "regression"

def ndarray_to_special_format(array, dtype_str, shape):
    """
    将numpy数组转换为特定的序列化格式。
    """
    return {
        "__ndarray__": [list(shape), dtype_str, array.flatten().tolist()]
    }
def atoms_to_special_dict(atoms):
    """
    将ASE Atoms对象转换为特定格式的字典，以符合问题描述中的JSON格式。
    """
    # 使用当前时间戳作为ctime和mtime
    current_time = time.time()  # 获取当前时间戳

    cell_array = ndarray_to_special_format(atoms.cell, "float64", atoms.cell.shape)
    numbers_array = ndarray_to_special_format(np.array(atoms.numbers), "int64", (len(atoms),))
    positions_array = ndarray_to_special_format(atoms.positions, "float64", atoms.positions.shape)
    pbc_array = ndarray_to_special_format(np.array(atoms.pbc, dtype=np.bool_), "bool", (3,))
    unique_id = str(atoms.unique_id)
    atoms_dict = {
        "1": {
            "cell": {"array": cell_array, "__ase_objtype__": "cell"},
            "ctime": current_time,  # 创建时间
            "mtime": current_time,  # 修改时间，此处与创建时间相同
            "numbers": numbers_array,
            "pbc": pbc_array,
            "positions": positions_array,
            # unique_id 和 user 根据需要填写或生成
            "unique_id": unique_id,
            "user": "vfung",
            "target":0
        },
        "ids": [1],
        "nextid": 2
    }
    return atoms_dict

def get_structure_identifier(structure):
    """
    生成结构的唯一标识符。
    """
    structure_str = str(structure)  # 使用结构的字符串表示
    return hash(structure_str)  # 返回哈希值作为唯一标识符

# 数据预处理
if __name__ == '__main__':
    # 初始化一个空的DataFrame来收集目标值
    targets = pd.DataFrame(columns=['index', 'target'])

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        dataset = load_dataset(dataset_name)

        # 只查看第一个结构作为示例
        first_row = dataset.iloc[0]
        structure = first_row["structure"]
        
        # 使用dir()获取所有属性和方法，然后过滤掉特殊方法
        attributes = [attr for attr in dir(structure) if not attr.startswith('__')]
        
        # 单纯打印属性列表
        print("Available attributes and methods in Structure object:")
        for attr in attributes:
            print(attr)
        # 打印属性列表和值
        print("Available attributes and methods in Structure object:")
        for attr in attributes:
            if not attr.startswith('__') and not callable(getattr(structure, attr)):
                try:
                    # 获取属性的值
                    value = getattr(structure, attr)
                    # 打印属性名和值
                    print(f"{attr}: {value}")
                except Exception as e:
                    # 如果访问属性时发生异常，则跳过
                    print(f"{attr}: <Error accessing this attribute>")

        # 如果你只想看属性和方法的数量，可以取消注释下面的行，有120个属性
        print(f"Total number of attributes and methods: {len(attributes)}")

        # 在此处添加检查并创建文件夹的逻辑
        dataset_folder_path = f"./data/{dataset_name}/"  # 指定每个数据集的文件夹路径
        os.makedirs(dataset_folder_path, exist_ok=True)  # 如果文件夹不存在，则创建
        
        
        adaptor = AseAtomsAdaptor()
        for index, row in dataset.iterrows():
            structure = row["structure"]
            target = row.iloc[1]
            unique_id = get_structure_identifier(structure)
            # atom1 = structure.to_ase_atoms()
            atom1 = adaptor.get_atoms(structure)
            atom1.unique_id = unique_id
            # 将结构转换为可以序列化的字典
            structure_dict = atoms_to_special_dict(atom1)
            # 将结构写入文件
            json_file_path = f"./data/{dataset_name}/{index}.json"  # 确保json_files目录已存在或先创建它
            with open(json_file_path, 'w') as f:
                json.dump(structure_dict, f, indent=4)
        
            # 收集目标值。使用一个临时DataFrame来存储当前的行，然后使用concat进行合并
            temp_df = pd.DataFrame([{'index': f"{index}", 'target': target}])
            targets = pd.concat([targets, temp_df], ignore_index=True)
        # 保存目标值到CSV文件
        targets.to_csv(f'./data/{dataset_name}/targets.csv', index=False, header=False)