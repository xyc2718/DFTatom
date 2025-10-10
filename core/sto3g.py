import numpy as np
from scipy.special import gamma
from typing import List, Dict
from .basis_set import BasisSet, AtomicOrbital
import os
import re
from pprint import pprint
from pathlib import Path


STO3GPATH=Path(__file__).parent / "sto-3g.gbs"

def generate_gaussian_basis_set(
    element: str,
    primitives: List[Dict],
    radius_cutoff: float = 7.0,
    mesh_size: int = 701
) -> BasisSet:
    """
    根据给定的高斯基元参数，生成一个数值化的 BasisSet 对象。

    Args:
        element: 元素符号 (e.g., 'C').
        primitives: 一个字典列表，每个字典描述一个高斯基元。
                    例如: [{'l': 0, 'm': 0, 'alpha': 0.5}, {'l': 1, 'm': 0, 'alpha': 0.3}]
        radius_cutoff: 径向网格的最大半径 (Bohr)。
        mesh_size: 径向网格的点数。

    Returns:
        一个填充了数值原子轨道的 BasisSet 对象。
    """
    if not all(k in p for k in ['l', 'm', 'alpha'] for p in primitives):
        raise ValueError("每个基元字典必须包含 'l', 'm', 'alpha' 三个键。")

    # 1. 创建径向网格和 BasisSet 对象
    dr = radius_cutoff / (mesh_size - 1)
    r_grid = np.linspace(0, radius_cutoff, mesh_size)
    
    basis_set = BasisSet(element)
    basis_set.set_basis_parameters(radius_cutoff=radius_cutoff)
    basis_set.dr = dr # 手动设置dr
    
    # 用于为轨道命名和索引的计数器
    l_counts = {}

    # 2. 遍历所有高斯基元
    for prim in primitives:
        l = prim['l']
        m = prim['m']
        alpha = prim['alpha']

        # 3. 计算归一化系数 N
        # 归一化条件: integral( |N * r^l * exp(-alpha*r^2)|^2 * r^2 dr ) = 1
        # 这求解为 N^2 * integral(r^(2l+2) * exp(-2*alpha*r^2) dr) = 1
        # 使用高斯积分公式: integral(x^n * exp(-a*x^2) dx) from 0 to inf
        # = Gamma((n+1)/2) / (2 * a^((n+1)/2))
        n_integral = 2 * l + 2
        a_integral = 2 * alpha
        
        integral_val = gamma((n_integral + 1) / 2) / (2 * a_integral**((n_integral + 1) / 2))
        N = 1.0 / np.sqrt(integral_val)

        # 4. 在网格上生成数值径向函数
        radial_function = N * (r_grid**l) * np.exp(-alpha * r_grid**2)
        
        # 5. 创建并添加 AtomicOrbital 对象
        # 更新轨道索引 (n_index)，例如 1s, 2s, 2p, 3p...
        if l not in l_counts:
            l_counts[l] = 0
        l_counts[l] += 1
        n_index = l_counts[l]
        
        atomic_orbital = AtomicOrbital(
            n=n_index,  # 主量子数 (为了命名方便，直接用 n_index)
            l=l,
            m=m,
            orbital_type=l, # 用 l 作为类型标识
            n_index=n_index,
            r_grid=r_grid.copy(),
            radial_function=radial_function.copy()
        )
        basis_set.add_orbital(atomic_orbital)
        
    return basis_set


def get_basis_params_from_file(atomic_symbol: str) -> list[dict]:
    """
    从 STO3GPATH 环境变量指定的文件中加载并解析指定原子的 STO-3G 基组参数。

    Args:
        atomic_symbol: 原子的化学符号 (例如, 'H', 'C', 'Ti')。

    Returns:
        一个字典列表，每个字典代表一个高斯基函数，包含 {'l', 'm', 'alpha'} 键。

    Raises:
        ValueError: 如果 STO3GPATH 环境变量未设置，或文件中找不到指定的原子符号。
        FileNotFoundError: 如果 STO3GPATH 指向的文件不存在。
    """

    basis_file_path=STO3GPATH
    # 2. 读取和解析文件
    atom_data = []
    found_atom = False
    
    # 将输入的符号格式化，以便在文件中查找 (例如, 'C' -> 'C ')
    search_symbol = atomic_symbol.strip().upper()

    with open(basis_file_path, 'r') as f:
        for line in f:
            # 如果已经找到原子且遇到下一个原子分隔符，则停止解析
            if found_atom and '****' in line:
                break

            # 查找包含原子符号的行
            if re.match(rf'^{re.escape(search_symbol)}\s+\w', line.strip(), re.IGNORECASE):
                found_atom = True
                continue # 从下一行开始是壳层数据
            
            if found_atom:
                # 匹配壳层定义行, 例如 "SP   3   1.00"
                shell_match = re.match(r'^\s*([A-ZSP]+)\s+(\d+)', line.strip())
                if shell_match:
                    shell_type = shell_match.group(1)
                    num_primitives = int(shell_match.group(2))
                    
                    shell = {'type': shell_type, 'primitives': []}
                    
                    # 读取接下来 num_primitives 行的参数
                    for _ in range(num_primitives):
                        primitive_line = next(f).strip()
                        # 将Fortran的'D'替换为'E'并分割
                        parts = primitive_line.replace('D', 'E').replace('d', 'e').split()
                        exponent = float(parts[0])
                        shell['primitives'].append({'alpha': exponent})
                    
                    atom_data.append(shell)

    if not found_atom:
        raise ValueError(f"错误: 在文件 '{basis_file_path}' 中未找到原子符号 '{atomic_symbol}'。")

    # 3. 根据解析出的数据构建最终结果
    basis_functions = []
    shell_to_l = {'S': [0], 'P': [1], 'D': [2], 'SP': [0, 1]}

    for shell in atom_data:
        shell_type = shell['type'].upper()
        l_values = shell_to_l.get(shell_type, [])

        for primitive in shell['primitives']:
            alpha = primitive['alpha']
            for l in l_values:
                # m 的取值范围是 [-l, +l]
                for m in range(-l, l + 1):
                    basis_functions.append({
                        'l': l,
                        'm': m,
                        'alpha': alpha
                    })

    return basis_functions

def get_sto3g_basis(atomic_symbol: str, radius_cutoff: float = 7.0,
    mesh_size: int = 701 ) -> BasisSet:
    """
    获取指定原子的 STO-3G 基组，并生成对应的 BasisSet 对象。

    Args:
        atomic_symbol: 原子的化学符号 (例如, 'H', 'C', 'Ti')。
        radius_cutoff: 径向网格的最大半径 (Bohr)。
        mesh_size: 径向网格的点数。
    Returns:
        一个填充了数值原子轨道的 BasisSet 对象。
    """
    primitives = get_basis_params_from_file(atomic_symbol)
    basis_set = generate_gaussian_basis_set(atomic_symbol, primitives, radius_cutoff, mesh_size)
    return basis_set


# --- 使用示例 ---
if __name__ == '__main__':
    try:
        print("从文件加载碳 (C) 的 STO-3G 基组参数:")
        carbon_basis = get_basis_params_from_file('C')
        pprint(carbon_basis)
        print(f"\n碳原子的总基函数数量: {len(carbon_basis)}")
        print("\n" + "-"*50 + "\n")

    
    except (ValueError, FileNotFoundError) as e:
        print(e)