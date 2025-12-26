import numpy as np
from scipy.special import gamma
from typing import List, Dict
from .basis_set import BasisSet, AtomicOrbital
import os
import re
from pathlib import Path

# 注意：通常在脚本中使用 __file__，如果 file 是你自定义的变量，请保持原样
# 这里假设是指当前文件路径
STO3GPATH = Path(__file__).parent / "sto-3g.gbs"
STO6GPATH = Path(__file__).parent / "sto-6g.gbs"
AUGCCPVTZPATH = Path(__file__).parent / "aug-cc-pvtz.gbs"
ANORCCVQZPPATH = Path(__file__).parent / "ano-rcc-vqzp.gbs"
AUGCCPVDZPATH = Path(__file__).parent / "aug-cc-pvdz.gbs"
CCPVTZPATH = Path(__file__).parent / "cc-pvtz.gbs"
SADLEPVTZPATH = Path(__file__).parent / "sadlej pvtz.gbs"


def get_basis_params_from_file(atomic_symbol: str, basis_file_path=STO3GPATH, uncontracted_parsing=False) -> List[Dict]:
    """
    加载并解析基组文件。
    
    Args:
        uncontracted_parsing: 如果为True，则按代码2的方式解析，忽略收缩系数
    """
    atom_data = []
    found_atom = False
    search_symbol = atomic_symbol.strip().upper()
    
    if not os.path.exists(basis_file_path):
        raise FileNotFoundError(f"基组文件未找到: {basis_file_path}")
    
    with open(basis_file_path, 'r') as f:
        for line in f:
            if found_atom and '****' in line:
                break
            if re.match(rf'^{re.escape(search_symbol)}\s+\w', line.strip(), re.IGNORECASE):
                found_atom = True
                continue
            if found_atom:
                shell_match = re.match(r'^\s*([A-Z]+)\s+(\d+)', line.strip(), re.IGNORECASE)
                if shell_match:
                    shell_type = shell_match.group(1).upper()
                    num_primitives = int(shell_match.group(2))
                    
                    if uncontracted_parsing:
                        # 按代码2的方式解析：忽略收缩系数，每个基元独立
                        shell_to_l = {'S': [0], 'P': [1], 'D': [2], 'F': [3], 'G': [4], 'SP': [0, 1]}
                        l_values = shell_to_l.get(shell_type, [])
                        
                        for _ in range(num_primitives):
                            parts = next(f).strip().replace('D', 'E').replace('d', 'e').split()
                            alpha = float(parts[0])
                            # 忽略收缩系数，直接为每个l,m生成独立基函数
                            for l in l_values:
                                for m in range(-l, l + 1):
                                    atom_data.append({
                                        'l': l,
                                        'm': m, 
                                        'alpha': alpha
                                    })
                    else:
                        # 按代码1原来的方式解析：保留壳层结构和收缩系数
                        current_shell = {'type': shell_type, 'primitives': []}
                        for _ in range(num_primitives):
                            parts = next(f).strip().replace('D', 'E').replace('d', 'e').split()
                            exponent = float(parts[0])
                            if shell_type == 'SP':
                                current_shell['primitives'].append({
                                    'alpha': exponent,
                                    'coeff_s': float(parts[1]),
                                    'coeff_p': float(parts[2])
                                })
                            else:
                                current_shell['primitives'].append({
                                    'alpha': exponent,
                                    'coeff': float(parts[1])
                                })
                        atom_data.append(current_shell)
    
    if not found_atom:
        raise ValueError(f"未找到原子符号: {atomic_symbol}")
    
    if uncontracted_parsing:
        return atom_data[::-1]  # 如果需要倒序
    return atom_data


def _calculate_primitive_norm(l: int, alpha: float) -> float:
    """
    计算单个高斯基元的解析归一化常数 N，
    使得 ∫ |N r^(l+1) e^{-alpha r^2}|^2 dr = 1 （以u(r)=rR(r)的定义）
    N = [ Gamma((2l+3)/2) / (2 * (2alpha)^((2l+3)/2)) ]^{-1/2}
    """
    n_integral = 2 * l + 2
    a_integral = 2 * alpha
    integral_val = gamma((n_integral + 1) / 2) / (2 * a_integral**((n_integral + 1) / 2))
    return 1.0 / np.sqrt(integral_val)


def _decontract_shells_like_code2(shells: List[Dict]) -> List[Dict]:
    """
    生成“代码2风格”的解收缩壳层：
    - 非SP：每个 primitive -> 独立壳层（同原类型），系数强制为 1.0，且只有该 primitive。
    - SP：每个 primitive -> 两个独立壳层：S (coeff=1.0) 和 P (coeff=1.0)。
    注意：这里的“解收缩”移除了原始收缩系数信息，匹配代码2的行为。
    """
    uncontracted = []
    for shell in shells:
        t = shell['type'].upper()
        if t == 'SP':
            for prim in shell['primitives']:
                alpha = prim['alpha']
                # 生成一个仅含单primitive的S壳层，coeff=1
                uncontracted.append({'type': 'S', 'primitives': [{'alpha': alpha, 'coeff': 1.0}]})
                # 生成一个仅含单primitive的P壳层，coeff=1
                uncontracted.append({'type': 'P', 'primitives': [{'alpha': alpha, 'coeff': 1.0}]})
        else:
            for prim in shell['primitives']:
                alpha = prim['alpha']
                # 仅保留alpha，coeff=1（丢弃原收缩系数），单primitive壳层
                uncontracted.append({'type': t, 'primitives': [{'alpha': alpha, 'coeff': 1.0}]})
    return uncontracted


def _shell_type_to_ls(shell_type: str):
    shell_to_l = {'S': [0], 'P': [1], 'D': [2], 'F': [3], 'G': [4], 'SP': [0, 1]}
    return shell_to_l.get(shell_type.upper(), [])


def _build_radial_function_linear_combo(r_grid, l: int, shell: Dict) -> np.ndarray:
    """
    常规定义（收缩态）：按壳层内的primitives与系数线性组合后，再进行数值归一化。
    支持 SP 的 coeff_s/coeff_p（调用处按具体l选择对应系数）。
    注意：调用者应传入已根据 l 提取好的 (alpha, coeff) 列表，或传入完整shell并在内部选择。
    这里直接使用 shell 的原始结构并按 l 选择正确的 coeff。
    """
    radial = np.zeros_like(r_grid)
    t = shell['type'].upper()
    for prim in shell['primitives']:
        alpha = prim['alpha']
        if t == 'SP':
            # 调用时按l区分：l=0 -> coeff_s; l=1 -> coeff_p
            coeff = prim['coeff_s'] if l == 0 else prim['coeff_p']
        else:
            coeff = prim['coeff']
        N = _calculate_primitive_norm(l, alpha)
        radial += coeff * N * (r_grid**(l + 1)) * np.exp(-alpha * r_grid**2)
    return radial


def _build_radial_function_single_primitive(r_grid, l: int, alpha: float) -> np.ndarray:
    """
    代码2风格（解收缩态）：每个primitive独立，使用解析归一化常数，不做数值归一化修正。
    """
    N = _calculate_primitive_norm(l, alpha)
    return N * (r_grid**(l + 1)) * np.exp(-alpha * r_grid**2)


def generate_gaussian_basis_set(
    element: str,
    shells: List[Dict],
    radius_cutoff: float = 7.0,
    mesh_size: int = 701,
    uncontracted: bool = False
) -> BasisSet:
    """
    基于线性均匀网格生成 BasisSet。
    - uncontracted=False: 按原壳层收缩系数组合（常规定义），随后数值归一化。
    - uncontracted=True: 代码2风格的解收缩（每个primitive独立、系数=1、解析归一化、不再数值归一化）。
    """
    if uncontracted:
        shells = _decontract_shells_like_code2(shells)
        
    dr = radius_cutoff / (mesh_size - 1)
    r_grid = np.linspace(0.0, radius_cutoff, mesh_size)
    dr_array = np.full_like(r_grid, dr)

    basis_set = BasisSet(element)
    basis_set.set_basis_parameters(radius_cutoff=radius_cutoff)
    basis_set.dr = dr

    l_counts = {}

    for shell in shells:
        shell_type = shell['type'].upper()
        ls = _shell_type_to_ls(shell_type)

        for l in ls:
            if uncontracted:
                # 每个壳层只含一个primitive（我们在解收缩器中保证了）
                prim = shell['primitives'][0]
                alpha = prim['alpha']
                radial_function = _build_radial_function_single_primitive(r_grid, l, alpha)
                # 不做数值归一化，保持与代码2一致
            else:
                # 按原coeff进行线性组合，然后数值归一化
                radial_function = _build_radial_function_linear_combo(r_grid, l, shell)
                norm_sq = np.dot(radial_function**2, dr_array)
                if norm_sq > 0:
                    radial_function *= (1.0 / np.sqrt(norm_sq))

            # 计数与添加AO（对每个l同一个径向函数复制(2l+1)个m）
            if l not in l_counts:
                l_counts[l] = 0
            l_counts[l] += 1
            n_index = l_counts[l]

            for m in range(-l, l + 1):
                ao = AtomicOrbital(
                    n=n_index,
                    l=l,
                    m=m,
                    orbital_type=l,
                    n_index=n_index - 1,
                    r_grid=r_grid.copy(),
                    radial_function=radial_function.copy()
                )
                basis_set.add_orbital(ao)

    return basis_set


def generate_gaussian_basis_set_log(
    element: str,
    shells: List[Dict],
    radius_cutoff: float = 7.0,
    mesh_size: int = 701,
    r_min: float = 1e-5,
    uncontracted: bool = False
) -> BasisSet:
    """
    基于对数网格生成 BasisSet。
    - uncontracted=False: 按原壳层收缩系数组合（常规定义），随后数值归一化（用变步长权重）。
    - uncontracted=True: 代码2风格的解收缩（每个primitive独立、系数=1、解析归一化、不再数值归一化）。
    """
    if uncontracted:
        shells = _decontract_shells_like_code2(shells)
        
    r_grid = np.geomspace(r_min, radius_cutoff, mesh_size)
    # 变步长积分权重
    dr_array = np.gradient(r_grid)
    dx = np.log(radius_cutoff / r_min) / (mesh_size - 1)

    basis_set = BasisSet(element)
    basis_set.set_basis_parameters(radius_cutoff=radius_cutoff)
    if hasattr(basis_set, 'dx'):
        basis_set.dx = dx
    basis_set.dr = None
    basis_set.r_grid = r_grid

    l_counts = {}

    for shell in shells:
        shell_type = shell['type'].upper()
        ls = _shell_type_to_ls(shell_type)

        for l in ls:
            if uncontracted:
                prim = shell['primitives'][0]
                alpha = prim['alpha']
                radial_function = _build_radial_function_single_primitive(r_grid, l, alpha)
                # 不做数值归一化，匹配代码2
            else:
                radial_function = _build_radial_function_linear_combo(r_grid, l, shell)
                norm_sq = np.dot(radial_function**2, dr_array)
                if norm_sq > 0:
                    radial_function *= (1.0 / np.sqrt(norm_sq))

            if l not in l_counts:
                l_counts[l] = 0
            l_counts[l] += 1
            n_index = l_counts[l]
            
            for m in range(-l, l + 1):
                ao = AtomicOrbital(
                    n=n_index,
                    l=l,
                    m=m,
                    orbital_type=l,
                    n_index=n_index - 1,
                    r_grid=r_grid.copy(),
                    radial_function=radial_function.copy()
                )
                basis_set.add_orbital(ao)

    return basis_set


# --- 接口函数 ---

def get_sto3g_basis(atomic_symbol: str, radius_cutoff: float = 7.0,
                    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, STO3GPATH,uncontracted_parsing=uncontracted)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-5, uncontracted=uncontracted)
    else:
        raise ValueError("grid_type 必须为 'linear' 或 'log'")


def get_sto6g_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
                    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, STO6GPATH,uncontracted_parsing=uncontracted)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-5, uncontracted=uncontracted)
    else:
        raise ValueError("grid_type 必须为 'linear' 或 'log'")


def get_aug_cc_pvtz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
                          mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, AUGCCPVTZPATH,uncontracted_parsing=uncontracted)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)
    else:
        raise ValueError("grid_type 必须为 'linear' 或 'log'")


def get_aug_cc_pvdz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
                          mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, AUGCCPVDZPATH,uncontracted_parsing=uncontracted)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)
    else:
        raise ValueError("grid_type 必须为 'linear' 或 'log'")


def get_ano_rcc_vqzp_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
                           mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, ANORCCVQZPPATH,uncontracted_parsing=uncontracted)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)
    else:
        raise ValueError("grid_type 必须为 'linear' 或 'log'")


def get_cc_pvtz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
                      mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, CCPVTZPATH,uncontracted_parsing=uncontracted)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)
    else:
        raise ValueError("grid_type 必须为 'linear' 或 'log'")


def get_sadlej_pvtz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
                          mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, SADLEPVTZPATH,uncontracted_parsing=uncontracted)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)
    else:
        raise ValueError("grid_type 必须为 'linear' 或 'log'")