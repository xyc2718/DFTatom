import numpy as np
from scipy.special import gamma
from typing import List, Dict
from .basis_set import BasisSet, AtomicOrbital
import os
import re
from pathlib import Path

# --- 路径定义 ---
STO3GPATH = Path(__file__).parent / "sto-3g.gbs"
STO6GPATH = Path(__file__).parent / "sto-6g.gbs"
AUGCCPVTZPATH = Path(__file__).parent / "aug-cc-pvtz.gbs"
ANORCCVQZPPATH = Path(__file__).parent / "ano-rcc-vqzp.gbs"
AUGCCPVDZPATH = Path(__file__).parent / "aug-cc-pvdz.gbs"
CCPVTZPATH = Path(__file__).parent / "cc-pvtz.gbs"
SADLEPVTZPATH = Path(__file__).parent / "sadlej pvtz.gbs"


def get_basis_params_from_file(atomic_symbol: str, basis_file_path=STO3GPATH) -> List[Dict]:
    """
    加载并解析基组文件，返回按壳层(Shell)组织的参数列表。
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
    return atom_data


def _expand_and_flatten_shells_legacy_mode(shells: List[Dict]) -> List[Dict]:
    """ 
    1. 按文件顺序 (Tight->Diffuse) 遍历壳层。
    2. 对每个壳层，立即展开其所有 m 分量 (m = -l ... l)。
    3. 生成一个巨大的临时列表。
    4. *最后* 对这个大列表进行反转。
    
    效果：
    - n 顺序变为 Diffuse -> Tight (符合 Code 2)。
    - m 顺序变为 l, l-1, ..., -l (即 1, 0, -1，符合 Code 2)。
    """
    temp_orbitals = []
    # 确保 S 在 P 之前被处理 (Code 2 逻辑)
    shell_to_l = {'S': [0], 'P': [1], 'D': [2], 'F': [3], 'G': [4], 'SP': [0, 1]}
    
    for shell in shells:
        shell_type = shell['type']
        l_values = shell_to_l.get(shell_type, [])
        
        for prim in shell['primitives']:
            alpha = prim['alpha']
            for l in l_values:
                # 关键：在这里展开 m (-l ... l)
                # 此时顺序是 -1, 0, 1
                for m in range(-l, l + 1):
                    temp_orbitals.append({
                        'l': l,
                        'm': m,
                        'alpha': alpha
                    })
                    
    # 核心步骤：整体反转
    # 此时顺序变为：Diffuse -> Tight, 且 m 变为 1, 0, -1
    return temp_orbitals[::-1]


def _calculate_primitive_norm(l: int, alpha: float) -> float:
    """计算单个高斯基元的解析归一化常数"""
    n_integral = 2 * l + 2
    a_integral = 2 * alpha
    integral_val = gamma((n_integral + 1) / 2) / (2 * a_integral**((n_integral + 1) / 2))
    return 1.0 / np.sqrt(integral_val)


def generate_gaussian_basis_set(
    element: str,
    shells: List[Dict],
    radius_cutoff: float = 7.0,
    mesh_size: int = 701,
    uncontracted: bool = False
) -> BasisSet:
    """
    基于线性均匀网格生成 BasisSet。
    """
    basis_set = BasisSet(element)
    basis_set.set_basis_parameters(radius_cutoff=radius_cutoff)
    
    dr = radius_cutoff / (mesh_size - 1)
    r_grid = np.linspace(0, radius_cutoff, mesh_size)
    dr_array = np.full_like(r_grid, dr)
    basis_set.dr = dr
    
    l_counts = {}

    if uncontracted:
        # 获取已经展开 m 并反转的列表
        flat_orbitals = _expand_and_flatten_shells_legacy_mode(shells)
        
        # 遍历这个列表，生成 AtomicOrbital
        for orb_params in flat_orbitals:
            l = orb_params['l']
            m = orb_params['m']
            alpha = orb_params['alpha']
            
            # 递增编号：确保每个基元都有唯一的 n
            if l not in l_counts: l_counts[l] = 0
            l_counts[l] += 1
            n_index = l_counts[l]
            
            # 解析归一化
            N = _calculate_primitive_norm(l, alpha)
            radial_function = N * (r_grid**(l+1)) * np.exp(-alpha * r_grid**2)
            
            # 直接创建轨道，不需要再循环 m (因为列表里已经包含了具体的 m)
            atomic_orbital = AtomicOrbital(
                n=n_index, 
                l=l, 
                m=m, 
                orbital_type=l, 
                n_index=n_index-1,
                r_grid=r_grid.copy(), 
                radial_function=radial_function.copy()
            )
            basis_set.add_orbital(atomic_orbital)
        
        return basis_set


    shell_to_l = {'S': [0], 'P': [1], 'D': [2], 'F': [3], 'G': [4], 'SP': [0, 1]}
    
    for shell in shells:
        shell_type = shell['type']
        
        for l in shell_to_l.get(shell_type, []):
            radial_function = np.zeros_like(r_grid)
            
            if l not in l_counts: l_counts[l] = 0
            l_counts[l] += 1
            n_index = l_counts[l]

            for prim in shell['primitives']:
                alpha = prim['alpha']
                if shell_type == 'SP':
                    coeff = prim['coeff_s'] if l == 0 else prim['coeff_p']
                else:
                    coeff = prim['coeff']
                
                N_prim = _calculate_primitive_norm(l, alpha)
                radial_function += coeff * N_prim * (r_grid**(l+1)) * np.exp(-alpha * r_grid**2)
            
            norm_sq = np.dot(radial_function**2, dr_array)
            if norm_sq > 1e-15:
                radial_function *= (1.0 / np.sqrt(norm_sq))
            
            for m in range(-l, l + 1):
                atomic_orbital = AtomicOrbital(
                    n=n_index, l=l, m=m, orbital_type=l, n_index=n_index-1,
                    r_grid=r_grid.copy(), radial_function=radial_function.copy()
                )
                basis_set.add_orbital(atomic_orbital)
                
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
    """
    basis_set = BasisSet(element)
    basis_set.set_basis_parameters(radius_cutoff=radius_cutoff)
    
    r_grid = np.geomspace(r_min, radius_cutoff, mesh_size)
    dx = np.log(radius_cutoff / r_min) / (mesh_size - 1)
    
    if hasattr(basis_set, 'dx'): basis_set.dx = dx
    basis_set.dr = None 
    basis_set.r_grid = r_grid

    l_counts = {}
    

    if uncontracted:
        flat_orbitals = _expand_and_flatten_shells_legacy_mode(shells)
        
        for orb_params in flat_orbitals:
            l = orb_params['l']
            m = orb_params['m']
            alpha = orb_params['alpha']
            
            if l not in l_counts: l_counts[l] = 0
            l_counts[l] += 1
            n_index = l_counts[l]
            
            N = _calculate_primitive_norm(l, alpha)
            radial_function = N * (r_grid**(l+1)) * np.exp(-alpha * r_grid**2)
            
            atomic_orbital = AtomicOrbital(
                n=n_index, l=l, m=m, orbital_type=l, n_index=n_index-1,
                r_grid=r_grid.copy(), radial_function=radial_function.copy()
            )
            basis_set.add_orbital(atomic_orbital)
        
        return basis_set

    dr_array = np.gradient(r_grid) 
    shell_to_l = {'S': [0], 'P': [1], 'D': [2], 'F': [3], 'G': [4], 'SP': [0, 1]}
    
    for shell in shells:
        shell_type = shell['type']
        
        for l in shell_to_l.get(shell_type, []):
            radial_function = np.zeros_like(r_grid)
            
            if l not in l_counts: l_counts[l] = 0
            l_counts[l] += 1
            n_index = l_counts[l]

            for prim in shell['primitives']:
                alpha = prim['alpha']
                if shell_type == 'SP':
                    coeff = prim['coeff_s'] if l == 0 else prim['coeff_p']
                else:
                    coeff = prim['coeff']
                
                N_prim = _calculate_primitive_norm(l, alpha)
                radial_function += coeff * N_prim * (r_grid**(l+1)) * np.exp(-alpha * r_grid**2)
            
            norm_sq = np.dot(radial_function**2, dr_array)
            if norm_sq > 1e-15:
                radial_function *= (1.0 / np.sqrt(norm_sq))
            
            for m in range(-l, l + 1):
                atomic_orbital = AtomicOrbital(
                    n=n_index, l=l, m=m, orbital_type=l, n_index=n_index-1,
                    r_grid=r_grid.copy(), radial_function=radial_function.copy()
                )
                basis_set.add_orbital(atomic_orbital)
                
    return basis_set


# --- 接口函数 ---
def get_sto3g_basis(atomic_symbol: str, radius_cutoff: float = 7.0,
    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, STO3GPATH)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-5, uncontracted=uncontracted)

def get_sto6g_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, STO6GPATH)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-5, uncontracted=uncontracted)

def get_aug_cc_pvtz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, AUGCCPVTZPATH)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)

def get_aug_cc_pvdz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, AUGCCPVDZPATH)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)

def get_ano_rcc_vqzp_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, ANORCCVQZPPATH)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)
    
def get_cc_pvtz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, CCPVTZPATH)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)
    
def get_sadlej_pvtz_basis(atomic_symbol: str, radius_cutoff: float = 10.0,
    mesh_size: int = 701, grid_type='linear', uncontracted: bool = False) -> BasisSet:
    shells = get_basis_params_from_file(atomic_symbol, SADLEPVTZPATH)
    if grid_type == 'linear':
        return generate_gaussian_basis_set(atomic_symbol, shells, radius_cutoff, mesh_size, uncontracted)
    elif grid_type == 'log':
        return generate_gaussian_basis_set_log(atomic_symbol, shells, radius_cutoff, mesh_size, r_min=1e-6, uncontracted=uncontracted)