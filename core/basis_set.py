import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import interp1d
import os
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

@dataclass
class AtomicOrbital:
    """原子轨道类"""
    n: int                      # 主量子数
    l: int                      # 角量子数 
    m: int                      # 磁量子数
    orbital_type: int          # 轨道类型标识
    n_index: int               # 同一l下的轨道索引(如2s中的2, 3s中的3)
    r_grid: np.ndarray         # 径向网格
    radial_function: np.ndarray # 径向波函数 r*R_nl(r)
    
    def __post_init__(self):
        """初始化后处理，创建插值函数"""
        self.interpolator = interp1d(
            self.r_grid, 
            self.radial_function, 
            kind='cubic', 
            bounds_error=False, 
            fill_value=0.0
        )
        # 计算轨道名称
        self.orbital_name = self._get_orbital_name()
    
    def _get_orbital_name(self) -> str:
        """生成轨道名称，如1s, 2px, 3dxy等"""
        l_names = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        l_char = l_names.get(self.l, str(self.l))
        
        if self.l == 0:  # s轨道
            return f"{self.n_index + self.l + 1}{l_char}"
        elif self.l == 1:  # p轨道
            p_names = ['z', 'x', 'y']  # m = 0, 1, -1
            m_map = {0: 0, 1: 1, -1: 2}
            suffix = p_names[m_map.get(self.m, 0)]
            return f"{self.n_index + self.l + 1}{l_char}{suffix}"
        elif self.l == 2:  # d轨道
            d_names = ['z2', 'xz', 'yz', 'x2-y2', 'xy']  # m = 0, 1, -1, 2, -2
            m_map = {0: 0, 1: 1, -1: 2, 2: 3, -2: 4}
            suffix = d_names[m_map.get(self.m, 0)]
            return f"{self.n_index + self.l + 1}{l_char}{suffix}"
        else:
            return f"{self.n_index + self.l + 1}{l_char}({self.m})"
    
    def evaluate_radial(self, r: float) -> float:
        """计算径向波函数在r处的值"""
        return self.interpolator(r)
    
    def get_spherical_harmonic_info(self) -> Tuple[int, int]:
        """返回球谐函数的l, m量子数"""
        return self.l, self.m

class BasisSet:
    """原子轨道基组类"""
    
    def __init__(self, element: str):
        self.element = element
        self.orbitals: List[AtomicOrbital] = []
        
        # 基组信息
        self.energy_cutoff: Optional[float] = None
        self.radius_cutoff: Optional[float] = None
        self.lmax: Optional[int] = None
        self.dr: Optional[float] = None
        self.mesh_size: Optional[int] = None
        self.r_grid: Optional[np.ndarray] = None
        
        # 轨道统计信息
        self.n_orbitals_by_l: Dict[int, int] = {}  # 每个l值的轨道数
        self.orbital_indices: Dict[Tuple[int, int, int], int] = {}  # (n,l,m) -> index
        
    def add_orbital(self, orbital: AtomicOrbital):
        """添加轨道到基组"""
        # 检查网格一致性
        if self.r_grid is not None:
            if not np.allclose(self.r_grid, orbital.r_grid):
                raise ValueError("轨道的径向网格与基组不一致")
        else:
            self.r_grid = orbital.r_grid.copy()
            self.dr = self.r_grid[1] - self.r_grid[0] if len(self.r_grid) > 1 else None
            self.mesh_size = len(self.r_grid)
        
        # 添加轨道
        orbital_index = len(self.orbitals)
        self.orbitals.append(orbital)
        
        # 更新索引
        key = (orbital.n_index, orbital.l, orbital.m)
        self.orbital_indices[key] = orbital_index
        
        # 更新统计信息
        if orbital.l not in self.n_orbitals_by_l:
            self.n_orbitals_by_l[orbital.l] = 0
        self.n_orbitals_by_l[orbital.l] += 1
        
        # 更新lmax
        if self.lmax is None or orbital.l > self.lmax:
            self.lmax = orbital.l
    
    def get_orbital_count(self) -> int:
        """获取轨道总数"""
        return len(self.orbitals)
    
    def get_orbital_by_index(self, index: int) -> AtomicOrbital:
        """通过索引获取轨道"""
        return self.orbitals[index]
    
    def get_orbital_by_quantum_numbers(self, n: int, l: int, m: int) -> Optional[AtomicOrbital]:
        """通过量子数获取轨道"""
        index = self.orbital_indices.get((n, l, m))
        return self.orbitals[index] if index is not None else None
    
    def get_orbitals_by_l(self, l: int) -> List[AtomicOrbital]:
        """获取指定角量子数的所有轨道"""
        return [orbital for orbital in self.orbitals if orbital.l == l]
    
    def get_orbital_names(self) -> List[str]:
        """获取所有轨道名称列表"""
        return [orbital.orbital_name for orbital in self.orbitals]
    
    def get_orbital_info_table(self) -> List[Dict]:
        """获取轨道信息表"""
        info_table = []
        for i, orbital in enumerate(self.orbitals):
            info_table.append({
                'index': i,
                'name': orbital.orbital_name,
                'n': orbital.n_index + orbital.l + 1,
                'l': orbital.l,
                'm': orbital.m,
                'type': orbital.orbital_type
            })
        return info_table
    
    def print_basis_info(self):
        """打印基组信息"""
        print(f"Basis Set Information for {self.element}")
        print("=" * 50)
        print(f"Total orbitals: {self.get_orbital_count()}")
        print(f"Energy cutoff: {self.energy_cutoff} Ry")
        print(f"Radius cutoff: {self.radius_cutoff} a.u.")
        print(f"Lmax: {self.lmax}")
        print(f"Mesh size: {self.mesh_size}")
        print(f"dr: {self.dr} a.u.")
        
        print("\nOrbitals by angular momentum:")
        l_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        for l in sorted(self.n_orbitals_by_l.keys()):
            l_name = l_names.get(l, str(l))
            count = self.n_orbitals_by_l[l]
            print(f"  {l_name}: {count} orbitals")
        
        print("\nDetailed orbital list:")
        for info in self.get_orbital_info_table():
            print(f"  {info['index']:2d}: {info['name']:>6s} "
                  f"(n={info['n']}, l={info['l']}, m={info['m']:2d})")
    
    def set_basis_parameters(self, energy_cutoff: float = None, 
                           radius_cutoff: float = None):
        """设置基组参数"""
        if energy_cutoff is not None:
            self.energy_cutoff = energy_cutoff
        if radius_cutoff is not None:
            self.radius_cutoff = radius_cutoff


def load_basis_set_from_file(filepath: str) -> BasisSet:
    """
    从ABACUS轨道文件加载基组
    
    Args:
        filepath: ABACUS轨道文件路径
        
    Returns:
        BasisSet: 构建好的基组对象
    """
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"基组文件不存在: {filepath}")
    
    # 读取文件内容
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 解析结果存储
    parsed_data = {
        'element': None,
        'energy_cutoff': None,
        'radius_cutoff': None,
        'lmax': None,
        'mesh_size': None,
        'dr': None,
        'orbitals': []
    }
    
    # 解析文件
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 解析头部信息
        if line.startswith('Element'):
            parsed_data['element'] = line.split()[-1]
            
        elif line.startswith('Energy Cutoff(Ry)'):
            parsed_data['energy_cutoff'] = float(line.split()[-1])
            
        elif line.startswith('Radius Cutoff(a.u.)'):
            parsed_data['radius_cutoff'] = float(line.split()[-1])
            
        elif line.startswith('Lmax'):
            parsed_data['lmax'] = int(line.split()[-1])
            
        elif line.startswith('Mesh'):
            parsed_data['mesh_size'] = int(line.split()[-1])
            
        elif line.startswith('dr'):
            parsed_data['dr'] = float(line.split()[-1])
            
        # 检查是否到达轨道数据区域
        elif 'Type' in line and 'L' in line and 'N' in line:
            # 开始解析轨道数据
            i = _parse_orbital_data(lines, i + 1, parsed_data)
            continue
            
        i += 1
    
    # 验证必要信息
    if parsed_data['element'] is None:
        raise ValueError("无法从文件中提取元素信息")
    if parsed_data['mesh_size'] is None or parsed_data['dr'] is None:
        raise ValueError("无法从文件中提取网格信息")
    
    # 生成径向网格
    r_grid = np.arange(parsed_data['mesh_size']) * parsed_data['dr']
    
    # 创建基组对象
    basis_set = BasisSet(parsed_data['element'])
    basis_set.set_basis_parameters(
        energy_cutoff=parsed_data['energy_cutoff'],
        radius_cutoff=parsed_data['radius_cutoff']
    )
    
    # 为每个解析出的轨道创建所有m值的AtomicOrbital对象
    for orbital_data in parsed_data['orbitals']:
        l = orbital_data['l']
        n_index = orbital_data['n']
        orbital_type = orbital_data['type']
        radial_function = orbital_data['radial_function']
        
        # 确保径向函数长度与网格匹配
        if len(radial_function) != parsed_data['mesh_size']:
            # 如果长度不匹配，截断或补零
            if len(radial_function) > parsed_data['mesh_size']:
                radial_function = radial_function[:parsed_data['mesh_size']]
            else:
                # 补零到正确长度
                padded_func = np.zeros(parsed_data['mesh_size'])
                padded_func[:len(radial_function)] = radial_function
                radial_function = padded_func
        radial_function=radial_function*r_grid  # 转换为r*R形式
        # 为这个l值创建所有可能的m值轨道
        for m in range(-l, l + 1):
            atomic_orbital = AtomicOrbital(
                n=n_index + l + 1,  # 真实的主量子数
                l=l,
                m=m,
                orbital_type=orbital_type,
                n_index=n_index,
                r_grid=r_grid.copy(),
                radial_function=radial_function.copy()
            )
            basis_set.add_orbital(atomic_orbital)
    
    return basis_set

def _parse_orbital_data(lines: List[str], start_idx: int, parsed_data: Dict) -> int:
    """
    解析轨道数据部分
    
    Args:
        lines: 文件行列表
        start_idx: 开始解析的行号
        parsed_data: 存储解析结果的字典
        
    Returns:
        int: 解析结束后的行号
    """
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过空行
        if not line:
            i += 1
            continue
        
        # 检查是否是轨道头部行（三个整数）
        if _is_orbital_header_line(line):
            # 解析轨道头部信息
            parts = line.split()
            orbital_type = int(parts[0])
            l = int(parts[1])
            n = int(parts[2])
            
            # 读取这个轨道的径向函数数据
            radial_data = []
            i += 1
            
            # 继续读取数值数据直到遇到下一个轨道头部或文件结束
            while i < len(lines):
                data_line = lines[i].strip()
                
                # 如果是空行，跳过
                if not data_line:
                    i += 1
                    continue
                
                # 如果遇到下一个轨道头部，停止读取当前轨道数据
                if _is_orbital_header_line(data_line):
                    break
                
                # 如果遇到其他关键字（如SUMMARY等），停止读取
                if any(keyword in data_line.upper() for keyword in ['SUMMARY', 'END', '---']):
                    break
                
                # 尝试解析数值数据
                try:
                    # 分割并转换为浮点数
                    values = []
                    for val_str in data_line.split():
                        try:
                            val = float(val_str)
                            values.append(val)
                        except ValueError:
                            # 如果转换失败，可能遇到了非数值行，停止读取
                            break
                    
                    if values:  # 如果成功解析出数值
                        radial_data.extend(values)
                    else:
                        # 没有数值，可能是文件结束或其他内容
                        break
                        
                except Exception:
                    # 解析出错，停止读取当前轨道
                    break
                
                i += 1
            
            # 存储轨道数据
            if radial_data:  # 只有当有数据时才存储
                orbital_info = {
                    'type': orbital_type,
                    'l': l,
                    'n': n,
                    'radial_function': np.array(radial_data)
                }
                parsed_data['orbitals'].append(orbital_info)
            
            # 不增加i，因为可能当前行是下一个轨道的头部
            continue
        
        # 如果遇到结束标记，停止解析
        if any(keyword in line.upper() for keyword in ['SUMMARY', 'END', '---']):
            break
            
        i += 1
    
    return i

def _is_orbital_header_line(line: str) -> bool:
    """
    判断是否是轨道头部信息行（包含Type L N三个整数）
    
    Args:
        line: 待检查的行
        
    Returns:
        bool: 是否是轨道头部行
    """
    parts = line.split()
    
    # 必须正好有3个部分
    if len(parts) != 3:
        return False
    
    # 尝试将三个部分都转换为整数
    try:
        [int(part) for part in parts]
        return True
    except ValueError:
        return False
