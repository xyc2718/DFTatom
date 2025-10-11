import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
@dataclass
class BetaProjector:
    """
    描述一个非局域赝势投影算符 (beta function) 的数据结构。
    """
    l: int
    index: int
    radial_function: interp1d

@dataclass
class Pseudopotential:
    """
    存储赝势所有信息的类。
    """
    element: str
    z_valence: float
    l_max: int
    mesh_size: int
    functional: str
    is_norm_conserving: bool
    r_grid: np.ndarray
    v_local: interp1d
    d_matrix: np.ndarray  #将 d_matrix 移到有默认值的参数之前
    beta_projectors: List[BetaProjector]

    rho_atomic: Optional[np.ndarray] = None

    def print_info(self):
        """打印赝势的基本信息。"""
        print("="*50)
        print(f"Pseudopotential Information for Element: {self.element}")
        print("="*50)
        print(f"  Valence Electrons (Z_valence): {self.z_valence}")
        print(f"  Functional: {self.functional}")
        print(f"  Norm-Conserving: {self.is_norm_conserving}")
        print(f"  Max Angular Momentum (l_max): {self.l_max}")
        print(f"  Radial Mesh Size: {self.mesh_size}")
        print(f"  Number of Projectors: {len(self.beta_projectors)}")
        print(f"  D_ij Matrix shape: {self.d_matrix.shape}")
        print("="*50)


def load_pseudopotential_from_upf(filepath: str) -> Pseudopotential:
    """
    从一个 UPF (Unified Pseudopotential Format) 文件中加载赝势。

    Args:
        filepath: .upf 文件的路径。

    Returns:
        一个填充了所有赝势信息的 Pseudopotential 对象。
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        # 有些 UPF 文件可能不是标准的 XML，需要手动读取和清理
        with open(filepath, 'r') as f:
            xml_content = f.read()
        # 移除 INFO 区块，因为它不是标准的 XML
        if '<PP_INFO>' in xml_content and '</PP_INFO>' in xml_content:
            start = xml_content.find('<PP_INFO>')
            end = xml_content.find('</PP_INFO>') + len('</PP_INFO>')
            xml_content = xml_content[:start] + xml_content[end:]
        root = ET.fromstring(xml_content)

    # --- 1. 解析头文件 <PP_HEADER> ---
    header = root.find('PP_HEADER').attrib
    element = header['element'].strip()
    z_valence = float(header['z_valence'])
    l_max = int(header['l_max'])
    mesh_size = int(header['mesh_size'])
    functional = header['functional'].strip()
    is_norm_conserving = header.get('pseudo_type', 'NC').strip() == 'NC'
    num_projectors = int(header['number_of_proj'])

    # --- 2. 解析径向网格 <PP_MESH> ---
    r_text = root.find('PP_MESH/PP_R').text
    r_grid = np.fromstring(r_text, sep=' ')
    
    # --- 3. 解析局域势 <PP_LOCAL> ---
    v_local_text = root.find('PP_LOCAL').text
    v_local_value = np.fromstring(v_local_text, sep=' ')/2.0 #文件为Ry单位，转换为Hartree
    v_local = interp1d(r_grid, v_local_value, kind='cubic', fill_value="extrapolate")

    # --- 4. 解析非局域投影 <PP_NONLOCAL> ---
    beta_projectors = []
    nonlocal_tag = root.find('PP_NONLOCAL')
    for i in range(1, num_projectors + 1):
        beta_tag = nonlocal_tag.find(f'PP_BETA.{i}')
        beta_l = int(beta_tag.attrib['angular_momentum'])
        beta_index = int(beta_tag.attrib['index'])
        beta_func_text = beta_tag.text
        beta_radial_func = interp1d(r_grid,np.fromstring(beta_func_text, sep=' '), kind='cubic', fill_value="extrapolate")
        projector = BetaProjector(
            l=beta_l,
            index=beta_index,
            radial_function=beta_radial_func
        )
        beta_projectors.append(projector)

    # --- 5. 解析 D_ij 矩阵 <PP_DIJ> ---
    d_matrix_text = root.find('PP_NONLOCAL/PP_DIJ').text
    d_values = np.fromstring(d_matrix_text, sep=' ')/2.0 #文件为Ry单位，转换为Hartree
    d_matrix = d_values.reshape((num_projectors, num_projectors))
    
    # --- 6. (可选) 解析赝原子电荷密度 <PP_RHOATOM> ---
    rho_atomic_tag = root.find('PP_RHOATOM')
    rho_atomic = None
    if rho_atomic_tag is not None:
        rho_atomic_text = rho_atomic_tag.text
        rho_atomic = np.fromstring(rho_atomic_text, sep=' ')

    # --- 7. 组装并返回 Pseudopotential 对象 ---
    pseudo = Pseudopotential(
        element=element,
        z_valence=z_valence,
        l_max=l_max,
        mesh_size=mesh_size,
        functional=functional,
        is_norm_conserving=is_norm_conserving,
        r_grid=r_grid,
        v_local=v_local,
        beta_projectors=beta_projectors,
        d_matrix=d_matrix,
        rho_atomic=rho_atomic
    )
    
    return pseudo
