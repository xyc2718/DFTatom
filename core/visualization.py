import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.special import sph_harm
import os
# 尝试导入结果类，如果不存在则使用占位符（方便单独测试）
try:
    from core.LSDA import KSResults
    from core.HF import HFResults
    CalculationResults = Union[KSResults, HFResults]
except ImportError:
    CalculationResults = "Any"  # 仅作类型提示占位


def get_radial_data(res, spin, orb_idx,divide_by_r=False):
    """从结果对象中提取 R(r) 数据，辅助函数"""
    calc = res.integral_calc
    r = calc.r_grid
    
    # 获取系数
    C = res.coefficients_alpha if spin == 'alpha' else res.coefficients_beta
    coeffs = C[:, orb_idx]
    
    # 组合波函数: psi_chi = sum(c * chi)
    psi_chi = np.einsum('m,mr->r', coeffs, calc.radial_functions)
    
    # 转换为 R(r) = chi/r，处理原点
    with np.errstate(divide='ignore', invalid='ignore'):
        if divide_by_r:
            psi_r = psi_chi / r
        else:
            psi_r = psi_chi 
        # 简单修补原点
        if np.abs(r[0]) < 1e-9:
            psi_r[0] = psi_r[1]
            
    return r, psi_r


def plot_radial_wavefunctions(results: CalculationResults, spin='alpha', max_r=5.0):
    """
    绘制 1D 径向波函数 R(r)
    
    """
    calc = results.integral_calc
    r_grid = calc.r_grid
    radial_funcs = calc.radial_functions 
    
    # 获取数据
    if spin == 'alpha':
        C = results.coefficients_alpha
        E = results.orbital_energies_alpha
        occs = results.electron_configuration.get('occupied_alpha', [])
    else:
        C = results.coefficients_beta
        E = results.orbital_energies_beta
        occs = results.electron_configuration.get('occupied_beta', [])

    n_occ = len(occs)
    labels = [o['name'] for o in occs]

    plt.figure(figsize=(10, 6))
    
    # 颜色循环，避免轨道太多颜色重复
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_occ))

    for i in range(n_occ):
        coeffs = C[:, i] 
        # 线性组合: psi_chi = Sum(coeffs * radial_funcs)
        # radial_funcs 存储的是 chi(r) = r * R(r)
        psi_chi = np.einsum('m,mr->r', coeffs, radial_funcs)
        
        # 转换为 R(r) = psi_chi / r
        # 使用 mask 处理 r=0，比直接赋值更安全
        with np.errstate(divide='ignore', invalid='ignore'):
            psi_r = psi_chi / r_grid
            # 简单线性外推处理原点：R(0) ≈ 2*R(dr) - R(2*dr) 或直接取邻近点
            if np.isnan(psi_r[0]) or np.isinf(psi_r[0]):
                psi_r[0] = psi_r[1] 
            
        plt.plot(r_grid, psi_r, label=f"{labels[i]} (E={E[i]:.2f} Ha)", color=colors[i], linewidth=1.5)

    # 自动判断方法名
    method_name = "Hartree-Fock" if "HFResults" in str(type(results)) else "DFT/LSDA"
    
    plt.title(f"Radial Wavefunctions ({spin}-spin) - {method_name}")
    plt.xlabel("Distance r (Bohr)")
    plt.ylabel(r"Radial Wavefunction $R(r)$") # 修正 label
    plt.xlim(0, max_r)
    # 添加零线
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放外侧防止遮挡
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_orbital_contour(results: CalculationResults, orbital_index: int, spin='alpha', 
                         plane='xz', range_au=4.0, grid_points=120): # 增加网格密度
    """
    绘制特定轨道的 2D 截面等值线图
    
    """
    # 1. 准备网格
    d = np.linspace(-range_au, range_au, grid_points)
    X, Y = np.meshgrid(d, d)
    
    # 2. 获取基组信息
    basis = results.integral_calc.basis
    orbital_names = basis.get_orbital_names()
    is_real_basis = getattr(results.integral_calc, 'real_basis', False)
    
    # 3. 获取系数
    if spin == 'alpha':
        coeffs = results.coefficients_alpha[:, orbital_index]
        energy = results.orbital_energies_alpha[orbital_index]
    else:
        coeffs = results.coefficients_beta[:, orbital_index]
        energy = results.orbital_energies_beta[orbital_index]
        
    # 4. 坐标变换
    if plane == 'xz':
        x_g, y_g, z_g = X, np.zeros_like(X), Y 
        title_plane = "XZ Plane (y=0)"
    elif plane == 'xy':
        x_g, y_g, z_g = X, Y, np.zeros_like(X)
        title_plane = "XY Plane (z=0)"
    elif plane == 'yz':
        x_g, y_g, z_g = np.zeros_like(X), X, Y
        title_plane = "YZ Plane (x=0)"
    
    # 球坐标 (r, theta, phi)
    r_grid_mesh = np.sqrt(x_g**2 + y_g**2 + z_g**2)
    # 防止 r=0 在 arccos 中导致 NaN，加一个小 epsilon
    r_safe = r_grid_mesh + 1e-12
    
    theta_grid = np.arccos(z_g / r_safe) # 极角 [0, pi]
    phi_grid = np.arctan2(y_g, x_g)      # 方位角 [-pi, pi]
    
    Psi_grid = np.zeros_like(X, dtype=np.complex128)
    
    # 5. 累加波函数
    for i, orb in enumerate(basis.orbitals):
        c_i = coeffs[i]
        if abs(c_i) < 1e-6: continue 
        
        # --- A. 径向部分 R(r) = chi(r)/r ---
        chi_values = orb.interpolator(r_grid_mesh) # 插值必须用原始 r_grid_mesh (包含0)
        
        # 处理 r=0 处的除法
        with np.errstate(divide='ignore', invalid='ignore'):
            radial_part = chi_values / r_grid_mesh
            # 修正原点：如果是 s 轨道 (l=0)，R(0) 非零；如果是 p,d... (l>0)，R(0)=0
            # 简单的数值处理是将中心点设为临近值的平均或 0 (视 l 而定)
            # 这里简单处理：将中心点设为 0 (对于 p,d 准确，对于 s 会有一个像素的黑点，但在 contour 中通常不可见)
            radial_part[r_grid_mesh < 1e-8] = 0.0 
            if orb.l == 0:
                # 对 s 轨道做一点修正，取稍微偏离原点的值
                radial_part[r_grid_mesh < 1e-8] = orb.interpolator(1e-4) / 1e-4

        # --- B. 角向部分 ---
        l, m = orb.l, orb.m
        
        if is_real_basis:
            # === 实球谐函数 (含归一化系数近似) ===
            # 添加 sqrt 系数是为了让 s, p, d 之间的相对大小物理意义更准确
            if l == 0:   # s
                ang_part = 1.0 # (1/sqrt(4pi)) 通常包含在径向定义中或忽略
            elif l == 1: # p
                # pz ~ z/r, px ~ x/r, py ~ y/r.  Factor sqrt(3)
                pre = np.sqrt(3)
                if m == 0:    ang_part = pre * z_g / r_safe
                elif m == 1:  ang_part = pre * x_g / r_safe
                elif m == -1: ang_part = pre * y_g / r_safe
            elif l == 2: # d
                # Factor sqrt(5)/2 or similar depending on def. Let's use proportional shapes.
                pre = np.sqrt(5/4) * 3 # 粗略系数
                if m == 0:    ang_part = (3*z_g**2 - r_grid_mesh**2) / (r_safe**2) * np.sqrt(5)/2 # dz2
                elif m == 1:  ang_part = np.sqrt(15) * (x_g * z_g) / (r_safe**2) # dxz
                elif m == -1: ang_part = np.sqrt(15) * (y_g * z_g) / (r_safe**2) # dyz
                elif m == 2:  ang_part = (np.sqrt(15)/2) * (x_g**2 - y_g**2) / (r_safe**2) # dx2-y2
                elif m == -2: ang_part = np.sqrt(15) * (x_g * y_g) / (r_safe**2) # dxy
            else:
                ang_part = 1.0 # Higher l not implemented manually
        else:
            # === 复球谐函数 ===
            # scipy.special.sph_harm(m, n, theta, phi)
            # 参数定义：theta 为方位角 (0-2pi)，phi 为极角 (0-pi)
            # 代码对应：theta -> phi_grid, phi -> theta_grid
            ang_part = sph_harm(m, l, phi_grid, theta_grid)

        Psi_grid += c_i * radial_part * ang_part

    # 6. 绘图
    plt.figure(figsize=(7, 6))
    Psi_plot = np.real(Psi_grid)
    
    # 对称色标
    val_max = np.max(np.abs(Psi_plot))
    if val_max < 1e-10: val_max = 1.0
    levels = np.linspace(-val_max, val_max, 60) # 增加层级使渐变更平滑
    
    cf = plt.contourf(X, Y, Psi_plot, levels=levels, cmap='RdBu_r')
    cbar = plt.colorbar(cf)
    if is_real_basis:
        cbar.set_label('Wavefunction Amplitude (a.u.)', rotation=270, labelpad=15)
    else:
        cbar.set_label('Real Part of Wavefunction Amplitude (a.u.)', rotation=270, labelpad=15)
    
    plt.scatter([0], [0], marker='+', color='black', s=80, alpha=0.7, label='Nucleus')
    plt.contour(X, Y, Psi_plot, levels=[0], colors='black', linewidths=0.5, alpha=0.3)
    
    # 标题生成
    max_c_idx = np.argmax(np.abs(coeffs))
    orb_label_guess = orbital_names[max_c_idx]
    title_spin = r"$\alpha$" if spin=='alpha' else r"$\beta$"
    if is_real_basis:
        
        plt.title(f"Orbital {orbital_index}: {orb_label_guess} ({title_spin})\n"
              f"E = {energy:.4f} Ha | Plane: {plane}")
    else:
        plt.title(f"Real part of Orbital {orbital_index}: {orb_label_guess} ({title_spin})\n"
              f"E = {energy:.4f} Ha | Plane: {plane} | Complex Spherical Harmonics")
    plt.xlabel(f"{plane[0].upper()} (Bohr)")
    plt.ylabel(f"{plane[1].upper()} (Bohr)")
    plt.axis('equal') # 保证圆形不变形
    plt.tight_layout()

def get_dominant_composition(coeffs, basis_names):
    """
    分析轨道的主要成分
    返回: (成分名称, 占比)
    """
    # 找到系数绝对值最大的基函数索引
    # 注意：这里使用 c_i^2 作为权重的简单近似（对于正交基是准确的，非正交基是近似）
    idx = np.argmax(np.abs(coeffs))
    name = basis_names[idx]
    
    # 归一化贡献度 (c_i^2 / sum(c^2))
    norm = np.sum(coeffs**2)
    weight = (coeffs[idx]**2) / norm
    
    # 清理名称 (例如 "C 2px" -> "2px")
    # 假设 basis_names 格式为 "Atom Label Function" 或直接是 Function
    clean_name = name.split()[-1] 
    
    # #去除数字
    # clean_name = ''.join([i for i in clean_name if not i.isdigit()])
    
    # # 转换为 LaTeX 格式 (例如 2px -> 2p_x)
    # if 'p' in clean_name and len(clean_name) > 2:
    #     clean_name = clean_name.replace('x', '_x').replace('y', '_y').replace('z', '_z')
        
    return clean_name, weight
def save_table2(filename, res_hf, res_lsda,output_dir,max_orbitals=None):
    """
    保存包含轨道成分分析和占据数的详细表格
    格式: Index | Spin | HF(E, Comp, Occ) | LSDA(E, Comp, Occ)
    """
    ha_to_ev = 27.2114
    OUT_DIR=output_dir
    
    # 获取基组名称列表
    basis = res_hf.integral_calc.basis
    orb_names = basis.get_orbital_names()
    
    # 获取占据数 (如果没有 explicit occupation 属性，则根据能量/ Aufbau 构造)
    # 假设 Results 对象里有 occupations_alpha/beta 属性
    # 如果没有，这里做一个简单的 Aufbau 推断（仅供参考，最好你的类里有存）
    def get_occs(res, spin):
        try:
            if spin == 'alpha':
                return res.occ_alpha
            elif spin == 'beta':
                return res.occ_beta
      
        except:
            # Fallback: 全1或0 (仅适用于简单 HF)
            n_occ = len(res.electron_configuration['occupied_' + spin])
            total_n = len(res.orbital_energies_alpha)
            occs = np.zeros(total_n)
            occs[:n_occ] = 1.0
            return occs

    hf_occ_a = get_occs(res_hf, 'alpha')
    hf_occ_b = get_occs(res_hf, 'beta')
    lsda_occ_a = get_occs(res_lsda, 'alpha')
    lsda_occ_b = get_occs(res_lsda, 'beta')

    with open(os.path.join(OUT_DIR, filename), 'w') as f:
        # 确定要打印的轨道数量 (取两者中较大的基组大小)
        n_orbitals = len(res_hf.orbital_energies_alpha)
        
        for i in range(n_orbitals):
            if max_orbitals is not None and i >= max_orbitals:
                break   
            # --- Alpha Spin Row ---
            idx_str = f"{i+1}"
            spin_str = r"$\alpha$"
            
            # HF Data
            hf_e = res_hf.orbital_energies_alpha[i]
            hf_comp, hf_pct = get_dominant_composition(res_hf.coefficients_alpha[:, i], orb_names)
            hf_occ = hf_occ_a[i]
            
            # LSDA Data
            lsda_e = res_lsda.orbital_energies_alpha[i]
            lsda_comp, lsda_pct = get_dominant_composition(res_lsda.coefficients_alpha[:, i], orb_names)
            lsda_occ = lsda_occ_a[i]
            
            # 格式化: 能量(Ha) & 成分(%) & 占据数
            # 示例: -11.2054 & 1s (100%) & 1.00
            line = (f"{idx_str} & {spin_str} & "
                    f"{hf_e:.4f} & ${hf_comp}$ ({hf_pct:.2f}) & {hf_occ:.2f} & "
                    f"{lsda_e:.4f} & ${lsda_comp}$ ({lsda_pct:.2f}) & {lsda_occ:.2f} \\\\")
            f.write(line + "\n")
            
            # --- Beta Spin Row ---
            # 只有当 Beta 存在时才写
            if i < len(res_hf.orbital_energies_beta):
                spin_str = r"$\beta$"
                
                # HF
                hf_e = res_hf.orbital_energies_beta[i]
                hf_comp, hf_pct = get_dominant_composition(res_hf.coefficients_beta[:, i], orb_names)
                hf_occ = hf_occ_b[i]
                
                # LSDA
                lsda_e = res_lsda.orbital_energies_beta[i]
                lsda_comp, lsda_pct = get_dominant_composition(res_lsda.coefficients_beta[:, i], orb_names)
                lsda_occ = lsda_occ_b[i]

                line = (f" & {spin_str} & "
                        f"{hf_e:.4f} & ${hf_comp}$ ({hf_pct:.2f}) & {hf_occ:.2f} & "
                        f"{lsda_e:.4f} & ${lsda_comp}$ ({lsda_pct:.2f}) & {lsda_occ:.2f} \\\\")
                f.write(line + "\n")
            
            # 在每个轨道对之后加一点额外的间距，方便阅读
            f.write(r"\addlinespace[0.3em]" + "\n")

def save_table(filename,res,output_dir,max_orbitals=None):
    """
    保存包含轨道成分分析和占据数的详细表格
    格式: Index | Spin | E, Comp, Occ
    """
    ha_to_ev = 27.2114
    OUT_DIR=output_dir
    
    # 获取基组名称列表
    basis = res.integral_calc.basis
    orb_names = basis.get_orbital_names()
    
    # 获取占据数 (如果没有 explicit occupation 属性，则根据能量/ Aufbau 构造)
    # 假设 Results 对象里有 occupations_alpha/beta 属性
    # 如果没有，这里做一个简单的 Aufbau 推断（仅供参考，最好你的类里有存）
    def get_occs(res, spin):
        try:
            if spin == 'alpha':
                return res.occ_alpha
            elif spin == 'beta':
                return res.occ_beta
      
        except:
            # Fallback: 全1或0 (仅适用于简单 HF)
            n_occ = len(res.electron_configuration['occupied_' + spin])
            total_n = len(res.orbital_energies_alpha)
            occs = np.zeros(total_n)
            occs[:n_occ] = 1.0
            return occs

    occ_a = get_occs(res, 'alpha')
    occ_b = get_occs(res, 'beta')

    with open(os.path.join(OUT_DIR, filename), 'w') as f:
        # 确定要打印的轨道数量 (取两者中较大的基组大小)
        n_orbitals = len(res.orbital_energies_alpha)
        
        for i in range(n_orbitals):
            if max_orbitals is not None and i >= max_orbitals:
                break   
            # --- Alpha Spin Row ---
            idx_str = f"{i+1}"
            spin_str = r"$\alpha$"
            
            # Data
            e = res.orbital_energies_alpha[i]
            comp, pct = get_dominant_composition(res.coefficients_alpha[:, i], orb_names)
            occ = occ_a[i]
            
            # 格式化: 能量(Ha) & 成分(%) & 占据数
            # 示例: -11.2054 & 1s (100%) & 1.00
            line = (f"{idx_str} & {spin_str} & "
                    f"{e:.4f} & ${comp}$ ({pct:.2f}) & {occ:.2f} \\\\")
                   
            f.write(line + "\n")
            
            # --- Beta Spin Row ---
            # 只有当 Beta 存在时才写
            if i < len(res.orbital_energies_beta):
                spin_str = r"$\beta$"
                
                # HF
                e = res.orbital_energies_beta[i]
                comp, pct = get_dominant_composition(res.coefficients_beta[:, i], orb_names)
                occ = occ_b[i]

                line = (f" & {spin_str} & "
                        f"{e:.4f} & ${comp}$ ({pct:.2f}) & {occ:.2f} \\\\")
                f.write(line + "\n")
            
            # 在每个轨道对之后加一点额外的间距，方便阅读
            f.write(r"\addlinespace[0.3em]" + "\n")


def plot_compare_radial(ax, res_hf, res_lsda, spin, orb_indices, labels, title,divide_by_r=False):
    """在指定 ax 上绘制 HF vs LSDA 对比"""
    colors = ["r","g","b"] # 红、绿、蓝
    
    for i, idx in enumerate(orb_indices):
        name = labels[i]
        c = colors[i % len(colors)]
        
        # 提取数据
        r, y_hf = get_radial_data(res_hf, spin, idx,divide_by_r=divide_by_r)
        _, y_lsda = get_radial_data(res_lsda, spin, idx,divide_by_r=divide_by_r)
        
        # 绘图 (限制 r < 5.0)
        mask = r <= 5.0
        ax.plot(r[mask], y_hf[mask], '-', color=c, label=f'{name} HF')
        ax.plot(r[mask], y_lsda[mask], '--', color=c, alpha=0.8, label=f'{name} LSDA')

    ax.set_title(title)
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("r*R(r)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)