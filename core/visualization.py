import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.special import sph_harm

# 尝试导入结果类，如果不存在则使用占位符（方便单独测试）
try:
    from core.LSDA import KSResults
    from core.HF import HFResults
    CalculationResults = Union[KSResults, HFResults]
except ImportError:
    CalculationResults = "Any"  # 仅作类型提示占位

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
    cbar.set_label('Wavefunction Amplitude (a.u.)', rotation=270, labelpad=15)
    
    plt.scatter([0], [0], marker='+', color='black', s=80, alpha=0.7, label='Nucleus')
    plt.contour(X, Y, Psi_plot, levels=[0], colors='black', linewidths=0.5, alpha=0.3)
    
    # 标题生成
    max_c_idx = np.argmax(np.abs(coeffs))
    orb_label_guess = orbital_names[max_c_idx]
    title_spin = r"$\alpha$" if spin=='alpha' else r"$\beta$"
    
    plt.title(f"Orbital {orbital_index}: {orb_label_guess} ({title_spin})\n"
              f"E = {energy:.4f} Ha | Plane: {plane}")
    plt.xlabel(f"{plane[0].upper()} (Bohr)")
    plt.ylabel(f"{plane[1].upper()} (Bohr)")
    plt.axis('equal') # 保证圆形不变形
    plt.tight_layout()