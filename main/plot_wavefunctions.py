import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 添加路径以导入 core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 core 导入复杂的 2D 绘图函数
try:
    from core.visualization import plot_orbital_contour
except ImportError:
    print("Warning: plot_orbital_contour not found in core.visualization")

# 配置
OUT_DIR = "readme/figures"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-paper') # 使用一个干净的学术风格
plt.rcParams['font.family'] = 'sans-serif'

def get_radial_data(res, spin, orb_idx):
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
        psi_r = psi_chi / r
        # 简单修补原点
        if np.abs(r[0]) < 1e-9:
            psi_r[0] = psi_r[1]
            
    return r, psi_r

def plot_compare_radial(ax, res_hf, res_lsda, spin, orb_indices, labels, title):
    """在指定 ax 上绘制 HF vs LSDA 对比"""
    colors = ["r","g","b"] # 红、绿、蓝
    
    for i, idx in enumerate(orb_indices):
        name = labels[i]
        c = colors[i % len(colors)]
        
        # 提取数据
        r, y_hf = get_radial_data(res_hf, spin, idx)
        _, y_lsda = get_radial_data(res_lsda, spin, idx)
        
        # 绘图 (限制 r < 5.0)
        mask = r <= 5.0
        ax.plot(r[mask], y_hf[mask], '-', color=c, label=f'{name} HF')
        ax.plot(r[mask], y_lsda[mask], '--', color=c, alpha=0.8, label=f'{name} LSDA')

    ax.set_title(title)
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("R(r)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    pkl_path = f"{OUT_DIR}/all_results.pkl"
    if not os.path.exists(pkl_path):
        print(f"数据文件缺失: {pkl_path}")
        sys.exit(1)

    print(f"Reading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # ==========================================
    # 1. 绘制径向波函数对比图 (1D)
    # ==========================================
    print("Plotting Radial Wavefunctions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # (0,0) H STO-3G (1s)
    plot_compare_radial(axes[0,0], data['h_sto3g_hf'], data['h_sto3g_lsda'], 
                       'alpha', [0], ['1s'], "H STO-3G")
    
    # (0,1) H NAO (1s, 2s)
    plot_compare_radial(axes[0,1], data['h_nao_hf'], data['h_nao_lsda'], 
                       'alpha', [0, 1], ['1s', '2s'], "H NAO+Pseudo")

    # (1,0) C STO-3G Alpha (1s, 2s, 2p) -> indices [0, 1, 4] (假设2p是第5个)
    # 注意：STO-3G顺序通常是 1s, 2s, 2px, 2py, 2pz。取 2pz 代表 p 轨道
    plot_compare_radial(axes[1,0], data['c_sto3g_hf'], data['c_sto3g_lsda'], 
                       'alpha', [0, 1, 4], ['1s', '2s', '2p'], "C STO-3G (Alpha)")

    # (1,1) C NAO Alpha (2s, 2p) -> indices [0, 1] (赝势无1s)
    plot_compare_radial(axes[1,1], data['c_nao_hf'], data['c_nao_lsda'], 
                       'alpha', [0, 1], ['2s', '2p'], "C NAO+Pseudo (Alpha)")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/radial_comparison.pdf")
    print("Saved radial_comparison.pdf")
    plt.close()

    # ==========================================
    # 2. 绘制电子密度等值面 (2D)
    # ==========================================
    print("Plotting 2D Contours...")

    # C STO-3G 2p轨道 (Index 4 usually 2pz)
    # 直接调用 core 里的函数
    plot_orbital_contour(data['c_sto3g_hf'], orbital_index=2, spin='alpha', 
                        plane='xz', range_au=5.0)
    plt.savefig(f"{OUT_DIR}/C_sto3g_2p_contour.pdf")
    plt.close()

    # C NAO 2p轨道 (Index 1 usually 2pz, after 2s)
    plot_orbital_contour(data['c_nao_hf'], orbital_index=3, spin='alpha', 
                        plane='xz', range_au=6.0)
    plt.savefig(f"{OUT_DIR}/C_nao_2p_contour.pdf")
    plt.close()
    
    print("Done!")