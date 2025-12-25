import sys
import os
import pickle
import numpy as np
from pathlib import Path
# 添加上级目录以便导入 core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import *

# 输出配置
OUT_DIR = Path(__file__).parent.parent/ "all_results/DFT"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


if __name__ == "__main__":
    real_basis = True
    print("开始生成所有计算数据...")
    all_results = {} # 用于存储结果对象

    ######################################################################
    # 1. 氢原子 STO-3G 计算
    ######################################################################
    print("\n[1/4] Running H STO-3G...")
    
    basis = get_sto3g_basis('H')
    ints = AtomicIntegrals(basis, nuclear_charge=1, real_basis=real_basis)

    # HF
    hf = AtomicHartreeFock(ints)
    h_sto3g_hf = hf.run_scf()
    
    # LSDA
    lsda = AtomicLSDA(ints)
    h_sto3g_lsda = lsda.run_scf()

    # 保存表格
    save_table2("H_sto3g.txt", h_sto3g_hf, h_sto3g_lsda, output_dir=OUT_DIR)
    
    # 存入字典
    all_results['h_sto3g_hf'] = h_sto3g_hf
    all_results['h_sto3g_lsda'] = h_sto3g_lsda


    ######################################################################
    # 2. 氢原子 NAO+赝势 计算
    ######################################################################
    print("\n[2/4] Running H NAO+Pseudo...")
    
    basis_path = "SG15-Version1p0__StandardOrbitals-Version2p0/H_gga_6au_100Ry_2s1p.orb"
    pseudo_path = "SG15-Version1p0_Pseudopotential/SG15_ONCV_v1.0_upf/H_ONCV_PBE-1.0.upf"
    
    basis = load_basis_set_from_file(basis_path)
    pseudo = load_pseudopotential_from_upf(pseudo_path)
    ints = AtomicIntegrals(basis, pseudo=pseudo, real_basis=real_basis)

    # 计算
    h_nao_hf = AtomicHartreeFock(ints).run_scf()
    h_nao_lsda = AtomicLSDA(ints).run_scf()
    
    # NAO 基组通常包含多个轨道，这里选前两个
    save_table2("H_nao.txt", h_nao_hf, h_nao_lsda,output_dir=OUT_DIR)
    
    all_results['h_nao_hf'] = h_nao_hf
    all_results['h_nao_lsda'] = h_nao_lsda


    ######################################################################
    # 3. 碳原子 STO-3G 计算
    ######################################################################
    print("\n[3/4] Running C STO-3G...")
    
    basis = get_sto3g_basis('C')
    ints = AtomicIntegrals(basis, nuclear_charge=6, real_basis=real_basis)

    # C Triplet (自旋多重度=3)
    c_sto3g_hf = AtomicHartreeFock(ints, multiplicity=3).run_scf()
    # LSDA 需要一点 damping 防止震荡
    c_sto3g_lsda = AtomicLSDA(ints, multiplicity=3, damping_factor=0.7).run_scf()
    
    # STO-3G C 轨道: 1s, 2s, 2px, 2py, 2pz
    labels = ["1s", "2s", "2p$_x$", "2p$_y$", "2p$_z$"]
    save_table2("C_sto3g.txt", c_sto3g_hf, c_sto3g_lsda, output_dir=OUT_DIR, max_orbitals=6)
    
    all_results['c_sto3g_hf'] = c_sto3g_hf
    all_results['c_sto3g_lsda'] = c_sto3g_lsda


    ######################################################################
    # 4. 碳原子 NAO+赝势 计算
    ######################################################################
    print("\n[4/4] Running C NAO+Pseudo...")
    
    b_path = "SG15-Version1p0__StandardOrbitals-Version2p0/C_gga_7au_100Ry_2s2p1d.orb"
    p_path = "SG15-Version1p0_Pseudopotential/SG15_ONCV_v1.0_upf/C_ONCV_PBE-1.0.upf"
    
    basis = load_basis_set_from_file(b_path)
    pseudo = load_pseudopotential_from_upf(p_path)
    ints = AtomicIntegrals(basis, pseudo=pseudo,real_basis=real_basis)

    c_nao_hf = AtomicHartreeFock(ints).run_scf()
    c_nao_lsda = AtomicLSDA(ints).run_scf()
    
    save_table2("C_nao.txt", c_nao_hf, c_nao_lsda, output_dir=OUT_DIR, max_orbitals=6)
    
    all_results['c_nao_hf'] = c_nao_hf
    all_results['c_nao_lsda'] = c_nao_lsda

        
        # ==========================================
    # 1. 绘制径向波函数对比图 (1D)
    # ==========================================
    print("Plotting Radial Wavefunctions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # (0,0) H STO-3G (1s)
    plot_compare_radial(axes[0,0], all_results['h_sto3g_hf'], all_results['h_sto3g_lsda'], 
                       'alpha', [0], ['1s'], "H STO-3G")
    
    # (0,1) H NAO (1s, 2s)
    plot_compare_radial(axes[0,1], all_results['h_nao_hf'], all_results['h_nao_lsda'], 
                       'alpha', [0, 1], ['1s', '2s'], "H NAO+Pseudo")

    # (1,0) C STO-3G Alpha (1s, 2s, 2p) -> indices [0, 1, 4] (假设2p是第5个)
    # 注意：STO-3G顺序通常是 1s, 2s, 2px, 2py, 2pz。取 2pz 代表 p 轨道
    plot_compare_radial(axes[1,0], all_results['c_sto3g_hf'], all_results['c_sto3g_lsda'], 
                       'alpha', [0, 1, 2], ['1s', '2s', '2p'], "C STO-3G (Alpha)")

    # (1,1) C NAO Alpha (2s, 2p) -> indices [0, 1] (赝势无1s)
    plot_compare_radial(axes[1,1], all_results['c_nao_hf'], all_results['c_nao_lsda'], 
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
    plot_orbital_contour(all_results['c_sto3g_hf'], orbital_index=2, spin='alpha', 
                        plane='xz', range_au=5.0)
    plt.savefig(f"{OUT_DIR}/C_sto3g_2p_contour.pdf")
    plt.close()

    # C NAO 2p轨道 (Index 1 usually 2pz, after 2s)
    plot_orbital_contour(all_results['c_nao_hf'], orbital_index=3, spin='alpha', 
                        plane='xz', range_au=6.0)
    plt.savefig(f"{OUT_DIR}/C_nao_2p_contour.pdf")
    plt.close()
    
    print("Done!")