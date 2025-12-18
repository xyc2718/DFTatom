import sys
import os
import pickle
import numpy as np

# 添加上级目录以便导入 core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import *

# 输出配置
OUT_DIR = "readme/figures"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
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

def save_table(filename, res_hf, res_lsda,max_orbitals=None):
    """
    保存包含轨道成分分析和占据数的详细表格
    格式: Index | Spin | HF(E, Comp, Occ) | LSDA(E, Comp, Occ)
    """
    ha_to_ev = 27.2114
    
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
    save_table("H_sto3g.txt", h_sto3g_hf, h_sto3g_lsda)
    
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
    save_table("H_nao.txt", h_nao_hf, h_nao_lsda)
    
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
    save_table("C_sto3g.txt", c_sto3g_hf, c_sto3g_lsda,max_orbitals=6)
    
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
    
    save_table("C_nao.txt", c_nao_hf, c_nao_lsda,max_orbitals=6)
    
    all_results['c_nao_hf'] = c_nao_hf
    all_results['c_nao_lsda'] = c_nao_lsda


    ######################################################################
    # 保存 Picke 数据供绘图脚本使用
    ######################################################################
    pkl_path = os.path.join(OUT_DIR, "all_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
        
    print(f"\n全部完成！数据已保存至 {pkl_path}")
    print(f"LaTeX 表格数据位于 {OUT_DIR}/")