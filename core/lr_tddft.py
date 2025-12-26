"""
tddft_lr.py - 线性响应 TDDFT (Linear Response TDDFT / Casida) 模块
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import List
import logging
import matplotlib.pyplot as plt # 导入绘图库

# 导入 AtomicLSDA 以便类型检查 (可选)
from .LSDA import AtomicLSDA, get_slater_vwn5, KSResults,get_lb94_vwn5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Excitation:
    energy_hartree: float
    energy_ev: float
    oscillator_strength: float
    spin: str 
    contributions: List[str]

class LinearResponseTDDFT:
    """
    基于 Casida 方程的线性响应 TDDFT 计算器
    """

    def __init__(self, ks_results: KSResults, lsda_solver: AtomicLSDA):
        """
        初始化 LR-TDDFT

        Args:
            lsda_solver: 已经完成 SCF 计算的 AtomicLSDA 实例
                         (需要包含 .ks_results 属性，或者你可以修改这里从外部传入结果)
        """
        self.lsda = lsda_solver
        self.ks = ks_results

        self.integral = self.lsda.integral_calc
        self.lsda_helper = self.lsda  
        
        self.n_basis = self.integral.n_basis
        
        # 整理轨道信息
        self.occ_idxs_a, self.virt_idxs_a = self._classify_orbitals(
            self.ks.occ_alpha, self.ks.orbital_energies_alpha
        )
        self.occ_idxs_b, self.virt_idxs_b = self._classify_orbitals(
            self.ks.occ_beta, self.ks.orbital_energies_beta
        )
        
        self.n_occ_a = len(self.occ_idxs_a)
        self.n_virt_a = len(self.virt_idxs_a)
        self.n_occ_b = len(self.occ_idxs_b)
        self.n_virt_b = len(self.virt_idxs_b)
        
        self.dim_a = self.n_occ_a * self.n_virt_a
        self.dim_b = self.n_occ_b * self.n_virt_b
        self.total_dim = self.dim_a + self.dim_b
        
        # 初始化存储变量 
        self.excitation_energies = None
        self.oscillator_strengths = None
        
        logging.info(f"初始化 LR-TDDFT (TDA) based on LSDA Solver:")
        logging.info(f"  Alpha: {self.n_occ_a} occ -> {self.n_virt_a} virt")
        logging.info(f"  Beta : {self.n_occ_b} occ -> {self.n_virt_b} virt")

    def _classify_orbitals(self, occs, energies):
        """根据占据数和能量阈值分类轨道"""
        occ_idxs = []
        virt_idxs = []
        for i, (occ, en) in enumerate(zip(occs, energies)):
            if en > 100.0: 
                continue
            if occ > 0.1:
                occ_idxs.append(i)
            else:
                virt_idxs.append(i)
        return occ_idxs, virt_idxs

    def solve(self, n_states: int = 10) -> List[Excitation]:
        """求解 Casida 方程，返回激发态列表"""

        logging.info("构建 Casida 矩阵...")
        A = np.zeros((self.total_dim, self.total_dim))
        
        C_a = self.ks.coefficients_alpha
        C_b = self.ks.coefficients_beta
        
        eri_mo_aa = self._transform_eri(C_a, C_a, self.occ_idxs_a, self.virt_idxs_a)
        eri_mo_bb = self._transform_eri(C_b, C_b, self.occ_idxs_b, self.virt_idxs_b)
        eri_mo_ab = self._transform_eri_mixed(C_a, C_b, 
                                              self.occ_idxs_a, self.virt_idxs_a,
                                              self.occ_idxs_b, self.virt_idxs_b)
        
        K_xc_aa, K_xc_bb, K_xc_ab = self._compute_xc_kernel_matrix()
        
        # 填充 Alpha-Alpha
        ediff_a = (self.ks.orbital_energies_alpha[self.virt_idxs_a][None, :] - 
                   self.ks.orbital_energies_alpha[self.occ_idxs_a][:, None])
        diag_a = ediff_a.flatten()
        coupling_aa = (eri_mo_aa + K_xc_aa).reshape(self.dim_a, self.dim_a)
        A[:self.dim_a, :self.dim_a] = coupling_aa + np.diag(diag_a)
        
        # 填充 Beta-Beta
        ediff_b = (self.ks.orbital_energies_beta[self.virt_idxs_b][None, :] - 
                   self.ks.orbital_energies_beta[self.occ_idxs_b][:, None])
        diag_b = ediff_b.flatten()
        coupling_bb = (eri_mo_bb + K_xc_bb).reshape(self.dim_b, self.dim_b)
        A[self.dim_a:, self.dim_a:] = coupling_bb + np.diag(diag_b)
        
        # 填充 Alpha-Beta
        coupling_ab = (eri_mo_ab + K_xc_ab).reshape(self.dim_a, self.dim_b)
        A[:self.dim_a, self.dim_a:] = coupling_ab
        A[self.dim_a:, :self.dim_a] = coupling_ab.T
        
        logging.info("求解特征值问题...")
        evals, evecs = eigh(A)
        self.A=A
        logging.info(f"Raw Eigenvalues (Hartree): {evals[:5]}")
        
        results = []
        dip_mo_a = self._compute_transition_dipoles(C_a, self.occ_idxs_a, self.virt_idxs_a)
        dip_mo_b = self._compute_transition_dipoles(C_b, self.occ_idxs_b, self.virt_idxs_b)
        flat_dip_a = dip_mo_a.reshape(3, self.dim_a)
        flat_dip_b = dip_mo_b.reshape(3, self.dim_b)
        
        # 用于临时存储结果的列表
        temp_energies = []
        temp_osc_strs = []

        for n in range(min(n_states, len(evals))):
            omega = evals[n]
            if omega < 1e-3: continue
            
            X = evecs[:, n]
            D_vec = (flat_dip_a @ X[:self.dim_a] + flat_dip_b @ X[self.dim_a:])
            osc_str = (2.0 / 3.0) * omega * np.sum(D_vec**2)
            contributions = self._analyze_composition(X)
            
            # 记录数据以便画图
            temp_energies.append(omega * 27.2114) # 存 eV
            temp_osc_strs.append(osc_str)

            results.append(Excitation(
                energy_hartree=omega,
                energy_ev=omega * 27.2114,
                oscillator_strength=osc_str,
                spin="Mixed",
                contributions=contributions
            ))
            
        #保存结果到 self，供后续画图使用
        self.excitation_energies = np.array(temp_energies)
        self.oscillator_strengths = np.array(temp_osc_strs)

        return results


    def _transform_eri(self, C1, C2, occ_idxs, virt_idxs):
        """MO 积分变换"""
        C_occ = C1[:, occ_idxs]
        C_virt = C1[:, virt_idxs]
        C_occ2 = C2[:, occ_idxs]
        C_virt2 = C2[:, virt_idxs]
        eri_ao = self.ks.eri
        
        temp1 = np.einsum('mnls,sb->mnlb', eri_ao, C_virt2)
        temp2 = np.einsum('mnlb,lj->mnjb', temp1, C_occ2)
        temp3 = np.einsum('mnjb,na->majb', temp2, C_virt)
        mo_eri = np.einsum('majb,mi->iajb', temp3, C_occ)
        return mo_eri

    def _transform_eri_mixed(self, Ca, Cb, occ_a, virt_a, occ_b, virt_b):
        """MO 积分变换 (Alpha-Beta)"""
        C_occ_a = Ca[:, occ_a]
        C_virt_a = Ca[:, virt_a]
        C_occ_b = Cb[:, occ_b]
        C_virt_b = Cb[:, virt_b]
        eri_ao = self.ks.eri
        
        temp1 = np.einsum('mnls,mi->inls', eri_ao, C_occ_a)
        temp2 = np.einsum('inls,na->ials', temp1, C_virt_a)
        temp3 = np.einsum('ials,lj->iajs', temp2, C_occ_b)
        mo_eri = np.einsum('iajs,sb->iajb', temp3, C_virt_b)
        return mo_eri
    
    def _compute_xc_kernel_matrix(self):
        """
        通过数值差分计算 f_xc 矩阵元。
        支持 GGA 泛函 (LB94) 的梯度依赖。
        
        包含修正:
        1. 密度截断 (Cutoff): 避免在极低密度区产生数值发散。
        2. 自适应步长 (Adaptive Delta): 确保差分求导的数值稳定性。
        3. 核限幅 (Kernel Clamping): 防止非物理的强排斥/吸引。
        4. 正确的网格积分权重 (dV = 4pi * r^2 * dr)。
        """
        import logging
        import numpy as np
        
        # 确定当前的泛函类型
        func_type = getattr(self.lsda_helper, 'functional_type', 'lda').lower()
        logging.info(f"计算 XC Kernel (基于 {func_type.upper()} 泛函的数值差分)...")
        
        # 1. 在网格上获取密度及其梯度 (LB94 必须需要梯度项)
        # 假设 lsda_helper._calculate_electron_density_on_grid 返回 (density, gradient)
        rho_a, grad_a = self.lsda_helper._calculate_electron_density_on_grid(self.ks.density_matrix_alpha)
        rho_b, grad_b = self.lsda_helper._calculate_electron_density_on_grid(self.ks.density_matrix_beta)
        
        rho_tot = rho_a + rho_b
        
        # 截断阈值：过低的密度区不参与差分计算
        cutoff_threshold = 1e-12
        mask = rho_tot > cutoff_threshold
        
        # 初始化 fxc 矩阵元数组
        fxc_aa = np.zeros_like(rho_a)
        fxc_bb = np.zeros_like(rho_b)
        fxc_ba = np.zeros_like(rho_a)
        
        if np.any(mask):
            ra = rho_a[mask]
            rb = rho_b[mask]
            ga = grad_a[mask]
            gb = grad_b[mask]
            
            # 自适应步长：步长随密度大小调整，确保数值求导的精度
            delta_a = np.maximum(0.01 * ra, 1e-10)
            delta_b = np.maximum(0.01 * rb, 1e-10)
            
            if func_type == 'lb94':
                # --- LB94 路径：必须传入 4 个参数 (rho_a, rho_b, grad_a, grad_b) ---
                # 获取基础势能
                v_xa_0, v_xb_0, _ = get_lb94_vwn5(ra, rb, ga, gb)
                
                # 扰动 Alpha 密度，保持当前梯度不变 (绝热近似)
                v_xa_p, v_xb_p_cross, _ = get_lb94_vwn5(ra + delta_a, rb, ga, gb)
                val_aa = (v_xa_p - v_xa_0) / delta_a
                val_ba = (v_xb_p_cross - v_xb_0) / delta_a 
                
                # 扰动 Beta 密度
                v_xa_p_cross, v_xb_p, _ = get_lb94_vwn5(ra, rb + delta_b, ga, gb)
                val_bb = (v_xb_p - v_xb_0) / delta_b
            else:
                # --- LDA 路径：仅需 2 个参数 (rho_a, rho_b) ---
                v_xa_0, v_xb_0, _ = get_slater_vwn5(ra, rb)
                
                v_xa_p, v_xb_p_cross, _ = get_slater_vwn5(ra + delta_a, rb)
                val_aa = (v_xa_p - v_xa_0) / delta_a
                val_ba = (v_xb_p_cross - v_xb_0) / delta_a 
                
                v_xa_p_cross, v_xb_p, _ = get_slater_vwn5(ra, rb + delta_b)
                val_bb = (v_xb_p - v_xb_0) / delta_b
            
            # 核限幅 (Kernel Clamping)：防止数值奇异性
            clamp_min = -10.0
            fxc_aa[mask] = np.clip(val_aa, clamp_min, None)
            fxc_bb[mask] = np.clip(val_bb, clamp_min, None)
            fxc_ba[mask] = np.clip(val_ba, clamp_min, None)

        # 2. 获取跃迁密度 rho_ia(r)
        td_a = self._get_trans_dens(self.ks.coefficients_alpha, self.occ_idxs_a, self.virt_idxs_a)
        td_b = self._get_trans_dens(self.ks.coefficients_beta, self.occ_idxs_b, self.virt_idxs_b)
        
        # 3. 计算对数网格下的体积元 dV = 4 * pi * r^2 * dr
        r = self.integral.r_grid
        dr = np.gradient(r)
        vol_element = 4.0 * np.pi * (r**2) * dr
        
        def integrate_kernel(td1, fxc, td2):
            """执行数值积分: ∫ rho_ia(r) * fxc(r) * rho_jb(r) dV"""
            if td1.shape[0] == 0 or td2.shape[0] == 0:
                return np.zeros((td1.shape[0], td2.shape[0]))
            
            # 广播计算: td2(N_ia, N_grid) * fxc(N_grid) * dV(N_grid)
            weighted_td2 = td2 * fxc[None, :] * vol_element[None, :]
            # 矩阵乘法完成空间求和
            return np.dot(td1, weighted_td2.T)

        # 4. 构建耦合矩阵项
        K_aa = integrate_kernel(td_a, fxc_aa, td_a).reshape(
            self.n_occ_a, self.n_virt_a, self.n_occ_a, self.n_virt_a
        )
        
        if self.n_occ_b > 0 and self.n_virt_b > 0:
            K_bb = integrate_kernel(td_b, fxc_bb, td_b).reshape(
                self.n_occ_b, self.n_virt_b, self.n_occ_b, self.n_virt_b
            )
        else:
            K_bb = np.zeros((self.n_occ_b, self.n_virt_b, self.n_occ_b, self.n_virt_b))
            
        if self.n_occ_a > 0 and self.n_virt_b > 0:
            K_ab = integrate_kernel(td_a, fxc_ba, td_b).reshape(
                self.n_occ_a, self.n_virt_a, self.n_occ_b, self.n_virt_b
            )
        else:
            K_ab = np.zeros((self.n_occ_a, self.n_virt_a, self.n_occ_b, self.n_virt_b))
        
        return K_aa, K_bb, K_ab
    def _get_trans_dens(self, C, occs, virts):
        n_pairs = len(occs) * len(virts)
        dens_grid = np.zeros((n_pairs, len(self.integral.r_grid)))
        rad_funcs = self.integral.radial_functions 
        idx = 0
        r_safe = np.where(self.integral.r_grid > 1e-10, self.integral.r_grid, 1e-10)
        
        for i in occs:
            for a in virts:
                mo_i_rad = np.einsum('m,mr->r', C[:, i], rad_funcs)
                mo_a_rad = np.einsum('n,nr->r', C[:, a], rad_funcs)
                rho_ia_val = (mo_i_rad * mo_a_rad) / (r_safe**2 * 4.0 * np.pi)
                dens_grid[idx] = rho_ia_val
                idx += 1
        return dens_grid

    def _compute_transition_dipoles(self, C, occs, virts):
    
        D_ao = self.integral.compute_dipole_matrix()
        dim = len(occs) * len(virts)
        dipoles = np.zeros((3, dim))
        idx = 0
        for i in occs:
            for a in virts:
                vec_i = C[:, i]
                vec_a = C[:, a]
                for k in range(3):
                    dipoles[k, idx] = vec_i @ D_ao[k] @ vec_a
                idx += 1
        return dipoles

    def _analyze_composition(self, eigenvector):
        """
        分析激发态特征向量，解析 MO 的主要成分。
        利用 AtomicOrbital.orbital_name 直接获取轨道名称。
        """
        threshold = 0.01
        comps = []
        
        # 获取基组中的原子轨道列表
        orbitals = self.integral.basis.orbitals
        
        # --- 内部函数：查找 MO 的主要成分名称 ---
        def get_mo_label(mo_idx, coeffs):
            """
            查找 MO 中系数最大的基函数，并返回其名称
            Args:
                mo_idx: 分子轨道(MO)的全局索引
                coeffs: 系数矩阵 (n_basis, n_orbitals)
            """
            # 获取该 MO 在所有基函数上的系数向量
            c_vec = coeffs[:, mo_idx]
            
            # 找到贡献最大的基函数索引 (绝对值最大)
            dom_idx = np.argmax(np.abs(c_vec))
            
            # 直接使用 AtomicOrbital 类中预计算好的 orbital_name
            return orbitals[dom_idx].orbital_name

        # 1. 遍历 Alpha 通道跃迁
        for idx in range(self.dim_a):
            coeff = eigenvector[idx]
            
            # 只显示贡献显著的成分
            if abs(coeff) ** 2 > threshold:
                # 解析 Casida 向量索引对应的 i (占据) 和 a (虚)
                i_local = idx // self.n_virt_a
                a_local = idx % self.n_virt_a
                
                # 转换回全局 MO 索引
                i_global = self.occ_idxs_a[i_local]
                a_global = self.virt_idxs_a[a_local]
                
                # 获取 MO 的主要成分名称
                name_i = get_mo_label(i_global, self.ks.coefficients_alpha)
                name_a = get_mo_label(a_global, self.ks.coefficients_alpha)
                
                comps.append(f"{name_i}(a)->{name_a}(a) ({coeff:.2f})")
                
        # 2. 遍历 Beta 通道跃迁 (如果有)
        for idx in range(self.dim_b):
            global_idx = idx + self.dim_a
            coeff = eigenvector[global_idx]
            
            if abs(coeff) ** 2 > threshold:
                i_local = idx // self.n_virt_b
                a_local = idx % self.n_virt_b
                
                i_global = self.occ_idxs_b[i_local]
                a_global = self.virt_idxs_b[a_local]
                
                name_i = get_mo_label(i_global, self.ks.coefficients_beta)
                name_a = get_mo_label(a_global, self.ks.coefficients_beta)
                
                comps.append(f"{name_i}(b)->{name_a}(b) ({coeff:.2f})")
                
        return comps

    def get_absorption_spectrum(self, 
                                start_ev: float = None, 
                                end_ev: float = None, 
                                step: float = 0.01, 
                                sigma: float = 0.4, 
                                kind: str = 'gaussian'):
        """
        根据计算出的激发态，生成吸收光谱曲线数据。
        
        Args:
            start_ev: 光谱起始能量 (eV)。默认自动根据最低激发能设定。
            end_ev:   光谱结束能量 (eV)。默认自动根据最高激发能设定。
            step:     能量网格步长 (eV)。
            sigma:    展宽参数 (eV)。
                      - 对于 'gaussian': 代表标准差 (FWHM ≈ 2.355 * sigma)
                      - 对于 'lorentzian': 代表半高宽的一半 (HWHM)
            kind:     展宽函数类型 ('gaussian' 或 'lorentzian')。
            
        Returns:
            energies (np.ndarray): X轴，能量点
            intensities (np.ndarray): Y轴，吸收强度 (任意单位)
        """
        if self.excitation_energies is None:
            raise ValueError("尚未计算激发态，请先运行 solve() 或 compute_excited_states()")

        # 1. 确定光谱范围
        if start_ev is None:
            start_ev = max(0.0, min(self.excitation_energies) - 2.0)
        if end_ev is None:
            end_ev = max(self.excitation_energies) + 2.0
            
        n_points = int((end_ev - start_ev) / step) + 1
        x_grid = np.linspace(start_ev, end_ev, n_points)
        y_grid = np.zeros_like(x_grid)
        
        # 2. 叠加每一个激发态的展宽函数
        for E_n, f_n in zip(self.excitation_energies, self.oscillator_strengths):
            # 忽略振子强度极小的态 (加速计算)
            if f_n < 1e-6:
                continue
                
            if kind == 'gaussian':
                # 高斯函数
                prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
                y_grid += f_n * prefactor * np.exp(-((x_grid - E_n)**2) / (2 * sigma**2))
                
            elif kind == 'lorentzian':
                # 洛伦兹函数
                prefactor = 1.0 / np.pi
                y_grid += f_n * prefactor * (sigma / ((x_grid - E_n)**2 + sigma**2))
            else:
                raise ValueError(f"不支持的展宽类型: {kind}")
                
        return x_grid, y_grid

    def plot_absorption_spectrum(self, 
                                 filename: str = None, 
                                 title: str = "Simulated UV-Vis Spectrum", 
                                 sigma: float = 0.3,
                                 kind: str = 'gaussian',
                                 start_ev: float = None,
                                 end_ev: float = None,):
        """
        直接绘制并显示/保存吸收光谱图。
        """
        
        # 获取数据
        x, y = self.get_absorption_spectrum(sigma=sigma, kind=kind,start_ev=start_ev,end_ev=end_ev)
        
        # 绘图
        plt.figure(figsize=(8, 5), dpi=120)
        
        # 绘制平滑曲线
        plt.plot(x, y, label=f'Broadening: {kind.capitalize()} ($\sigma$={sigma} eV)', color='b', linewidth=2)
        
        # 绘制离散的棒状图 (Stick Spectrum) 以指示精确位置
        max_y = np.max(y) if np.max(y) > 1e-6 else 1.0
        max_f = np.max(self.oscillator_strengths) if np.max(self.oscillator_strengths) > 1e-6 else 1.0
        scale_factor = max_y / max_f
        
        plt.vlines(self.excitation_energies, 0, self.oscillator_strengths * scale_factor, 
                   colors='r', linestyle='--', alpha=0.5, label='Oscillator Strength (Scaled)')
        
        plt.xlabel("Energy (eV)", fontsize=12)
        plt.ylabel("Intensity (arb. units)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"光谱图已保存至: {filename}")
        else:
            plt.show()