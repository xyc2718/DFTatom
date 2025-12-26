"""
tddft_lr.py - 线性响应 TDDFT (Linear Response TDDFT / Casida) 模块
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import List
import logging

# 导入 AtomicLSDA 以便类型检查 (可选)
from .LSDA import AtomicLSDA, get_slater_vwn5,KSResults

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
        
        # 假设 AtomicLSDA 运行完 run_scf 后会把结果保存在 self.ks_results 中
        # 如果你的 AtomicLSDA 没有保存这个属性，你需要修改 AtomicLSDA 或者在这里额外传入 ks_results
        self.ks = ks_results

        # 直接复用 LSDA 的组件，无需重新初始化
        self.integral = self.lsda.integral_calc
        self.lsda_helper = self.lsda  
        
        self.n_basis = self.integral.n_basis
        
        # 1. 整理轨道信息
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
        # ... (solve 方法内部代码完全不变，因为它使用的是 self.ks 和 self.lsda_helper) ...
        # ... 只需确保 _compute_xc_kernel_matrix 里调用的是 self.lsda_helper ...
        
        # 复制之前的 solve 代码 ...
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
        
        results = []
        dip_mo_a = self._compute_transition_dipoles(C_a, self.occ_idxs_a, self.virt_idxs_a)
        dip_mo_b = self._compute_transition_dipoles(C_b, self.occ_idxs_b, self.virt_idxs_b)
        flat_dip_a = dip_mo_a.reshape(3, self.dim_a)
        flat_dip_b = dip_mo_b.reshape(3, self.dim_b)
        
        for n in range(min(n_states, len(evals))):
            omega = evals[n]
            if omega < 1e-3: continue
            
            X = evecs[:, n]
            D_vec = (flat_dip_a @ X[:self.dim_a] + flat_dip_b @ X[self.dim_a:])
            osc_str = (2.0 / 3.0) * omega * np.sum(D_vec**2)
            contributions = self._analyze_composition(X)
            
            results.append(Excitation(
                energy_hartree=omega,
                energy_ev=omega * 27.2114,
                oscillator_strength=osc_str,
                spin="Mixed",
                contributions=contributions
            ))
            
        return results

    # ... _transform_eri, _transform_eri_mixed, _get_trans_dens, _compute_transition_dipoles, _analyze_composition 保持不变 ...
    # ... 只需要复制过来即可 ...

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
        """通过数值差分计算 f_xc 矩阵元"""
        logging.info("计算 XC Kernel (数值差分)...")
        
        # 1. 计算密度
        rho_a_3d, _ = self.lsda_helper._calculate_electron_density_on_grid(self.ks.density_matrix_alpha)
        rho_b_3d, _ = self.lsda_helper._calculate_electron_density_on_grid(self.ks.density_matrix_beta)
        
        # 2. dV/drho (f_xc)
        delta = 1e-4
        v_xa_0, v_xb_0, _ = get_slater_vwn5(rho_a_3d, rho_b_3d)
        
        v_xa_p, v_xb_p_cross, _ = get_slater_vwn5(rho_a_3d + delta, rho_b_3d)
        fxc_aa = (v_xa_p - v_xa_0) / delta
        fxc_ba = (v_xb_p_cross - v_xb_0) / delta 
        
        v_xa_p_cross, v_xb_p, _ = get_slater_vwn5(rho_a_3d, rho_b_3d + delta)
        fxc_bb = (v_xb_p - v_xb_0) / delta
        
        # 3. 计算跃迁密度
        td_a = self._get_trans_dens(self.ks.coefficients_alpha, self.occ_idxs_a, self.virt_idxs_a)
        td_b = self._get_trans_dens(self.ks.coefficients_beta, self.occ_idxs_b, self.virt_idxs_b)
        
        # 4. 积分
        vol_element = 4.0 * np.pi * self.integral.r_grid**2
        
        def integrate_kernel(td1, fxc, td2):
            weighted_td2 = td2 * fxc[None, :] * vol_element[None, :]
            return np.dot(td1, weighted_td2.T)

        # --- 修复点：使用正确的 n_occ 和 n_virt 进行 Reshape ---
        # K_aa 形状: (dim_a, dim_a) -> (ia, jb) -> (i, a, j, b)
        K_aa_2d = integrate_kernel(td_a, fxc_aa, td_a)
        K_aa = K_aa_2d.reshape(self.n_occ_a, self.n_virt_a, self.n_occ_a, self.n_virt_a)
        
        # K_bb 形状: (dim_b, dim_b) -> (i, a, j, b) [Beta]
        K_bb_2d = integrate_kernel(td_b, fxc_bb, td_b)
        K_bb = K_bb_2d.reshape(self.n_occ_b, self.n_virt_b, self.n_occ_b, self.n_virt_b)
        
        # K_ab 形状: (dim_a, dim_b) -> (i_a, a_a, j_b, b_b)
        K_ab_2d = integrate_kernel(td_a, fxc_ba, td_b)
        K_ab = K_ab_2d.reshape(self.n_occ_a, self.n_virt_a, self.n_occ_b, self.n_virt_b)
        
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
        if 'dipole' in self.integral.compute_all_integrals():
            D_ao = self.integral.compute_all_integrals()['dipole']
        else:
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
        threshold = 0.1
        comps = []
        try:
            names = self.integral.basis.get_orbital_names()
        except:
            names = [f"Orb{i}" for i in range(self.n_basis)]

        for idx in range(self.dim_a):
            coeff = eigenvector[idx]
            if abs(coeff) ** 2 > threshold:
                i_local = idx // self.n_virt_a
                a_local = idx % self.n_virt_a
                i_idx = self.occ_idxs_a[i_local]
                a_idx = self.virt_idxs_a[a_local]
                comps.append(f"{names[i_idx]}(a)->{names[a_idx]}(a) ({coeff:.2f})")
                
        for idx in range(self.dim_b):
            global_idx = idx + self.dim_a
            coeff = eigenvector[global_idx]
            if abs(coeff) ** 2 > threshold:
                i_local = idx // self.n_virt_b
                a_local = idx % self.n_virt_b
                i_idx = self.occ_idxs_b[i_local]
                a_idx = self.virt_idxs_b[a_local]
                comps.append(f"{names[i_idx]}(b)->{names[a_idx]}(b) ({coeff:.2f})")
                
        return comps