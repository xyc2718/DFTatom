"""
tddft_rt.py - 实时时间相关密度泛函理论 (RT-TDDFT) 模块
特性：
1. 支持 strict_diagonalize 处理 aug-cc-pvtz 等病态基组
2. 始终保持系数矩阵 C 为原始 AO 基组形状 (N_basis x N_occ)，方便可视化
"""

import numpy as np
from scipy.linalg import solve, eigh
from scipy.fftpack import fft, fftfreq
import logging
from typing import Dict, Tuple

from .LSDA import KSResults, AtomicLSDA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealTimeTDDFT:
    """
    实时 TDDFT 计算类
    使用 Crank-Nicolson 传播子 (Predictor-Corrector)
    """

    def __init__(self, ks_results: KSResults, lsda_calc: AtomicLSDA, dt: float = 0.05, 
                 strict_diagonalize: bool = False, threshold: float = 1e-4):
        self.ks = ks_results
        self.integral_calc = ks_results.integral_calc
        self.n_basis = self.integral_calc.n_basis
        self.dt = dt
        self.strict_diagonalize = strict_diagonalize
        self.threshold = threshold
        
        # 基础矩阵 (AO 基组)
        self.S_ao = ks_results.s
        self.H_core_ao = ks_results.h_core
        self.eri = ks_results.eri
        
        # 构建正交化变换矩阵 X
        if self.strict_diagonalize:
            # 严格模式：投影到线性无关子空间
            s_vals, U = eigh(self.S_ao)
            mask = s_vals > self.threshold
            s_reg = s_vals[mask]
            U_reg = U[:, mask]
            
            # X 形状: (n_basis, n_valid)
            self.X = U_reg / np.sqrt(s_reg)
            self.S_sub = np.eye(self.X.shape[1]) # 子空间重叠矩阵是单位阵
            logging.info(f"RT-TDDFT 启用严格对角化: 有效维度 {self.X.shape[1]} / {self.n_basis}")
        else:
            # 普通模式
            self.X = None
            self.S_sub = self.S_ao
            logging.info(f"RT-TDDFT 标准模式: 维度 {self.n_basis}")

        self.D_ao = self.integral_calc.compute_dipole_matrix()

        # 初始化系数矩阵
        n_occ_alpha = int(self.ks.occ_alpha.sum())
        n_occ_beta = int(self.ks.occ_beta.sum())
        
        # 复制静态计算的系数 (AO基组)
        self.C_alpha = self.ks.coefficients_alpha[:, :n_occ_alpha].astype(np.complex128)
        self.C_beta = self.ks.coefficients_beta[:, :n_occ_beta].astype(np.complex128)

        # 初始化 helper
        self.lsda_helper = lsda_calc

        # 历史记录
        self.time_history = []
        self.dipole_history = []
    def _compute_dipole_subspace(self, c_a_sub, c_b_sub, D_sub) -> np.ndarray:
        """
        在子空间计算偶极矩: Tr(P_sub @ D_sub)
        比 AO 空间计算快得多
        """
        # 构建子空间密度
        P_a = (c_a_sub @ c_a_sub.conj().T).real
        P_b = (c_b_sub @ c_b_sub.conj().T).real
        P_tot = P_a + P_b
        
        mu = np.zeros(3)
        for i in range(3):
            # 这里的矩阵乘法维度是 (n_valid, n_valid)，远小于 (n_basis, n_basis)
            mu[i] = np.sum(P_tot * D_sub[i].T)
        return mu

    def _to_subspace(self, C_ao):
        """将 AO 系数投影到计算子空间"""
        if self.strict_diagonalize:
            # 投影公式: C_sub = X.T @ S @ C_ao
            return self.X.T @ self.S_ao @ C_ao
        else:
            return C_ao

    def _from_subspace(self, C_sub):
        """将子空间系数还原回 AO 系数"""
        if self.strict_diagonalize:
            # 还原公式: C_ao = X @ C_sub
            return self.X @ C_sub
        else:
            return C_sub

    def apply_delta_kick(self, strength: float = 0.001, direction: str = 'z'):
        dir_map = {'x': 0, 'y': 1, 'z': 2}
        idx = dir_map[direction.lower()]
        
        # 1. 准备子空间矩阵
        if self.strict_diagonalize:
            D_op = self.X.T @ self.D_ao[idx] @ self.X
        else:
            D_op = self.D_ao[idx]
            
        # 2. 投影系数到子空间
        c_a_sub = self._to_subspace(self.C_alpha)
        c_b_sub = self._to_subspace(self.C_beta)
        
        # 3. 执行 Kick (在子空间求解)
        # (S_sub - i*k/2*D) C_new = (S_sub + i*k/2*D) C_old
        factor = 1j * (strength / 2.0)
        LHS = self.S_sub - factor * D_op
        RHS = self.S_sub + factor * D_op
        
        c_a_new = solve(LHS, RHS @ c_a_sub)
        c_b_new = solve(LHS, RHS @ c_b_sub)
        
        # 4. 还原系数到 AO 空间并保存
        self.C_alpha = self._from_subspace(c_a_new)
        self.C_beta = self._from_subspace(c_b_new)
        
        # 重置历史
        self.time_history = [0.0]
        self.dipole_history = [self._compute_current_dipole()]
        logging.info(f"Kick 施加成功 (方向 {direction})")

    def propagate(self, total_time: float, 
                  kick_params: dict = None, 
                  field_func = None, 
                  field_direction: str = 'z',
                  print_interval: int = 100):
        """
        通用的时间演化函数 (支持 Delta Kick 和 任意激光场)
        
        Args:
            total_time: 总演化时间 (a.u.)
            kick_params: 字典 {'strength': float, 'direction': str}. 
                         如果存在，在 t=0 施加瞬时相位激发。
            field_func: 函数 f(t) -> float. 返回 t 时刻的电场强度 E(t)。
                        如果为 None，则进行自由演化。
            field_direction: 外场 f(t) 的极化方向 ('x', 'y', 'z')。
            print_interval: 打印间隔。
        """
        steps = int(total_time / self.dt)
        logging.info(f"开始通用演化: {total_time} a.u. ({steps} 步)")
        
 
        # 预计算子空间偶极矩阵 (用于 Kick 或 Laser)
        D_sub_all = np.zeros((3, self.S_sub.shape[0], self.S_sub.shape[1]))
        if self.strict_diagonalize:
            for i in range(3):
                D_sub_all[i] = self.X.T @ self.D_ao[i] @ self.X
        else:
            D_sub_all = self.D_ao

        # 确定激光方向索引
        dir_map = {'x': 0, 'y': 1, 'z': 2}
        laser_dir_idx = dir_map[field_direction.lower()]
        D_laser_op = D_sub_all[laser_dir_idx]

        #处理 Delta Kick (t=0 初始化)
        if kick_params:
            k_str = kick_params.get('strength', 0.0)
            k_dir = kick_params.get('direction', 'z')
            idx = dir_map[k_dir.lower()]
            D_kick = D_sub_all[idx]
            
            logging.info(f"应用 Delta Kick: strength={k_str}, dir={k_dir}")
            
            # (S - ik/2 D) C = (S + ik/2 D) C
            factor = 1j * (k_str / 2.0)
            LHS = self.S_sub - factor * D_kick
            RHS = self.S_sub + factor * D_kick
            
            # 将当前系数投影到子空间 -> Kick -> 更新
            c_a_sub = self._to_subspace(self.C_alpha)
            c_b_sub = self._to_subspace(self.C_beta)
            c_a_sub = solve(LHS, RHS @ c_a_sub)
            c_b_sub = solve(LHS, RHS @ c_b_sub)
            
            # 更新类成员
            self.C_alpha = self._from_subspace(c_a_sub)
            self.C_beta = self._from_subspace(c_b_sub)
            
            # 重置历史
            self.time_history = [0.0]
            self.dipole_history = [self._compute_current_dipole()] # AO空间计算最准
        
        current_time = 0.0 if not self.time_history else self.time_history[-1]
        
        # 投影系数常驻子空间
        c_a_sub = self._to_subspace(self.C_alpha)
        c_b_sub = self._to_subspace(self.C_beta)
        
        # 初始 Hamiltonian (仅 KS 部分)
        H_a_ks, H_b_ks = self._build_hamiltonian_subspace(self.C_alpha, self.C_beta)
        
        # 预计算常数
        factor_plus = 1j * (self.dt / 2.0)
        factor_minus = -1j * (self.dt / 2.0)

        for step in range(steps):
            t_now = current_time
            t_next = current_time + self.dt
            
            # --- 计算外场势 V_ext ---
            V_ext_now = 0.0
            V_ext_next = 0.0
            field_val = 0.0
            
            if field_func is not None:
                field_val = field_func(t_now)
                E_next = field_func(t_next)
                # V = E(t) * D
                V_ext_now = field_val * D_laser_op
                V_ext_next = E_next * D_laser_op
            
            # --- 1. Predictor ---
            H_a_tot_now = H_a_ks + V_ext_now
            H_b_tot_now = H_b_ks + V_ext_now
            
            LHS_a = self.S_sub + factor_plus * H_a_tot_now
            RHS_a = self.S_sub + factor_minus * H_a_tot_now
            c_a_pred = solve(LHS_a, RHS_a @ c_a_sub)
            
            LHS_b = self.S_sub + factor_plus * H_b_tot_now
            RHS_b = self.S_sub + factor_minus * H_b_tot_now
            c_b_pred = solve(LHS_b, RHS_b @ c_b_sub)
            
            # --- 2. Corrector ---
            # 还原预估波函数算 H_KS(t+dt)
            C_a_pred_ao = self._from_subspace(c_a_pred)
            C_b_pred_ao = self._from_subspace(c_b_pred)
            H_a_ks_next, H_b_ks_next = self._build_hamiltonian_subspace(C_a_pred_ao, C_b_pred_ao)
            
            # 组合 H_total(t+dt)
            H_a_tot_next = H_a_ks_next + V_ext_next
            H_b_tot_next = H_b_ks_next + V_ext_next
            
            # 取平均
            H_a_avg = 0.5 * (H_a_tot_now + H_a_tot_next)
            H_b_avg = 0.5 * (H_b_tot_now + H_b_tot_next)
            
            # 最终演化
            LHS_a = self.S_sub + factor_plus * H_a_avg
            RHS_a = self.S_sub + factor_minus * H_a_avg
            c_a_sub = solve(LHS_a, RHS_a @ c_a_sub)
            
            LHS_b = self.S_sub + factor_plus * H_b_avg
            RHS_b = self.S_sub + factor_minus * H_b_avg
            c_b_sub = solve(LHS_b, RHS_b @ c_b_sub)
            
            # 滚动更新
            H_a_ks = H_a_ks_next
            H_b_ks = H_b_ks_next
            
            current_time += self.dt
            self.time_history.append(current_time)
            
            # 子空间快速计算偶极矩
            mu = self._compute_dipole_subspace(c_a_sub, c_b_sub, D_sub_all)
            self.dipole_history.append(mu)
            
            if (step + 1) % print_interval == 0:
                logging.info(f"Step {step+1}: T={current_time:.2f}, E={field_val:.4f}, Mu_z={mu[2]:.6f}")

        # 还原系数
        self.C_alpha = self._from_subspace(c_a_sub)
        self.C_beta = self._from_subspace(c_b_sub)

    def _build_hamiltonian_subspace(self, C_a_ao, C_b_ao):
        """
        输入 AO 系数，返回子空间 Hamiltonian
        """
        # 构建 AO 密度 (直接用输入的 AO 系数)
        P_a_ao = (C_a_ao @ C_a_ao.conj().T).real
        P_b_ao = (C_b_ao @ C_b_ao.conj().T).real
        P_tot_ao = P_a_ao + P_b_ao
        
        # 构建 AO Hamiltonian
        J = self.lsda_helper._build_coulomb_matrix(P_tot_ao, self.eri)
        V_xc_a, V_xc_b, _, _, _ = self.lsda_helper._build_xc_potential_matrix(P_a_ao, P_b_ao)
        
        H_a_ao = self.H_core_ao + J + V_xc_a
        H_b_ao = self.H_core_ao + J + V_xc_b
        
        # 投影到子空间
        if self.strict_diagonalize:
            H_a_sub = self.X.T @ H_a_ao @ self.X
            H_b_sub = self.X.T @ H_b_ao @ self.X
        else:
            H_a_sub = H_a_ao
            H_b_sub = H_b_ao
            
        return H_a_sub, H_b_sub

    def _compute_current_dipole(self) -> np.ndarray:
        """直接使用 AO 系数计算偶极矩"""
        P_a = (self.C_alpha @ self.C_alpha.conj().T).real
        P_b = (self.C_beta @ self.C_beta.conj().T).real
        P_tot = P_a + P_b
        
        mu = np.zeros(3)
        for i in range(3):
            mu[i] = np.sum(P_tot * self.D_ao[i].T)
        return mu

    def calculate_spectrum(self, kick_strength: float, damping: float = 0.005) -> Dict:
        """计算光吸收谱 (带符号修正)"""
        time = np.array(self.time_history)
        mu_z = np.array([d[2] for d in self.dipole_history])
        
        mu_induced = mu_z - mu_z[0]
        window = np.exp(-time * damping)
        signal = mu_induced * window
        
        n_points = len(signal)
        n_fft = n_points * 4
        freqs = fftfreq(n_fft, d=self.dt) * 2 * np.pi
        f_signal = fft(signal, n=n_fft)
        
        alpha = f_signal * self.dt / kick_strength
        mask = freqs > 0
        freqs_pos = freqs[mask]
        intensity = freqs_pos * np.imag(alpha[mask]) * -1.0 # 修正符号
        
        return {
            'time': time, 'mu_induced': mu_induced,
            'energy_ev': freqs_pos * 27.2114, 'intensity': intensity
        }