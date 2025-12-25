"""
tddft_rt.py - 实时时间相关密度泛函理论 (RT-TDDFT) 模块
用于模拟电子在时域内的演化并计算光吸收谱
"""

import numpy as np
from scipy.linalg import solve, inv, expm
from scipy.integrate import simpson
from scipy.fftpack import fft, fftfreq
import logging
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt

from .LSDA import KSResults, AtomicLSDA, get_slater_vwn5

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealTimeTDDFT:
    """
    实时 TDDFT 计算类
    使用 Crank-Nicolson 传播子求解含时 Kohn-Sham 方程
    """

    def __init__(self, ks_results: KSResults, dt: float = 0.05):
        """
        初始化 RT-TDDFT 模拟器

        Args:
            ks_results: 基态 LSDA 计算结果
            dt: 时间步长 (atomic units, 1 a.u. time ≈ 24 as)
        """
        self.ks = ks_results
        self.integral_calc = ks_results.integral_calc
        self.n_basis = self.integral_calc.n_basis
        self.dt = dt
        
        # 基础矩阵 (S, T, V_nuc)
        self.S = self.integral_calc.compute_overlap_matrix()
        self.H_core = self.integral_calc.compute_core_hamiltonian()
        self.eri = self.integral_calc.compute_electron_repulsion_integrals()
        
        # 偶极矩阵 (用于 Kick 和 观测) [3, n_basis, n_basis]
        # 假设 atomic_integrals 已经实现了 compute_dipole_matrix
        if 'dipole' in self.integral_calc.compute_all_integrals():
            self.D = self.integral_calc.compute_all_integrals()['dipole']
        else:
            # 如果未缓存，现场计算
            self.D = self.integral_calc.compute_dipole_matrix()

        # 初始化复数系数矩阵 (Alpha / Beta)
        # 形状: [n_basis, n_occ] (只传播占据轨道以节省时间)
        self.C_alpha = self.ks.coefficients_alpha[:, :self.ks.occ_alpha.sum().astype(int)].astype(np.complex128)
        self.C_beta = self.ks.coefficients_beta[:, :self.ks.occ_beta.sum().astype(int)].astype(np.complex128)
        
        # 记录占据数 (用于构建密度矩阵)
        # 在 RT-TDDFT 中，轨道占据数通常随时间保持不变 (幺正演化)
        self.n_occ_alpha = self.C_alpha.shape[1]
        self.n_occ_beta = self.C_beta.shape[1]

        # 历史记录
        self.time_history = []
        self.dipole_history = [] # 记录 [mu_x, mu_y, mu_z]
        
        # 辅助工具：AtomicLSDA 实例 (用于复用势构建逻辑)
        # 我们创建一个轻量级的实例，主要用它的 grid 和方法
        self.lsda_helper = AtomicLSDA(self.integral_calc, 
                                      n_electrons=self.ks.occ_alpha.sum()+self.ks.occ_beta.sum(),
                                      multiplicity=1) # 参数不重要，只用方法

        logging.info(f"RT-TDDFT 初始化完成。时间步长 dt = {self.dt} a.u.")

    def apply_delta_kick(self, strength: float = 0.001, direction: str = 'z'):
        """
        在 t=0 时刻施加 Delta 脉冲电场激发 (Delta Kick)
        V_ext(t) = - k * δ(t) * r_direction
        
        等效于对波函数施加相位因子: ψ(0+) = exp(i * k * r) * ψ(0)
        
        Args:
            strength (k): 激发电场强度 (通常取小值，如 0.001 - 0.01 a.u.)
            direction: 电场方向 'x', 'y', 'z'
        """
        logging.info(f"施加 Delta Kick: 强度={strength}, 方向={direction}")
        
        dir_map = {'x': 0, 'y': 1, 'z': 2}
        idx = dir_map[direction.lower()]
        
        # 获取对应方向的偶极矩阵 D_dir
        D_dir = self.D[idx]
        
        # 计算 Kick 算符 U = exp(i * k * z)
        # 在非正交基组下，这比较复杂。
        # 两种近似方法：
        # 1. 严格法：C_new = expm(i * k * S^-1 * D) @ C_old
        # 2. Crank-Nicolson Kick (保持幺正性): (S - i*k/2 * D) C_new = (S + i*k/2 * D) C_old
        
        # 这里使用方法 2 (CN Kick)，数值上更稳定且不需要计算矩阵指数
        # LHS * C_new = RHS * C_old
        LHS = self.S - 1j * (strength / 2.0) * D_dir
        RHS = self.S + 1j * (strength / 2.0) * D_dir
        
        # 更新系数
        # 注意：solve(A, b) 解 Ax = b
        self.C_alpha = solve(LHS, RHS @ self.C_alpha)
        self.C_beta  = solve(LHS, RHS @ self.C_beta)
        
        # 重置历史
        self.time_history = [0.0]
        self.dipole_history = [self._compute_current_dipole()]

    def propagate(self, total_time: float, print_interval: int = 100):
        """
        时间传播主循环
        
        Args:
            total_time: 总模拟时间 (a.u.)
            print_interval: 打印进度的步数间隔
        """
        steps = int(total_time / self.dt)
        logging.info(f"开始时间演化: 总时间 {total_time} a.u., 总步数 {steps}")
        
        current_time = 0.0
        if len(self.time_history) > 0:
            current_time = self.time_history[-1]

        # 预计算 S 的逆或者 LU 分解可能加速，但 S + i*dt/2*H 是变化的，无法预计算 LHS
        
        for step in range(steps):
            # 1. 构建当前时刻的哈密顿量 H(t)
            H_a_t, H_b_t = self._build_hamiltonian()
            
            # 2. Crank-Nicolson 传播
            # (S + i*dt/2 * H) * C(t+dt) = (S - i*dt/2 * H) * C(t)
            
        # 2. 预估 C(t+dt) [使用 explicit 传播或简单的 CN]
            # 这里为了方便复用代码，做一次标准的 CN 预估
            LHS_a = self.S + 1j * (self.dt / 2.0) * H_a_t
            RHS_a = self.S - 1j * (self.dt / 2.0) * H_a_t
            C_alpha_pred = solve(LHS_a, RHS_a @ self.C_alpha)
            
            LHS_b = self.S + 1j * (self.dt / 2.0) * H_b_t
            RHS_b = self.S - 1j * (self.dt / 2.0) * H_b_t
            C_beta_pred = solve(LHS_b, RHS_b @ self.C_beta)
            
            # --- Corrector 步 ---
            # 3. 使用预估的波函数构建 H(t+dt)
            # 注意：我们需要临时把 C 替换成 C_pred 来算 H，算完再换回来，或者传参给 build_hamiltonian
            # 这里我们临时保存真实 C
            C_alpha_real, C_beta_real = self.C_alpha, self.C_beta
            self.C_alpha, self.C_beta = C_alpha_pred, C_beta_pred
            H_a_next, H_b_next = self._build_hamiltonian()
            self.C_alpha, self.C_beta = C_alpha_real, C_beta_real # 恢复
            
            # 4. 取平均哈密顿量 H_avg = 0.5 * (H(t) + H(t+dt))
            H_a_avg = 0.5 * (H_a_t + H_a_next)
            H_b_avg = 0.5 * (H_b_t + H_b_next)
            
            # 5. 使用 H_avg 进行最终传播
            LHS_a = self.S + 1j * (self.dt / 2.0) * H_a_avg
            RHS_a = self.S - 1j * (self.dt / 2.0) * H_a_avg
            self.C_alpha = solve(LHS_a, RHS_a @ self.C_alpha)
            
            LHS_b = self.S + 1j * (self.dt / 2.0) * H_b_avg
            RHS_b = self.S - 1j * (self.dt / 2.0) * H_b_avg
            self.C_beta = solve(LHS_b, RHS_b @ self.C_beta)
            
            # 3. 更新时间并记录观测值
            current_time += self.dt
            self.time_history.append(current_time)
            
            # 计算偶极矩
            mu = self._compute_current_dipole()
            self.dipole_history.append(mu)
            
            if (step + 1) % print_interval == 0:
                logging.info(f"Step {step+1}/{steps}: Time = {current_time:.2f} a.u., Dipole(z) = {mu[2]:.6f}")

    def _build_density_matrix(self, C: np.ndarray) -> np.ndarray:
        """
        构建复数密度矩阵 P_mu,nu = sum_i C_mu,i * C*_nu,i
        """
        # Einsum: C (n_basis, n_occ)
        # P = C @ C.conj().T
        return C @ C.conj().T

    def _build_hamiltonian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建当前的 Kohn-Sham 哈密顿量 (绝热近似 ALDA)
        H = H_core + J[rho] + V_xc[rho]
        """
        # 1. 构建密度矩阵 (复数)
        P_alpha_complex = self._build_density_matrix(self.C_alpha)
        P_beta_complex = self._build_density_matrix(self.C_beta)
        
        # 2. 获取实部密度矩阵用于构建势 (ALDA 近似只依赖粒子数密度)
        # 注意：P 的对角元是实数 (粒子数)，非对角元是复数。
        # 但 calculate_electron_density_on_grid 需要处理 angular_selection_matrix
        # 只要 rho(r) 是实数即可。
        # rho(r) = sum_mn P_mn phi_m phi_n。
        # 因为 phi 是实函数，且 P 是 Hermitian，所以 sum P_mn ... 必然是实数。
        # 我们取 P 的实部即可安全计算 J 和 rho。
        P_alpha_real = P_alpha_complex.real
        P_beta_real = P_beta_complex.real
        P_total_real = P_alpha_real + P_beta_real
        
        # 3. 计算 Coulomb 矩阵 J (使用 helper)
        J = self.lsda_helper._build_coulomb_matrix(P_total_real, self.eri)
        
        # 4. 计算 XC 势 (使用 helper)
        # 这里复用了 LSDA 中的逻辑，包含网格积分等
        V_xc_alpha, V_xc_beta, _, _, _ = self.lsda_helper._build_xc_potential_matrix(P_alpha_real, P_beta_real)
        
        # 5. 组装 H
        # H 是实对称矩阵 (在没有磁场/自旋轨道耦合且使用 ALDA 的情况下)
        # 如果要包含激光场 E(t)*z，需要在这里加 V_ext
        H_alpha = self.H_core + J + V_xc_alpha
        H_beta = self.H_core + J + V_xc_beta
        
        return H_alpha, H_beta

    def _compute_current_dipole(self) -> np.ndarray:
        """
        计算当前时刻的偶极矩 expectation value
        <mu> = Tr(P * D)
        """
        P_alpha = self._build_density_matrix(self.C_alpha)
        P_beta = self._build_density_matrix(self.C_beta)
        P_total = P_alpha + P_beta
        
        # D shape: (3, n, n), P shape: (n, n)
        # Result: (3,)
        # mu_i = sum_mn P_nm * D_i_mn = Tr(P @ D_i)
        # 由于 D 是实对称，P 是 Hermitian，Tr(PD) 是实数
        
        mu = np.zeros(3)
        for i in range(3):
            # complex trace -> real
            val = np.sum(P_total * self.D[i].T) 
            mu[i] = val.real
            
        return mu

    def calculate_spectrum(self, kick_strength: float, damping: float = 0.005) -> Dict:
        """
        计算光吸收谱 (偶极强度的 Fourier 变换)
        
        S(omega) = (4 * pi * omega / (3 * c)) * Im[ alpha(omega) ]
        其中 alpha(omega) = FT[ mu(t) - mu(0) ] / kick_strength
        
        Args:
            kick_strength: 之前施加的 kick 强度
            damping: 施加在时域信号上的指数阻尼因子 exp(-t/tau) (a.u.)，用于展宽峰值
        
        Returns:
            Dict: {energy_ev, intensity}
        """
        # 1. 准备数据
        time = np.array(self.time_history)
        # 假设 kick 沿 z 方向，分析 z 方向的偶极矩响应
        # 如果要全谱，需要做三次计算 (x, y, z kick) 并平均
        # 这里默认取 z 分量
        mu_z = np.array([d[2] for d in self.dipole_history])
        
        # 减去基态偶极矩
        mu_induced = mu_z - mu_z[0]
        
        # 2. 应用阻尼 (Windowing)
        # 避免截断误差导致的 Gibbs 振荡，同时引入 Lorentzian 展宽
        window = np.exp(-time * damping)
        signal = mu_induced * window
        
        # 3. FFT
        n_points = len(signal)
        # 补零以增加频域分辨率 (可选)
        pad_factor = 4
        n_fft = n_points * pad_factor
        
        freqs = fftfreq(n_fft, d=self.dt) * 2 * np.pi # 转换为角频率 omega (a.u.)
        f_signal = fft(signal, n=n_fft)
        
        # 4. 计算极化率 alpha(omega) 和 强度 S(omega)
        # S ~ omega * Im(alpha)
        # 注意 fft 的结果需要归一化 dt
        # alpha(omega) = (1/k) * integral(mu(t) e^iwt dt)
        
        alpha = f_signal * self.dt / kick_strength # 简单离散积分近似
        
        # 只取正频率部分
        mask = freqs > 0
        freqs_pos = freqs[mask]
        alpha_pos = alpha[mask]
        
        # 强度 (任意单位，通常正比于 omega * Im(alpha))
        intensity = freqs_pos * np.imag(alpha_pos)
        
        # 转换能量单位 to eV
        energy_ev = freqs_pos * 27.2114
        
        return {
            'time': time,
            'mu_induced': mu_induced,
            'energy_ev': energy_ev,
            'intensity': intensity,
            'alpha_imag': np.imag(alpha_pos)
        }

    def plot_spectrum(self, spectrum_data: Dict, filename: str = None):
        """简单的绘图辅助函数"""
        plt.figure(figsize=(10, 8))
        
        # 时域图
        plt.subplot(2, 1, 1)
        plt.plot(spectrum_data['time'], spectrum_data['mu_induced'])
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Induced Dipole (z)')
        plt.title('Time Domain Response')
        plt.grid(True)
        
        # 频域图
        plt.subplot(2, 1, 2)
        plt.plot(spectrum_data['energy_ev'], spectrum_data['intensity'])
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity (arb. units)')
        plt.title('Absorption Spectrum')
        plt.xlim(0, 50) # 通常关注的范围
        plt.grid(True)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            logging.info(f"光谱图已保存至 {filename}")
        else:
            plt.show()