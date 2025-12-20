"""
atomic_lsda.py - 完整版自旋极化原子LSDA计算模块
基于已实现的积分模块进行Kohn-Sham DFT的自洽场计算
使用完整的、基于解析导数的Slater-VWN5交换相关泛函
"""
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import simpson
from typing import Dict, Tuple
import time
from dataclasses import dataclass
from .atomic_integrals import AtomicIntegrals
import logging
from scipy.optimize import brentq
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 交换相关泛函 (XC Functional) 的独立实现 ---
# 这部分代码实现了完整的、解析的Slater Exchange + VWN5 Correlation

def get_slater_vwn5(rho_alpha: np.ndarray, rho_beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算Slater Exchange + VWN5 Correlation的交换相关势和能量密度。
    
    该实现是完整的，并使用解析导数计算势。
    
    Args:
        rho_alpha: α自旋密度数组 (在径向网格上)
        rho_beta:  β自旋密度数组 (在径向网格上)
        
    Returns:
        (v_xc_alpha, v_xc_beta, eps_xc): XC势(α, β)和XC能量密度
    """
    # 定义常数
    # Slater exchange: C_x = (3/4) * (3/pi)^(1/3) * 2^(1/3)
    X_FACTOR = (3.0 / 4.0) * (3.0 / np.pi)**(1.0 / 3.0) * (2.0**(1.0/3.0))
    FOUR_THIRDS = 4.0 / 3.0
    
    # VWN5 参数 (P: Paramagnetic, F: Ferromagnetic)
    VWN_PARAMS = {
        'P': {'A': 0.0310907, 'x0': -0.10498, 'b': 3.72744, 'c': 12.9352},
        'F': {'A': 0.01554535, 'x0': -0.32500, 'b': 7.06042, 'c': 18.0578}
    }
    
    # 安全处理，避免除以零或对负数开方
    rho_alpha = np.maximum(rho_alpha, 1e-40)
    rho_beta = np.maximum(rho_beta, 1e-40)
    rho_total = rho_alpha + rho_beta

    # --- 1. Slater Exchange ---
    # 自旋极化交换能密度:
    # ε_x = -C_x * [rho_alpha^(4/3) + rho_beta^(4/3)] / rho_total
    # 其中 C_x = (3/4) * (3/pi)^(1/3) * 2^(1/3)
    Ex_total_density = -X_FACTOR * (rho_alpha**FOUR_THIRDS + rho_beta**FOUR_THIRDS)
    eps_x = Ex_total_density / rho_total

    # 势 v_x_sigma = d(rho*eps_x)/d(rho_sigma)
    # = -C_x * (4/3) * rho_sigma^(1/3)
    v_x_alpha = -FOUR_THIRDS * X_FACTOR * (rho_alpha)**(1.0/3.0)
    v_x_beta = -FOUR_THIRDS * X_FACTOR * (rho_beta)**(1.0/3.0)

    # --- 2. VWN5 Correlation ---
    # 定义核心VWN函数及其解析导数
    def _vwn_G(rs, p):
        x = np.sqrt(rs)
        X = x**2 + p['b']*x + p['c']
        Q = np.sqrt(4*p['c'] - p['b']**2)
        
        log_term1 = np.log(x**2 / X)
        atan_term1 = 2 * p['b'] / Q * np.arctan(Q / (2*x + p['b']))
        
        log_term2 = np.log((x - p['x0'])**2 / X)
        atan_term2 = 2 * (p['b'] + 2*p['x0']) / Q * np.arctan(Q / (2*x + p['b']))
        
        return p['A'] * (log_term1 + atan_term1 - (p['b']*p['x0'] / (p['x0']**2 + p['b']*p['x0'] + p['c'])) * (log_term2 + atan_term2))

    def _d_vwn_G_d_rs(rs, p):
        x = np.sqrt(rs)
        X = x**2 + p['b']*x + p['c']
        
        dX_dx = 2*x + p['b']
        
        term1 = (2/x - dX_dx/X)
        term2 = -2 * p['b'] / ((2*x + p['b'])**2 + Q**2)
        
        prefactor2 = p['b']*p['x0'] / (p['x0']**2 + p['b']*p['x0'] + p['c'])
        term3 = (2/(x-p['x0']) - dX_dx/X)
        term4 = -2 * (p['b']+2*p['x0']) / ((2*x + p['b'])**2 + Q**2)
        
        dG_dx = p['A'] * (term1 + term2 - prefactor2 * (term3 + term4))
        return dG_dx * (0.5 / x) # chain rule: dG/drs = dG/dx * dx/drs

    # 自旋极化率及其相关函数
    zeta = (rho_alpha - rho_beta) / rho_total
    
    # 自旋插值函数 f(zeta) 及其导数 f'(zeta)
    f_zeta_numerator = (1 + zeta)**FOUR_THIRDS + (1 - zeta)**FOUR_THIRDS - 2
    f_zeta_denominator = 2**FOUR_THIRDS - 2
    f_zeta = f_zeta_numerator / f_zeta_denominator
    
    df_dzeta = FOUR_THIRDS / f_zeta_denominator * ((1 + zeta)**(1.0/3.0) - (1 - zeta)**(1.0/3.0))

    # 计算 Wigner-Seitz 半径
    rs = (3.0 / (4.0 * np.pi * rho_total))**(1.0 / 3.0)

    # 计算P和F极限下的关联能密度及其对rs的导数
    p_P, p_F = VWN_PARAMS['P'], VWN_PARAMS['F']
    Q = np.sqrt(4*p_P['c'] - p_P['b']**2) # Q is the same for both
    
    eps_c_P = _vwn_G(rs, p_P)
    eps_c_F = _vwn_G(rs, p_F)
    
    d_eps_c_P_d_rs = _d_vwn_G_d_rs(rs, p_P)
    d_eps_c_F_d_rs = _d_vwn_G_d_rs(rs, p_F)
    
    # 最终的关联能密度 eps_c
    delta_eps_c = eps_c_F - eps_c_P
    eps_c = eps_c_P + f_zeta * delta_eps_c

    # 计算关联势 v_c
    # v_c_sigma = d(rho * eps_c) / d(rho_sigma)
    # 使用链式法则: v_c = eps_c - (rs/3)*d(eps_c/drs) + (1-sigma*zeta)*d(eps_c/dzeta)
    d_eps_c_d_rs = d_eps_c_P_d_rs + f_zeta * (d_eps_c_F_d_rs - d_eps_c_P_d_rs)
    d_eps_c_d_zeta = df_dzeta * delta_eps_c

    common_v_c_term = eps_c - (rs / 3.0) * d_eps_c_d_rs
    v_c_alpha = common_v_c_term + d_eps_c_d_zeta * (1 - zeta)
    v_c_beta = common_v_c_term - d_eps_c_d_zeta * (1 + zeta)

    # --- 3. 组合XC部分 ---
    eps_xc = eps_x + eps_c
    v_xc_alpha = v_x_alpha + v_c_alpha
    v_xc_beta = v_x_beta + v_c_beta
    
    return v_xc_alpha, v_xc_beta, eps_xc

# --- 主LSDA计算类 ---
# (与之前版本结构相同，但调用了新的XC泛函)
@dataclass
class KSResults:
    """Kohn-Sham DFT计算结果的数据结构"""
    # (内容与之前版本完全相同)
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    coefficients_alpha: np.ndarray
    coefficients_beta: np.ndarray
    density_matrix_alpha: np.ndarray
    density_matrix_beta: np.ndarray
    ks_matrix_alpha: np.ndarray
    ks_matrix_beta: np.ndarray
    electron_configuration: Dict
    integral_calc: AtomicIntegrals
    energies: Dict[str, float]
    occ_alpha: float
    occ_beta: float
    total_potential_alpha: np.ndarray = None
    total_potential_beta: np.ndarray = None

class AtomicLSDA:
    """
    自旋极化原子Kohn-Sham LSDA计算类
    实现基于完整SVWN5泛函的自洽场计算
    """
    def __init__(self, integral_calc: AtomicIntegrals, n_electrons: int = None,
                 multiplicity: int = None,
                 max_iterations: int = 100,
                 energy_threshold: float = 1e-8,
                 density_threshold: float = 1e-6,
                 damping_factor: float = 0.5,
                 strict_diagonalize: bool = False,
                 diagonalization_threshold: float = 1e-6,
                 occ_method: str = 'fermi',  # 'fermi' or 'aufbau' or else
                 temperature: float = 0.005):
        self.integral_calc = integral_calc
        self.n_electrons = n_electrons or self.integral_calc.nuclear_charge
        self.n_basis = self.integral_calc.n_basis
        self.r_grid = self.integral_calc.r_grid
        self.r_squared = self.integral_calc.r_grid**2
        self.strict_diagonalize= strict_diagonalize
        self.diagonalization_threshold = diagonalization_threshold
        
        self.multiplicity = multiplicity or self._get_ground_state_multiplicity_default()
        if self.multiplicity is None or self.multiplicity < 1:
            raise ValueError("必须提供有效的自旋多重度")
        self.n_alpha, self.n_beta = self._determine_electron_numbers()
        
        self.max_iterations = max_iterations
        self.energy_threshold = energy_threshold
        self.density_threshold = density_threshold
        self.damping_factor = damping_factor
        self.temperature = temperature
        self.occ_method = occ_method
        logging.info("初始化LSDA计算:")
        logging.info(f"基组数量: {self.n_basis}")
        logging.info(f"电子数: {self.n_electrons}")
        logging.info(f"电子配置: {self.n_alpha}α + {self.n_beta}β = {self.n_electrons}")
        logging.info(f"自旋多重度: {self.multiplicity}")
        logging.info(f"最大迭代次数: {self.max_iterations}")
        logging.info(f"能量收敛阈值: {self.energy_threshold}")
        logging.info(f"密度收敛阈值: {self.density_threshold}")
        logging.info(f"阻尼因子: {self.damping_factor}")
        if self.occ_method == 'fermi':
            logging.info("使用费米-狄拉克分布计算占据数")
            logging.info(f"温度 (用于费米-狄拉克占据): {self.temperature}")
        elif self.occ_method == 'aufbau':
            logging.info("使用Aufbau方法占据")
        else:
            logging.info("未知方法,使用默认占据")
        
        self.S = self.integral_calc.compute_overlap_matrix()
        self.T = self.integral_calc.compute_kinetic_matrix()
        self.V_nuc = self.integral_calc.compute_nuclear_attraction_matrix()
        self.H_core = self.T + self.V_nuc
        self.eri= self.integral_calc.compute_electron_repulsion_integrals()
        
        logging.info("单电子积分计算完成，开始SCF迭代...")
    
    def _get_ground_state_multiplicity_default(self) -> int:
        multiplicities = {"H": 2, "He": 1, "Li": 2, "Be": 1, "B": 2, "C": 3, "N": 4, "O": 3, "F": 2, "Ne": 1,"Al":2}
        if self.integral_calc.basis.element not in multiplicities:
            logging.warning("无法确定默认自旋多重度，请手动指定。")
        return multiplicities.get(self.integral_calc.basis.element, 1)

    def _determine_electron_numbers(self) -> Tuple[int, int]:
        S = (self.multiplicity - 1) / 2
        n_unpaired = int(2 * S)
        n_alpha = (self.n_electrons + n_unpaired) // 2
        n_beta = self.n_electrons - n_alpha
        return n_alpha, n_beta

    def _strict_diagonalize(self, H, S, threshold=1e-5):
            """
            求解广义本征值问题 HC = SCe，处理线性相关性。
            
            关键修正：为了保持矩阵形状一致性 (n_basis, n_basis)，
            我们将被剔除的线性相关轨道的系数设为 0，能量设为极大值 (10000 Ha)。
            这样外部代码 (如 run_scf) 就不需要修改形状逻辑。
            """
            # 1. 对角化重叠矩阵 S
            s_vals, U = eigh(S)
            
            # 2. 检测线性相关性
            mask = s_vals > threshold
            
            # if np.sum(mask) < len(s_vals):
            #     logging.debug(f"剔除线性相关基函数: {len(s_vals)} -> {np.sum(mask)}")
                
            s_reg = s_vals[mask]
            U_reg = U[:, mask]
            
            # 3. 正则正交化 (Canonical Orthogonalization)
            # 构造变换矩阵 X = U * s^(-1/2)
            X = U_reg / np.sqrt(s_reg)
            
            # 变换哈密顿量: H' = X.T * H * X
            H_prime = X.T @ H @ X
            
            # 求解子空间内的本征值
            eps_reg, C_prime = eigh(H_prime)
            
            # 变换回原子轨道基组: C_valid = X * C'
            # C_valid 的形状是 (n_basis, n_valid_mo)
            C_valid = X @ C_prime
            
            # --- 关键：恢复矩阵形状 (Padding) ---
            n_basis = H.shape[0]
            n_valid = C_valid.shape[1]
            
            # 1. 填充能量：将无效轨道的能量设为 10000.0 (远高于费米面，确保不被占据)
            eps_full = np.full(n_basis, 10000.0)
            eps_full[:n_valid] = eps_reg
            
            # 2. 填充系数：将无效轨道的系数设为 0.0
            C_full = np.zeros((n_basis, n_basis))
            C_full[:, :n_valid] = C_valid
            
            return eps_full, C_full
    def _solve_ks_equation(self, K: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.strict_diagonalize:
            return self._strict_diagonalize(K, S)
        else:
            return eigh(K, S)
    def run_scf(self) -> KSResults:
        logging.info(f"\n开始LSDA SCF迭代...")
        logging.info(f"收敛阈值: 能量 < {self.energy_threshold:.2e}, 密度 < {self.density_threshold:.2e}")

        _, C_guess = self._solve_ks_equation(self.H_core, self.S)
        # 构造简单的初始占据 (alpha)
        occ_alpha_guess = np.zeros(self.n_basis)
        occ_alpha_guess[:self.n_alpha] = 1.0
        occ_alpha=occ_alpha_guess.copy()
        # 构造简单的初始占据 (beta)
        occ_beta_guess = np.zeros(self.n_basis)
        occ_beta_guess[:self.n_beta] = 1.0
        occ_beta=occ_beta_guess.copy()
        P_alpha = self._build_density_matrix(C_guess, occ_alpha)
        P_beta = self._build_density_matrix(C_guess, occ_beta)
        
        E_old = 0.0

        for iteration in range(self.max_iterations):
            P_total = P_alpha + P_beta
            J = self._build_coulomb_matrix(P_total, self.eri)
            V_xc_alpha, V_xc_beta, eps_xc_grid, rho_3d_total, rho_radial_total = self._build_xc_potential_matrix(P_alpha, P_beta)

            K_alpha = self.H_core + J + V_xc_alpha
            K_beta = self.H_core + J + V_xc_beta

            eps_alpha, C_alpha = self._solve_ks_equation(K_alpha, self.S)
            eps_beta, C_beta = self._solve_ks_equation(K_beta, self.S)
            occ_alpha = self._compute_occupations(eps_alpha, self.n_alpha)
            occ_beta = self._compute_occupations(eps_beta, self.n_beta)
           

            # === 构建新的密度矩阵 ===
            P_alpha_new = self._build_density_matrix(C_alpha, occ_alpha)
            P_beta_new = self._build_density_matrix(C_beta, occ_beta)

            P_alpha = self._apply_damping(P_alpha, P_alpha_new)
            P_beta = self._apply_damping(P_beta, P_beta_new)

            E_new, energies = self._calculate_total_energy(P_alpha, P_beta, J, eps_xc_grid, rho_radial_total)

            energy_change = abs(E_new - E_old)
            density_change = self._calculate_density_change(P_alpha, P_alpha_new, P_beta, P_beta_new)
            
            logging.info(f"迭代 {iteration+1:3d}: E = {E_new:15.8f} Ha, "
                         f"ΔE = {energy_change:10.2e}, Δρ = {density_change:10.2e}")

            if energy_change < self.energy_threshold and density_change < self.density_threshold:
                logging.info(f"\nSCF收敛! 迭代次数: {iteration+1}")
                converged = True
                break
            
            E_old = E_new
        else:
            logging.warning(f"\n警告: SCF未在{self.max_iterations}次迭代内收敛!")
            converged = False

        electron_config = self._analyze_electron_configuration(eps_alpha, eps_beta, C_alpha, C_beta)

        rho_alpha_3d, rho_alpha_radial = self._calculate_electron_density_on_grid(P_alpha)
        rho_beta_3d, rho_beta_radial = self._calculate_electron_density_on_grid(P_beta)
        v_xc_alpha_grid, v_xc_beta_grid, _ = get_slater_vwn5(rho_alpha_3d, rho_beta_3d)
        
        # 2. 计算最终的 Hartree 势网格值
        rho_radial_total = rho_alpha_radial + rho_beta_radial
        v_hartree_grid = self._solve_hartree_potential(rho_radial_total)
        
        # 3. 计算核势 V_nuc = -Z/r
        r_safe = np.where(self.r_grid < 1e-12, 1e-12, self.r_grid)
        v_nuc_grid = -self.integral_calc.nuclear_charge / r_safe
        
        # 4. 组装总有效势 V_eff = V_nuc + V_H + V_XC
        v_eff_alpha = v_nuc_grid + v_hartree_grid + v_xc_alpha_grid
        v_eff_beta = v_nuc_grid + v_hartree_grid + v_xc_beta_grid

        calc_nelec = simpson(rho_radial_total, x=self.r_grid)
        logging.info(f"设定电子数: {self.n_electrons}")
        logging.info(f"积分电子数: {calc_nelec:.4f}")
        logging.info(f"核电荷数 Z: {self.integral_calc.nuclear_charge}")


        return KSResults(
            converged=converged, iterations=iteration + 1, total_energy=E_new,
            orbital_energies_alpha=eps_alpha, orbital_energies_beta=eps_beta,
            coefficients_alpha=C_alpha, coefficients_beta=C_beta,
            density_matrix_alpha=P_alpha, density_matrix_beta=P_beta,
            ks_matrix_alpha=K_alpha, ks_matrix_beta=K_beta,
            electron_configuration=electron_config, integral_calc=self.integral_calc,
            energies=energies,
            occ_alpha=occ_alpha, occ_beta=occ_beta,
            total_potential_alpha=v_eff_alpha,
            total_potential_beta=v_eff_beta
        )

    # def _solve_hartree_potential(self, rho_radial: np.ndarray) -> np.ndarray:
    #     """
    #     求解径向泊松方程计算 Hartree 势 V_H(r)。
        
    #     修正版：使用反向积分策略，强制保证 V_H 在边界处的物理正确性，
    #     消除因 Simpson/Trapezoid 精度不匹配导致的常数平移误差。
        
    #     公式: V_H(r) = (1/r) * ∫[0->r] ρ(x) dx  +  ∫[r->∞] (ρ(x)/x) dx
    #     """
    #     from scipy.integrate import cumulative_trapezoid
        
    #     r = self.r_grid
    #     # 避免除以 0
    #     r_safe = np.where(r < 1e-12, 1e-12, r)
        
    #     # --- 第一部分：内部电荷贡献 (1/r * Q(r)) ---
    #     # 使用梯形规则计算累积电荷
    #     Q_r = cumulative_trapezoid(rho_radial, r, initial=0)
    #     v_part1 = Q_r / r_safe
        
    #     # --- 第二部分：外部壳层贡献 (∫[r->∞] rho/x dx) ---
    #     # 关键修正：使用反向积分 (从 r_max 到 r)
    #     # 这样能保证当 r -> r_max 时，积分值自然趋近于 0
    #     integrand2 = rho_radial / r_safe
        
    #     # [::-1] 倒序数组，积分后再倒序回来
    #     # initial=0 意味着从无穷远(数组末端)开始积分为0
    #     v_part2 = cumulative_trapezoid(integrand2[::-1], r[::-1], initial=0)[::-1]
        
    #     # 总 Hartree 势
    #     v_h = v_part1 + v_part2
        
    #     return v_h

    def _solve_hartree_potential(self, rho_radial: np.ndarray) -> np.ndarray:
        """
        求解径向泊松方程计算 Hartree 势 V_H(r)。
        利用静电学公式 (Atomic units):
        V_H(r) = (1/r) * ∫[0->r] ρ(x) dx  +  ∫[r->∞] (ρ(x)/x) dx
        
        Args:
            rho_radial: 径向电荷密度 (即 4πr² * ρ_3D)
            
        Returns:
            v_h: 在 r_grid 上的 Hartree 势数组
        """
        # 确保引入积分函数
        from scipy.integrate import cumulative_trapezoid, simpson
        
        r = self.r_grid
        # 避免除以 0 (r=0 处)
        r_safe = np.where(r < 1e-12, 1e-12, r)
        
        # --- 第一部分：内部电荷的贡献 (像点电荷 Q/r) ---
        # Q(r) = ∫[0->r] rho_radial(x) dx
        Q_r = cumulative_trapezoid(rho_radial, r, initial=0)
        v_part1 = Q_r / r_safe
        
        # --- 第二部分：外部壳层的贡献 (像球壳势 dQ/r') ---
        # Integral = ∫[r->∞] (rho_radial(x) / x) dx
        # 计算方法：先算 0->∞ 的总积分，减去 0->r 的累积积分
        integrand = rho_radial / r_safe
        total_integral = simpson(integrand, x=r) # 0 到无穷的总积分
        cum_integral = cumulative_trapezoid(integrand, r, initial=0) # 0 到 r 的积分
        
        v_part2 = total_integral - cum_integral
        
        # 总势 V_H(r)
        v_h = v_part1 + v_part2
        
        return v_h
    
    def _calculate_electron_density_on_grid(self, P: np.ndarray) -> np.ndarray:
        # 计算径向密度 rho_radial(r) = r² × ρ_3D(r)
        # 注意：radial_functions 存储的是 χ(r) = r*R(r)
        # 径向密度: rho_radial(r) = r² |R(r)|² = |χ(r)|²
        # 所以: Σ P_mn χ_m χ_n = rho_radial
        #
        # 只有相同角动量量子数(l,m)的轨道对才对球对称平均密度有贡献
        # 因为不同m的球谐函数正交: ∫ Y_lm × Y_lm' dΩ = δ_mm'
        # 因此需要应用angular_selection_matrix来过滤
        radial_funcs = self.integral_calc.radial_functions
        angular_mask = self.integral_calc.angular_selection_matrix
        rho_radial_grid = np.einsum('mn,mr,nr->r', P * angular_mask, radial_funcs, radial_funcs)

        # 返回径向密度和3D密度
        # 我们的归一化: ∫ χ² dr = ∫ r²R² dr = 1 (无4π)
        # 标准3D归一化: ∫ |R|² 4πr² dr = 1
        # 因此标准3D密度 ρ_3D = |R|²/(4π) = rho_radial/(4π r²)
        r_safe = np.where(self.r_grid > 1e-10, self.r_grid, 1e-10)
        rho_3d_grid = rho_radial_grid / (4.0 * np.pi * r_safe**2)

        return rho_3d_grid, rho_radial_grid

    def _build_xc_potential_matrix(self, P_alpha: np.ndarray, P_beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        #计算密度
        rho_alpha_3d, rho_alpha_radial = self._calculate_electron_density_on_grid(P_alpha)
        rho_beta_3d, rho_beta_radial = self._calculate_electron_density_on_grid(P_beta)

        #计算XC势网格值
        v_xc_alpha_grid, v_xc_beta_grid, eps_xc_grid = get_slater_vwn5(rho_alpha_3d, rho_beta_3d)

        #计算矩阵元 
        radial_funcs = self.integral_calc.radial_functions
        
        # 获取角动量选择矩阵 (Angular Mask)
        angular_mask = self.integral_calc.angular_selection_matrix

        # 计算径向积分
        integrand_alpha = np.einsum('mr,r,nr->mnr', radial_funcs, v_xc_alpha_grid, radial_funcs)
        integrand_beta = np.einsum('mr,r,nr->mnr', radial_funcs, v_xc_beta_grid, radial_funcs)

        V_xc_alpha_radial = simpson(integrand_alpha, x=self.r_grid, axis=-1)
        V_xc_beta_radial = simpson(integrand_beta, x=self.r_grid, axis=-1)

        #应用角动量遮罩
        V_xc_alpha = V_xc_alpha_radial * angular_mask
        V_xc_beta = V_xc_beta_radial * angular_mask

        return V_xc_alpha, V_xc_beta, eps_xc_grid, (rho_alpha_3d + rho_beta_3d), (rho_alpha_radial + rho_beta_radial)

    def _calculate_total_energy(self, P_alpha: np.ndarray, P_beta: np.ndarray,
                                J: np.ndarray, eps_xc_grid: np.ndarray, rho_radial_total: np.ndarray) -> Tuple[float, Dict]:
        # 计算总能量
        # E_total = T + V_nuc + J + E_XC
        P_total = P_alpha + P_beta
        E_T = np.einsum('mn,mn', P_total, self.T)
        E_Vnuc = np.einsum('mn,mn', P_total, self.V_nuc)
        E_J = 0.5 * np.einsum('mn,mn', P_total, J)

        # XC能量计算
        #rho_3d = rho_radial / (4π r²)
        #E_XC = ∫ ρ_3D(r) ε_xc(r) d³r = ∫ ρ_3D(r) ε_xc(r) 4πr² dr
        # E_XC = ∫ [rho_radial/(4πr²)] × ε_xc(r) × 4πr² dr= ∫ rho_radial(r) × ε_xc(r) dr
        integrand_exc = rho_radial_total * eps_xc_grid
        E_XC = simpson(integrand_exc, x=self.r_grid)

        E_total = E_T + E_Vnuc + E_J + E_XC

        energies = {'Kinetic': E_T, 'Nuclear_Attraction': E_Vnuc, 'Hartree': E_J, 'Exchange_Correlation': E_XC, 'Total': E_total}
        return E_total, energies
    
    def _compute_occupations(self, energies: np.ndarray, n_electrons: int) -> np.ndarray:
        if self.occ_method == 'fermi':
            return self._compute_occupations_fermi(energies, n_electrons, temperature=self.temperature)
        elif self.occ_method == 'aufbau':
            return self._compute_occupations_aufbau(energies, n_electrons)
        else:
            raise ValueError(f"未知占据方法: {self.occ_method}")
    def _compute_occupations_fermi(self, energies: np.ndarray, n_electrons: int, temperature: float = 1e-3) -> np.ndarray:
        """
        根据轨道能量和电子数，计算费米-狄拉克分数占据数。
        这能自动处理简并轨道的平均占据 (例如 2个电子分给3个p轨道 -> 各0.66)。
        """
        # 如果没有电子，直接返回全零
        if n_electrons <= 0:
            return np.zeros_like(energies)

        # 费米分布函数
        def fermi_dist(mu, eps, T):
            # 为了数值稳定性，限制指数范围
            arg = (eps - mu) / T
            arg = np.clip(arg, -100, 100)
            return 1.0 / (1.0 + np.exp(arg))

        # 寻找费米能级 mu，使得 sum(occupations) == n_electrons
        def target_func(mu):
            return np.sum(fermi_dist(mu, energies, temperature)) - n_electrons

        # 确定搜索范围 (min_E - 1, max_E + 1)
        min_e, max_e = np.min(energies), np.max(energies)
        
        # 使用 brentq 求解根 (即找到 mu)
        try:
            mu = brentq(target_func, min_e - 10.0, max_e + 10.0)
        except ValueError:
            # 如果找不到根，回退到Aufbau 原则
            logging.warning("费米能级搜索失败，回退到整数占据")
            occ = np.zeros_like(energies)
            occ[:int(n_electrons)] = 1.0
            return occ

        return fermi_dist(mu, energies, temperature)

    def _compute_occupations_aufbau(self, energies: np.ndarray, n_electrons: float, threshold: float =3) -> np.ndarray:
            # 默认简单整数占据
            occ = np.zeros_like(energies)
            occ[:int(n_electrons)] = 1.0
            return occ

    # --- 以下方法与之前版本完全相同 ---
    def _initial_guess(self, S: np.ndarray, H_core: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, C_guess = eigh(H_core, S)
        return self._build_density_matrix(C_guess, self.n_alpha), self._build_density_matrix(C_guess, self.n_beta)

    def _build_coulomb_matrix(self, P: np.ndarray, eri: np.ndarray) -> np.ndarray:
        return np.einsum('ls,mnls->mn', P, eri)

    def _solve_ks_equation(self, K: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return eigh(K, S)

    def _build_density_matrix(self, C: np.ndarray, occupations: np.ndarray) -> np.ndarray:
            """
            使用分数占据数构建密度矩阵
            P = C @ diag(occupations) @ C.T
            """
            # 利用广播机制进行加权 (相当于矩阵乘法 C * diag(occ) * C.T)
            # C 形状 (n_basis, n_basis), occupations 形状 (n_basis,)
            # C * occupations 会将每一列乘以对应的占据数
            C_weighted = C * occupations
            return C_weighted @ C.T

    def _apply_damping(self, P_old: np.ndarray, P_new: np.ndarray) -> np.ndarray:
        return (1.0 - self.damping_factor) * P_new + self.damping_factor * P_old
    
    def _calculate_density_change(self, P_alpha_old: np.ndarray, P_alpha_new: np.ndarray,
                                  P_beta_old: np.ndarray, P_beta_new: np.ndarray) -> float:
        return max(np.max(np.abs(P_alpha_new - P_alpha_old)), np.max(np.abs(P_beta_new - P_beta_old)))

    def _analyze_electron_configuration(self, eps_alpha: np.ndarray, eps_beta: np.ndarray,
                                        C_alpha: np.ndarray, C_beta: np.ndarray) -> Dict:
  
        orbital_names = self.integral_calc.basis.get_orbital_names()
        occupied_alpha, virtual_alpha, occupied_beta, virtual_beta = [], [], [], []
        for i in range(self.n_basis):
            alpha_info = {'index': i, 'name': orbital_names[i], 'energy': eps_alpha[i], 'coefficients': C_alpha[:, i]}
            if i < self.n_alpha: occupied_alpha.append(alpha_info)
            else: virtual_alpha.append(alpha_info)
            beta_info = {'index': i, 'name': orbital_names[i], 'energy': eps_beta[i], 'coefficients': C_beta[:, i]}
            if i < self.n_beta: occupied_beta.append(beta_info)
            else: virtual_beta.append(beta_info)
        return {'occupied_alpha': occupied_alpha, 'occupied_beta': occupied_beta, 'virtual_alpha': virtual_alpha, 'virtual_beta': virtual_beta,
                'homo_alpha': eps_alpha[self.n_alpha-1] if self.n_alpha > 0 else None, 'homo_beta': eps_beta[self.n_beta-1] if self.n_beta > 0 else None,
                'lumo_alpha': eps_alpha[self.n_alpha] if self.n_alpha < self.n_basis else None, 'lumo_beta': eps_beta[self.n_beta] if self.n_beta < self.n_basis else None}

def print_ks_results(results: KSResults):
    """打印 Kohn-Sham LSDA 计算结果"""

    print(f"\n" + "="*60)
    print(f"               Kohn-Sham LSDA 计算结果 (完整泛函)")
    print(f"="*60)
    
    print(f"收敛状态: {'收敛' if results.converged else '未收敛'}")
    print(f"迭代次数: {results.iterations}")
    
    # --- DFT 特有的能量分量打印 ---
    print("\n--- 能量分量 (Hartree) ---")
    # 按照物理意义排序打印
    order = ['Kinetic', 'Nuclear_Attraction', 'Hartree', 'Exchange_Correlation']
    for key in order:
        if key in results.energies:
            print(f"{key:<20}: {results.energies[key]:15.8f}")
            
    print(f"\n总能量: {results.total_energy:.8f} Hartree")
    print(f"总能量: {results.total_energy * 27.2114:.4f} eV")
    
    # 轨道能级
    print(f"\n{'='*60}")
    print(f"           轨道能级和电子构型")
    print(f"{'='*60}")
    
    config = results.electron_configuration
    
    # 打印 Alpha 轨道
    print(f"\nα自旋轨道 (占据):")
    print(f"{'序号':<4} {'轨道':<8} {'能量(Ha)':<12} {'能量(eV)':<12}")
    print(f"-" * 40)
    for orbital in config['occupied_alpha']:
        energy_ev = orbital['energy'] * 27.2114
        print(f"{orbital['index']+1:<4} {orbital['name']:<8} {orbital['energy']:<12.6f} {energy_ev:<12.4f}")
    
    # 打印 Beta 轨道
    print(f"\nβ自旋轨道 (占据):")
    print(f"{'序号':<4} {'轨道':<8} {'能量(Ha)':<12} {'能量(eV)':<12}")
    print(f"-" * 40)
    for orbital in config['occupied_beta']:
        energy_ev = orbital['energy'] * 27.2114
        print(f"{orbital['index']+1:<4} {orbital['name']:<8} {orbital['energy']:<12.6f} {energy_ev:<12.4f}")
    
    # HOMO-LUMO信息
    print(f"\n关键能级:")
    if config['homo_alpha'] is not None:
        print(f"HOMO(α): {config['homo_alpha']:.6f} Ha ({config['homo_alpha']*27.2114:.4f} eV)")
    if config['homo_beta'] is not None:
        print(f"HOMO(β): {config['homo_beta']:.6f} Ha ({config['homo_beta']*27.2114:.4f} eV)")
    if config['lumo_alpha'] is not None:
        print(f"LUMO(α): {config['lumo_alpha']:.6f} Ha ({config['lumo_alpha']*27.2114:.4f} eV)")
    if config['lumo_beta'] is not None:
        print(f"LUMO(β): {config['lumo_beta']:.6f} Ha ({config['lumo_beta']*27.2114:.4f} eV)")
    
    # 第一个未占据态 (Virtual)
    if config['virtual_alpha']:
        print(f"\n第一个未占据态:")
        first_virtual_alpha = config['virtual_alpha'][0]
        energy_ev = first_virtual_alpha['energy'] * 27.2114
        print(f"α: {first_virtual_alpha['name']} = {first_virtual_alpha['energy']:.6f} Ha ({energy_ev:.4f} eV)")
    
    if config['virtual_beta']:
        if not config['virtual_alpha']: print(f"\n第一个未占据态:") # 避免重复打印标题
        first_virtual_beta = config['virtual_beta'][0]
        energy_ev = first_virtual_beta['energy'] * 27.2114
        print(f"β: {first_virtual_beta['name']} = {first_virtual_beta['energy']:.6f} Ha ({energy_ev:.4f} eV)")