"""
atomic_hartree_fock.py - 自旋极化原子Hartree-Fock计算

基于已实现的积分模块进行自洽场计算
"""

import numpy as np
from scipy.linalg import eigh, solve
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from .atomic_integrals import AtomicIntegrals
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class HFResults:
    """HF计算结果的数据结构"""
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    coefficients_alpha: np.ndarray
    coefficients_beta: np.ndarray
    density_matrix_alpha: np.ndarray
    density_matrix_beta: np.ndarray
    fock_matrix_alpha: np.ndarray
    fock_matrix_beta: np.ndarray
    electron_configuration: Dict
    integral_calc:AtomicIntegrals
    occ_alpha: np.ndarray
    occ_beta: np.ndarray

class AtomicHartreeFock:
    """
    自旋极化原子Hartree-Fock计算类
    
    实现自旋极化的自洽场计算，适用于原子系统
    支持开壳层和闭壳层电子结构
    """

    def __init__(self, integral_calc: AtomicIntegrals, n_electrons: int = None,
                 multiplicity: int = None,
                 max_iterations: int = 100,
                 energy_threshold: float = 1e-8,
                 density_threshold: float = 1e-6,
                 damping_factor: float = 0.0):
        """
        初始化HF计算
        
        Args:
            integral_calc: AtomicIntegrals
            n_electrons: 电子总数（默认等于核电荷数）
        """ 
        self.integral_calc = integral_calc
        self.n_electrons = n_electrons or self.integral_calc.nuclear_charge
        self.n_basis = self.integral_calc.n_basis
        
        # 确定电子配置
        self.multiplicity = multiplicity or self._get_ground_state_multiplicity_default()
        if self.multiplicity is None or self.multiplicity < 1:
            raise ValueError("必须提供有效的自旋多重度")
        self.n_alpha, self.n_beta = self._determine_electron_numbers()
        
        # 设置收敛参数
        self.max_iterations = max_iterations
        self.energy_threshold = energy_threshold
        self.density_threshold = density_threshold
        self.damping_factor = damping_factor  # 密度矩阵阻尼

        # 计算积分
        logging.info(f"初始化UHF计算:")
        logging.info(f"基组数量: {self.n_basis}")
        logging.info(f"电子数: {self.n_electrons}")
        logging.info(f"电子配置: {self.n_alpha}α + {self.n_beta}β = {self.n_electrons}")
        logging.info(f"自旋多重度: {self.multiplicity}")

        self.integrals = self.integral_calc.compute_all_integrals()

        logging.info(f"积分计算完成，开始SCF迭代...")

    def _get_ground_state_multiplicity_default(self) -> int:
        """常见元素基态自旋多重度"""
        # 基于Hund规则确定基态多重度
        multiplicities = {
            "H": 2,   # H: doublet
            "He": 1,   # He: singlet
            "Li": 2,   # Li: doublet
            "Be": 1,   # Be: singlet
            "B": 2,   # B: doublet
            "C": 3,   # C: triplet
            "N": 4,   # N: quartet
            "O": 3,   # O: triplet
            "F": 2,   # F: doublet
            "Ne": 1,  # Ne: singlet
        }
        return multiplicities.get(self.integral_calc.basis.element, 1)
    
    def _determine_electron_numbers(self) -> Tuple[int, int]:
        """确定α和β电子数"""
        # 2S = multiplicity - 1
        # n_alpha - n_beta = 2S
        # n_alpha + n_beta = n_electrons
        
        S = (self.multiplicity - 1) / 2  # 自旋量子数
        n_unpaired = int(2 * S)
        
        n_alpha = (self.n_electrons + n_unpaired) // 2
        n_beta = self.n_electrons - n_alpha
        
        return n_alpha, n_beta
    
    def run_scf(self) -> HFResults:
        """执行自洽场计算"""
        logging.info(f"\n开始Hartree-Fock SCF迭代...")
        logging.info(f"收敛阈值: 能量 < {self.energy_threshold:.2e}, 密度 {self.density_threshold:.2e}")

        # 获取积分矩阵
        S = self.integrals['overlap']
        H_core = self.integrals['core_hamiltonian']
        eri = self.integrals['electron_repulsion']
        
        # 初始猜测
        P_alpha, P_beta = self._initial_guess(S, H_core)
        E_old = 0.0
        
        # SCF迭代
        for iteration in range(self.max_iterations):
            # 构建Fock矩阵
            F_alpha = self._build_fock_matrix(H_core, P_alpha, P_beta, eri, spin='alpha')
            F_beta = self._build_fock_matrix(H_core, P_alpha, P_beta, eri, spin='beta')
            
            # 求解Fock方程
            eps_alpha, C_alpha = self._solve_fock_equation(F_alpha, S)
            eps_beta, C_beta = self._solve_fock_equation(F_beta, S)
            
            # 构建新的密度矩阵
            P_alpha_new = self._build_density_matrix(C_alpha, self.n_alpha)
            P_beta_new = self._build_density_matrix(C_beta, self.n_beta)
            
            # 应用阻尼
            P_alpha_new = self._apply_damping(P_alpha, P_alpha_new)
            P_beta_new = self._apply_damping(P_beta, P_beta_new)
            
            # 计算能量
            E_new = self._calculate_total_energy(P_alpha_new, P_beta_new, H_core, F_alpha, F_beta)
            
            # 检查收敛
            energy_change = abs(E_new - E_old)
            density_change = self._calculate_density_change(P_alpha, P_alpha_new, P_beta, P_beta_new)
            
            logging.info(f"迭代 {iteration+1:3d}: E = {E_new:15.8f} Ha, "
                  f"ΔE = {energy_change:10.2e}, Δρ = {density_change:10.2e}")
            
            if energy_change < self.energy_threshold and density_change < self.density_threshold:
                logging.info(f"\nSCF收敛! 迭代次数: {iteration+1}")
                converged = True
                break
            
            # 更新变量
            P_alpha, P_beta = P_alpha_new, P_beta_new
            E_old = E_new
        else:
            logging.warning(f"\n警告: SCF未在{self.max_iterations}次迭代内收敛!")
            converged = False
        
        # 返回结果
        electron_config = self._analyze_electron_configuration(eps_alpha, eps_beta, C_alpha, C_beta)
        occ_alpha = np.zeros_like(eps_alpha)
        occ_alpha[:self.n_alpha] = 1.0
        occ_beta = np.zeros_like(eps_beta)
        occ_beta[:self.n_beta] = 1.0
        return HFResults(
            converged=converged,
            iterations=iteration + 1,
            total_energy=E_new,
            orbital_energies_alpha=eps_alpha,
            orbital_energies_beta=eps_beta,
            coefficients_alpha=C_alpha,
            coefficients_beta=C_beta,
            density_matrix_alpha=P_alpha_new,
            density_matrix_beta=P_beta_new,
            fock_matrix_alpha=F_alpha,
            fock_matrix_beta=F_beta,
            electron_configuration=electron_config,
            integral_calc=self.integral_calc,
            occ_alpha=occ_alpha,
            occ_beta=occ_beta
        )
    
    def _initial_guess(self, S: np.ndarray, H_core: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """构建初始密度矩阵猜测"""
        logging.info("构建初始猜测...")
        
        # 使用核心哈密顿的本征态作为初始猜测
        eps_guess, C_guess = eigh(H_core, S)
        
        # 构建初始密度矩阵
        P_alpha = self._build_density_matrix(C_guess, self.n_alpha)
        P_beta = self._build_density_matrix(C_guess, self.n_beta)
        
        return P_alpha, P_beta
    
    def _build_fock_matrix(self, H_core: np.ndarray, P_alpha: np.ndarray, 
                          P_beta: np.ndarray, eri: np.ndarray, spin: str) -> np.ndarray:
        """
        构建Fock矩阵
        
        F^α = H_core + J[P^α + P^β] - K[P^α]
        F^β = H_core + J[P^α + P^β] - K[P^β]
        """
        P_total = P_alpha + P_beta
        
        if spin == 'alpha':
            P_same_spin = P_alpha
        else:
            P_same_spin = P_beta
        
        # 构建库伦算符 J
        J = self._build_coulomb_matrix(P_total, eri)
        
        # 构建交换算符 K (只对相同自旋)
        K = self._build_exchange_matrix(P_same_spin, eri)
        
        # Fock矩阵
        F = H_core + J - K
        
        return F
    
    def _build_coulomb_matrix(self, P: np.ndarray, eri: np.ndarray) -> np.ndarray:
        """构建库伦算符矩阵 J_μν = Σ_λσ P_λσ (μν|λσ)"""
        J = np.zeros((self.n_basis, self.n_basis))
        
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sigma in range(self.n_basis):
                        J[mu, nu] += P[lam, sigma] * eri[mu, nu, lam, sigma]
        
        return J
    
    def _build_exchange_matrix(self, P: np.ndarray, eri: np.ndarray) -> np.ndarray:
        """构建交换算符矩阵 K_μν = Σ_λσ P_λσ (μλ|νσ)"""
        K = np.zeros((self.n_basis, self.n_basis))
        
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sigma in range(self.n_basis):
                        K[mu, nu] += P[lam, sigma] * eri[mu, lam, nu, sigma]
        
        return K
    
    def _solve_fock_equation(self, F: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """求解广义本征值问题 FC = SCε"""
        eigenvalues, eigenvectors = eigh(F, S)
        return eigenvalues, eigenvectors
    
    def _build_density_matrix(self, C: np.ndarray, n_electrons: int) -> np.ndarray:
        """构建密度矩阵 P_μν = Σ_i^occ C_μi C_νi"""
        P = np.zeros((self.n_basis, self.n_basis))
        
        for i in range(min(n_electrons, self.n_basis)):
            for mu in range(self.n_basis):
                for nu in range(self.n_basis):
                    P[mu, nu] += C[mu, i] * C[nu, i]
        
        return P
    
    def _apply_damping(self, P_old: np.ndarray, P_new: np.ndarray) -> np.ndarray:
        """应用密度矩阵阻尼以提高收敛性"""
        return (1.0 - self.damping_factor) * P_new + self.damping_factor * P_old
    
    def _calculate_total_energy(self, P_alpha: np.ndarray, P_beta: np.ndarray,
                               H_core: np.ndarray, F_alpha: np.ndarray, 
                               F_beta: np.ndarray) -> float:
        """计算总能量"""
        # 电子能量
        E_elec = 0.5 * (np.trace(P_alpha @ (H_core + F_alpha)) + 
                       np.trace(P_beta @ (H_core + F_beta)))
        
        # 核排斥能量 (对于原子为零)
        E_nuc = 0.0
        
        return E_elec + E_nuc
    
    def _calculate_density_change(self, P_alpha_old: np.ndarray, P_alpha_new: np.ndarray,
                                 P_beta_old: np.ndarray, P_beta_new: np.ndarray) -> float:
        """计算密度矩阵变化"""
        change_alpha = np.max(np.abs(P_alpha_new - P_alpha_old))
        change_beta = np.max(np.abs(P_beta_new - P_beta_old))
        return max(change_alpha, change_beta)
    
    def _analyze_electron_configuration(self, eps_alpha: np.ndarray, eps_beta: np.ndarray,
                                       C_alpha: np.ndarray, C_beta: np.ndarray) -> Dict:
        """分析电子构型"""
        
        # 获取轨道名称
        orbital_names = self.integral_calc.basis.get_orbital_names()
        
        # 占据轨道信息
        occupied_alpha = []
        occupied_beta = []
        virtual_alpha = []
        virtual_beta = []
        
        for i in range(self.n_basis):
            orbital_info = {
                'index': i,
                'name': orbital_names[i] if i < len(orbital_names) else f'轨道{i+1}',
                'energy': eps_alpha[i],
                'coefficients': C_alpha[:, i]
            }
            
            if i < self.n_alpha:
                occupied_alpha.append(orbital_info)
            else:
                virtual_alpha.append(orbital_info)
        
        for i in range(self.n_basis):
            orbital_info = {
                'index': i,
                'name': orbital_names[i] if i < len(orbital_names) else f'轨道{i+1}',
                'energy': eps_beta[i],
                'coefficients': C_beta[:, i]
            }
            
            if i < self.n_beta:
                occupied_beta.append(orbital_info)
            else:
                virtual_beta.append(orbital_info)
        
        return {
            'occupied_alpha': occupied_alpha,
            'occupied_beta': occupied_beta,
            'virtual_alpha': virtual_alpha,
            'virtual_beta': virtual_beta,
            'homo_alpha': eps_alpha[self.n_alpha-1] if self.n_alpha > 0 else None,
            'homo_beta': eps_beta[self.n_beta-1] if self.n_beta > 0 else None,
            'lumo_alpha': eps_alpha[self.n_alpha] if self.n_alpha < self.n_basis else None,
            'lumo_beta': eps_beta[self.n_beta] if self.n_beta < self.n_basis else None
        }

def print_hf_results(results: HFResults):
    """打印HF计算结果"""

    
    print(f"\n" + "="*60)
    print(f"           Hartree-Fock 计算结果")
    print(f"="*60)
    
    print(f"收敛状态: {'收敛' if results.converged else '未收敛'}")
    print(f"迭代次数: {results.iterations}")
    print(f"总能量: {results.total_energy:.8f} Hartree")
    print(f"总能量: {results.total_energy * 27.2114:.4f} eV")
    
    # 轨道能级
    print(f"\n{'='*60}")
    print(f"           轨道能级和电子构型")
    print(f"{'='*60}")
    
    config = results.electron_configuration
    
    print(f"\nα自旋轨道 (占据):")
    print(f"{'序号':<4} {'轨道':<8} {'能量(Ha)':<12} {'能量(eV)':<12}")
    print(f"-" * 40)
    for orbital in config['occupied_alpha']:
        energy_ev = orbital['energy'] * 27.2114
        print(f"{orbital['index']+1:<4} {orbital['name']:<8} {orbital['energy']:<12.6f} {energy_ev:<12.4f}")
    
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
    
    # 第一个未占据态
    if config['virtual_alpha']:
        print(f"\n第一个未占据态:")
        first_virtual_alpha = config['virtual_alpha'][0]
        energy_ev = first_virtual_alpha['energy'] * 27.2114
        print(f"α: {first_virtual_alpha['name']} = {first_virtual_alpha['energy']:.6f} Ha ({energy_ev:.4f} eV)")
    
    if config['virtual_beta']:
        first_virtual_beta = config['virtual_beta'][0]
        energy_ev = first_virtual_beta['energy'] * 27.2114
        print(f"β: {first_virtual_beta['name']} = {first_virtual_beta['energy']:.6f} Ha ({energy_ev:.4f} eV)")

def analyze_orbital_compositions(results, threshold=0.01):
    """
    轨道分析函数：
    1. 直接使用 KS 轨道序号。
    2. 自动解析基函数名称 (n, l, m)。
    3. 显示每个轨道的前3个主要成分及系数。
    
    Args:
        results: KSResults 或 HFResults 对象
        threshold: 忽略系数绝对值小于此值的成分（仅用于显示筛选）
    """
    basis_set = results.integral_calc.basis
    orbitals = basis_set.orbitals
    n_basis = len(orbitals)
    
    # --- 1. 预先构建美观的基函数标签 ---
    # 格式如: "1s", "2pz", "3dxy"
    basis_labels = []
    l_sym = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    
    for i, orb in enumerate(orbitals):
        # 处理 m 的字符串表示
        if orb.l == 0:
            m_str = ""
        elif orb.l == 1:
            m_str = {0: 'z', -1: 'y', 1: 'x'}.get(orb.m, str(orb.m))
        elif orb.l == 2:
            m_str = {0: 'dz2', -1: 'dyz', 1: 'dxz', -2: 'dxy', 2: 'dx2-y2'}.get(orb.m, str(orb.m))
        else:
            m_str = str(orb.m)
            
        # 注意：这里的 orb.n 通常是主量子数。
        # 如果你的 AtomicOrbital 中 n 是索引，可能需要 +1 或调整。
        # 假设 orb.n 是真实的物理主量子数 (如 1, 2, 3...)
        label = f"{orb.n}{l_sym.get(orb.l, '?')}{m_str}"
        basis_labels.append(label)

    # --- 2. 定义打印单个自旋通道的辅助函数 ---
    def print_channel_info(energies, occs, coeffs, spin_name):
        print(f"\n{'-'*80}")
        print(f" {spin_name} 自旋轨道分析 (Top 3 成分)")
        print(f"{'-'*80}")
        print(f"{'序号':<6} {'能量(Ha)':<12} {'占据':<6} | {'主要成分 (系数)'}")
        print(f"{'-'*80}")

        # 遍历该通道的所有轨道
        for idx in range(len(energies)):
            e = energies[idx]
            occ = occs[idx]
            c_vec = coeffs[:, idx] # 获取该轨道的系数向量
            
            # 只显示占据轨道 或 能量较低的虚轨道 (例如前5个虚轨道)
            # 这里为了简洁，如果占据数 < 1e-3 且 索引超过了 (占据数+5)，可以跳过
            # 但为了你看清所有轨道，暂不跳过，或者你可以手动加限制
            
            # --- 核心逻辑：寻找 Top 3 成分 ---
            # 1. 计算贡献度 (这里简单用系数平方，如果是正交基近似)
            weights = c_vec**2
            
            # 2. 获取排序后的索引 (从大到小)
            sorted_indices = np.argsort(weights)[::-1]
            
            # 3. 提取前 3 个
            top_components = []
            for k in range(min(3, n_basis)):
                b_idx = sorted_indices[k]
                coeff_val = c_vec[b_idx]
                
                # 如果系数太小，就不显示了
                if abs(coeff_val) < threshold:
                    continue
                    
                b_name = basis_labels[b_idx]
                # 格式: "2pz(0.85)"
                top_components.append(f"{b_name}({coeff_val:+.2f})")
            
            comp_str = " + ".join(top_components)
            
            # 打印一行
            # 序号从 1 开始
            print(f"{idx+1:<6d} {e:<12.4f} {occ:<6.2f} | {comp_str}")

    # --- 3. 执行打印 ---
    # Alpha 通道
    if hasattr(results, 'orbital_energies_alpha'):
        print_channel_info(
            results.orbital_energies_alpha, 
            results.occ_alpha, 
            results.coefficients_alpha, 
            "Alpha (α)"
        )
    
    # Beta 通道 (如果是自旋极化计算)
    if hasattr(results, 'orbital_energies_beta') and results.orbital_energies_beta is not None:
        # 检查是否真的有 beta 数据 (有时非自旋计算 beta 是 None 或与 alpha 相同)
        # 这里假设如果是 RHF，可能 coefficients_beta 是 None，视你的实现而定
        if results.coefficients_beta is not None:
            print_channel_info(
                results.orbital_energies_beta, 
                results.occ_beta, 
                results.coefficients_beta, 
                "Beta (β)"
            )


