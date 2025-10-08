"""
atomic_integrals.py - 优化的原子积分计算模块

使用向量化计算和scipy.integrate.simpson来加速积分计算
包含完整的多极展开电子排斥积分计算
"""

import numpy as np
from scipy.integrate import simpson
from typing import Dict, List, Tuple, Optional
import time
from functools import lru_cache
import math
import logging
# 尝试使用sympy的精确计算
from sympy.physics.wigner import wigner_3j
logging.basicConfig(level=logging.DEBUG)
class AtomicIntegrals:
    """
    原子积分计算类 - 完整版本
    
    使用向量化计算和scipy高效积分函数来计算原子轨道间的各种积分：
    - 重叠积分
    - 动能积分  
    - 核吸引积分
    - 电子排斥积分（支持完整多极展开）
    """
    
    def __init__(self, basis_set, eri_method='simplified', max_multipole=2):
        """
        初始化积分计算器
        
        Args:
            basis_set: BasisSet对象，包含原子轨道基组信息
            eri_method: ERI计算方法 ('simplified', 'complete', 'efficient')
            max_multipole: 多极展开的最大k值（当eri_method='efficient'时使用）
        """
        self.basis = basis_set
        self.n_basis = basis_set.get_orbital_count()
        self.eri_method = eri_method
        self.max_multipole = max_multipole
        
        # 预计算所有需要的数据以加速后续计算
        self._precompute_orbital_data()
        self._setup_integration_grids()
        self._precompute_derivatives()
        
        # 如果使用完整或高效ERI方法，预计算Gaunt系数
        if eri_method in ['complete', 'efficient']:
            self._precompute_gaunt_coefficients()
        
        print(f"积分计算器初始化完成 - 基组大小: {self.n_basis}, ERI方法: {eri_method}")
    
    def _precompute_orbital_data(self):
        """预计算轨道数据矩阵"""
        # 径向函数矩阵 [n_basis, n_grid]
        self.radial_functions = np.array([orb.radial_function for orb in self.basis.orbitals])
        
        # 量子数数组
        self.orbital_l = np.array([orb.l for orb in self.basis.orbitals], dtype=np.int32)
        self.orbital_m = np.array([orb.m for orb in self.basis.orbitals], dtype=np.int32)
        self.orbital_n = np.array([orb.n for orb in self.basis.orbitals], dtype=np.int32)
        
        # 轨道名称
        self.orbital_names = [orb.orbital_name for orb in self.basis.orbitals]
    
    def _setup_integration_grids(self):
        """设置积分网格"""
        self.r_grid = self.basis.r_grid
        self.dr = self.r_grid[1] - self.r_grid[0] if len(self.r_grid) > 1 else 0.01
        self.n_grid = len(self.r_grid)
        
        # 预计算常用的网格幂次和安全版本
        self.r_squared = self.r_grid**2
        self.r_safe = np.where(self.r_grid > 1e-12, self.r_grid, 1e-12)
        
        # 角动量选择矩阵：只有l和m都相同的轨道对才有非零积分
        self.angular_selection_matrix = (
            (self.orbital_l[:, np.newaxis] == self.orbital_l[np.newaxis, :]) & 
            (self.orbital_m[:, np.newaxis] == self.orbital_m[np.newaxis, :])
        )
    
    def _precompute_derivatives(self):
        """预计算径向函数的导数（用于动能积分）"""
        self.radial_first_derivatives = np.array([
            np.gradient(R, self.dr) for R in self.radial_functions
        ])
        
        self.radial_second_derivatives = np.array([
            np.gradient(dR, self.dr) for dR in self.radial_first_derivatives
        ])
    
    def _precompute_gaunt_coefficients(self):
        """预计算Gaunt系数"""
        print("预计算Gaunt系数...")
        
        max_l = np.max(self.orbital_l)
        if self.eri_method == 'complete':
            max_k = 2 * max_l
        else:  # efficient
            max_k = min(self.max_multipole, 2 * max_l)
        
        self.gaunt_cache = {}
        self.zero_gaunt_integrals = set()  # 记录所有Gaunt系数都为零的积分
        
        total_combinations = 0
        computed_combinations = 0
        
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sig in range(self.n_basis):
                        total_combinations += 1
                        l1, m1 = self.orbital_l[mu], self.orbital_m[mu]
                        l2, m2 = self.orbital_l[nu], self.orbital_m[nu]
                        l3, m3 = self.orbital_l[lam], self.orbital_m[lam]
                        l4, m4 = self.orbital_l[sig], self.orbital_m[sig]
                        
                        has_nonzero_gaunt = False
                        
                        # 计算这个轨道组合的所有有效k值的Gaunt系数
                        for k in range(max_k + 1):
                            for mk in range(-k, k + 1):
                                gaunt_coeff = self._compute_gaunt_coefficient_product(
                                    l1, m1, l2, m2, l3, m3, l4, m4, k, mk
                                )
                                if abs(gaunt_coeff) > 1e-12:
                                    key = (mu, nu, lam, sig, k, mk)
                                    self.gaunt_cache[key] = gaunt_coeff
                                    has_nonzero_gaunt = True
                        
                        if has_nonzero_gaunt:
                            computed_combinations += 1
                        else:
                            # 记录这个组合没有非零Gaunt系数
                            self.zero_gaunt_integrals.add((mu, nu, lam, sig))
        
        logging.debug(f"Gaunt系数预计算完成：")
        logging.debug(f"  总轨道组合: {total_combinations}")
        logging.debug(f"  有效组合: {computed_combinations}")
        logging.debug(f"  缓存系数数量: {len(self.gaunt_cache)}")
        logging.debug(f"  零积分组合: {len(self.zero_gaunt_integrals)}")
    
    def _compute_gaunt_coefficient_product(self, l1, m1, l2, m2, l3, m3, l4, m4, k, mk):
        """
        计算两个Gaunt系数的乘积：
        G(l1,m1; l2,m2; k,mk) × G(l3,m3; l4,m4; k,-mk)
        """
        # 检查选择定则
        if not self._check_gaunt_selection_rules(l1, m1, l2, m2, k, mk):
            return 0.0
        if not self._check_gaunt_selection_rules(l3, m3, l4, m4, k, -mk):
            return 0.0
        
        # 计算两个Gaunt系数
        gaunt1 = self._compute_single_gaunt_coefficient(l1, m1, l2, m2, k, mk)
        gaunt2 = self._compute_single_gaunt_coefficient(l3, m3, l4, m4, k, -mk)
        
        return gaunt1 * gaunt2
    
    def _check_gaunt_selection_rules(self, l1, m1, l2, m2, l3, m3):
        """检查Gaunt系数的选择定则"""
        # 磁量子数守恒
        if m1 + m2 + m3 != 0:
            return False
        
        # 三角不等式
        if abs(l1 - l2) > l3 or l1 + l2 < l3:
            return False
        
        # 宇称选择定则
        if (l1 + l2 + l3) % 2 != 0:
            return False
        
        return True
    
    @lru_cache(maxsize=1000)
    def _compute_single_gaunt_coefficient(self, l1, m1, l2, m2, l3, m3):
        """计算单个Gaunt系数"""
        if not self._check_gaunt_selection_rules(l1, m1, l2, m2, l3, m3):
            return 0.0
            
        coeff = math.sqrt((2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4*math.pi))

        if int(l1)!=l1 or int(l2)!=l2 or int(l3)!=l3 or int(m1)!=m1 or int(m2)!=m2 or int(m3)!=m3:
            logging.warning(f"gaunt系数遇到输入非整数值: l1={l1}, l2={l2}, l3={l3}, m1={m1}, m2={m2}, m3={m3}")
        else:
            l1=int(l1); l2=int(l2); l3=int(l3); m1=int(m1); m2=int(m2); m3=int(m3)
        
   
        w1 = float(wigner_3j(l1, l2, l3, 0, 0, 0))
        w2 = float(wigner_3j(l1, l2, l3, m1, m2, m3))
        
        return coeff * w1 * w2 * ((-1)**m3)
            
    def _approximate_gaunt_coefficient(self, l1, m1, l2, m2, l3, m3):
        """Gaunt系数的近似计算"""
        # 最简单的情况
        if l1 == l2 == l3 == 0:
            return 1.0 / math.sqrt(4*math.pi)
        
        # s轨道与其他轨道的重叠
        if l3 == 0 and m3 == 0:
            if l1 == l2 and m1 == m2:
                return math.sqrt((2*l1 + 1) / (4*math.pi))
        
        return 0.0
    
    def compute_overlap_matrix(self) -> np.ndarray:
        """
        计算重叠积分矩阵
        
        S_μν = ⟨φ_μ|φ_ν⟩ = ∫ R_μ(r) R_ν(r) r² dr
        
        Returns:
            np.ndarray: 重叠矩阵 [n_basis, n_basis]
        """
        print("计算重叠积分矩阵...")
        
        # 构建被积函数矩阵 [n_basis, n_basis, n_grid]
        integrand_matrix = (
            self.radial_functions[:, np.newaxis, :] * 
            self.radial_functions[np.newaxis, :, :] * 
            self.r_squared[np.newaxis, np.newaxis, :]
        )
        
        S = np.array([
            [simpson(integrand_matrix[mu, nu], x=self.r_grid) 
             for nu in range(self.n_basis)]
            for mu in range(self.n_basis)
        ])
        
        # 应用角动量选择规则
        return S * self.angular_selection_matrix
    
    def compute_kinetic_matrix(self) -> np.ndarray:
        """
        计算动能积分矩阵
        
        T_μν = ⟨φ_μ|-½∇²|φ_ν⟩ = -½ ∫ R_μ(r) [d²R_ν/dr² + (2/r)dR_ν/dr - l_ν(l_ν+1)R_ν/r²] r² dr
        
        Returns:
            np.ndarray: 动能矩阵 [n_basis, n_basis]
        """
        print("计算动能积分矩阵...")
        
        T = np.zeros((self.n_basis, self.n_basis))
        
        # 对于每个轨道对计算动能积分
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                if self.angular_selection_matrix[mu, nu]:
                    l_nu = self.orbital_l[nu]
                    
                    # 动能算符作用在径向波函数上
                    kinetic_operator = (
                        self.radial_second_derivatives[nu] + 
                        (2.0 / self.r_safe) * self.radial_first_derivatives[nu] - 
                        l_nu * (l_nu + 1) * self.radial_functions[nu] / (self.r_safe**2)
                    )
                    
                    # 计算积分
                    integrand = -0.5 * self.radial_functions[mu] * kinetic_operator * self.r_squared
                    T[mu, nu] = simpson(integrand, x=self.r_grid)
        
        return T
    
    def compute_nuclear_attraction_matrix(self, nuclear_charge: int) -> np.ndarray:
        """
        计算核吸引积分矩阵
        
        V_μν = ⟨φ_μ|-Z/r|φ_ν⟩ = -Z ∫ R_μ(r) R_ν(r) r dr
        
        Args:
            nuclear_charge: 核电荷数
            
        Returns:
            np.ndarray: 核吸引矩阵 [n_basis, n_basis]
        """
        print("计算核吸引积分矩阵...")
        
        # 构建被积函数矩阵：R_μ(r) * R_ν(r) * r  (注意：r²/r = r)
        integrand_matrix = (
            self.radial_functions[:, np.newaxis, :] * 
            self.radial_functions[np.newaxis, :, :] * 
            self.r_safe[np.newaxis, np.newaxis, :]
        )
        
        # 批量计算积分
        V = np.array([
            [simpson(integrand_matrix[mu, nu], x=self.r_grid) 
             for nu in range(self.n_basis)]
            for mu in range(self.n_basis)
        ])
        
        # 应用核电荷和角动量选择规则
        return -nuclear_charge * V * self.angular_selection_matrix
    
    def compute_electron_repulsion_integrals(self) -> np.ndarray:
        """
        计算电子排斥积分
        
        (μν|λσ) = ⟨φ_μφ_ν|1/r₁₂|φ_λφ_σ⟩
        
        根据初始化时选择的方法计算：
        - 'simplified': 只计算垄断项k=0
        - 'efficient': 计算到指定的最大k值
        - 'complete': 完整的多极展开
        
        Returns:
            np.ndarray: 电子排斥积分张量 [n_basis, n_basis, n_basis, n_basis]
        """
        if self.eri_method == 'simplified':
            print("计算电子排斥积分（简化版 - 仅k=0项）...")
            return self._compute_monopole_eri()
        elif self.eri_method == 'efficient':
            print(f"计算电子排斥积分（高效版 - k=0到{self.max_multipole}）...")
            return self._compute_efficient_eri()
        elif self.eri_method == 'complete':
            print("计算电子排斥积分（完整版 - 所有多极项）...")
            return self._compute_complete_eri()
        else:
            raise ValueError(f"未知的ERI计算方法: {self.eri_method}")
    
    def _compute_monopole_eri(self) -> np.ndarray:
        """计算垄断项（k=0）电子排斥积分"""
        eri = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))
        
        print("  计算垄断项...")
        
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sig in range(self.n_basis):
                        # 简化的角动量选择规则（垄断项）
                        if self._check_monopole_selection_rules(mu, nu, lam, sig):
                            eri[mu, nu, lam, sig] = self._compute_monopole_integral(mu, nu, lam, sig)
        
        return eri
    
    def _check_monopole_selection_rules(self, mu: int, nu: int, lam: int, sig: int) -> bool:
        """检查垄断项的角动量选择规则"""
        return (self.orbital_l[mu] == self.orbital_l[nu] and 
                self.orbital_m[mu] == self.orbital_m[nu] and
                self.orbital_l[lam] == self.orbital_l[sig] and 
                self.orbital_m[lam] == self.orbital_m[sig])
    
    def _compute_monopole_integral(self, mu: int, nu: int, lam: int, sig: int) -> float:
        """计算单个垄断项积分"""
        # 获取径向密度函数
        rho1 = self.radial_functions[mu] * self.radial_functions[nu] * self.r_squared
        rho2 = self.radial_functions[lam] * self.radial_functions[sig] * self.r_squared
        
        # 简化计算：使用粗网格以提高速度
        step = max(1, self.n_grid // 50)
        r_coarse = self.r_grid[::step]
        rho1_coarse = rho1[::step]
        rho2_coarse = rho2[::step]
        
        integral = 0.0
        dr_coarse = self.dr * step
        
        # 双重积分：∫∫ ρ₁(r₁) ρ₂(r₂) / max(r₁,r₂) dr₁ dr₂
        for i, r1 in enumerate(r_coarse):
            for j, r2 in enumerate(r_coarse):
                r_max = max(r1, r2)
                if r_max > 1e-12:
                    integral += rho1_coarse[i] * rho2_coarse[j] / r_max
        
        return integral * dr_coarse**2
    
    def _compute_efficient_eri(self) -> np.ndarray:
        """计算高效版本的ERI（到指定的最大k值）"""
        eri = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))
        
        computed_integrals = 0
        total_integrals = 0
        
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sig in range(self.n_basis):
                        total_integrals += 1
                        
                        # 检查是否所有Gaunt系数都为零
                        if (mu, nu, lam, sig) in self.zero_gaunt_integrals:
                            continue
                        
                        # 计算这个积分
                        integral_value = self._compute_eri_element_up_to_k(mu, nu, lam, sig, self.max_multipole)
                        eri[mu, nu, lam, sig] = integral_value
                        
                        if abs(integral_value) > 1e-12:
                            computed_integrals += 1
        
        print(f"  完成：{computed_integrals}/{total_integrals} 个非零积分")
        return eri
    
    def _compute_complete_eri(self) -> np.ndarray:
        """计算完整的ERI（所有多极项）"""
        max_l = np.max(self.orbital_l)
        max_k = 2 * max_l
        
        return self._compute_eri_element_up_to_k_all(max_k)
    
    def _compute_eri_element_up_to_k(self, mu, nu, lam, sig, max_k):
        """计算单个ERI元素（到指定的k值）"""
        total_integral = 0.0
        
        # 对所有k和mk求和
        for k in range(max_k + 1):
            for mk in range(-k, k + 1):
                # 获取Gaunt系数
                gaunt_key = (mu, nu, lam, sig, k, mk)
                if gaunt_key not in self.gaunt_cache:
                    continue
                
                gaunt_coeff = self.gaunt_cache[gaunt_key]
                if abs(gaunt_coeff) < 1e-12:
                    continue
                
                # 计算径向积分
                radial_integral = self._compute_radial_multipole_integral(mu, nu, lam, sig, k)
                
                # 多极展开系数
                multipole_coeff = (4.0 * math.pi) / (2.0 * k + 1.0)
                
                # 累加贡献
                contribution = multipole_coeff * gaunt_coeff * radial_integral
                total_integral += contribution
        
        return total_integral
    
    def _compute_eri_element_up_to_k_all(self, max_k):
        """计算所有ERI元素（完整版本）"""
        eri = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))
        
        computed_integrals = 0
        total_integrals = 0
        
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sig in range(self.n_basis):
                        total_integrals += 1
                        
                        if (mu, nu, lam, sig) in self.zero_gaunt_integrals:
                            continue
                        
                        integral_value = self._compute_eri_element_up_to_k(mu, nu, lam, sig, max_k)
                        eri[mu, nu, lam, sig] = integral_value
                        
                        if abs(integral_value) > 1e-12:
                            computed_integrals += 1
        
        print(f"  完成：{computed_integrals}/{total_integrals} 个非零积分")
        return eri
    
    def _compute_radial_multipole_integral(self, mu, nu, lam, sig, k):
        """计算k阶多极展开的径向积分"""
        R1, R2 = self.radial_functions[mu], self.radial_functions[nu]
        R3, R4 = self.radial_functions[lam], self.radial_functions[sig]
        
        # 预计算径向密度
        rho1 = R1 * R2 * self.r_squared
        rho2 = R3 * R4 * self.r_squared
        
        # 使用自适应网格
        step = max(1, self.n_grid // (50 + 10*k))  # k越大，网格越密
        
        integral = 0.0
        
        for i in range(0, self.n_grid, step):
            r1 = self.r_grid[i]
            for j in range(0, self.n_grid, step):
                r2 = self.r_grid[j]
                
                if r1 < 1e-12 or r2 < 1e-12:
                    continue
                
                # 计算多极因子
                r_min = min(r1, r2)
                r_max = max(r1, r2)
                
                if r_max > 1e-12:
                    multipole_factor = (r_min**k) / (r_max**(k+1))
                    integrand = rho1[i] * rho2[j] * multipole_factor
                    integral += integrand
        
        return integral * (self.dr * step)**2
    
    def analyze_eri_convergence(self) -> Dict:
        """分析ERI多极展开的收敛性"""
        if self.eri_method == 'simplified':
            print("当前使用简化版ERI，无法进行收敛性分析")
            return {}
        
        print("=== ERI多极展开收敛性分析 ===")
        
        max_l = np.max(self.orbital_l)
        max_k_analysis = min(2 * max_l, 4)  # 限制分析范围
        
        eri_by_k = {}
        cumulative_eri = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))
        
        for k in range(max_k_analysis + 1):
            print(f"分析k={k}项...")
            
            # 计算k项的贡献
            eri_k = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))
            
            for mu in range(self.n_basis):
                for nu in range(self.n_basis):
                    for lam in range(self.n_basis):
                        for sig in range(self.n_basis):
                            if (mu, nu, lam, sig) not in self.zero_gaunt_integrals:
                                # 只计算k项的贡献
                                k_contribution = self._compute_single_k_contribution(mu, nu, lam, sig, k)
                                eri_k[mu, nu, lam, sig] = k_contribution
                                cumulative_eri[mu, nu, lam, sig] += k_contribution
            
            eri_by_k[k] = eri_k
            
            # 分析统计
            total_contrib = np.sum(np.abs(eri_k))
            max_contrib = np.max(np.abs(eri_k))
            nonzero_count = np.sum(np.abs(eri_k) > 1e-12)
            
            print(f"  k={k}: 总贡献={total_contrib:.6f}, 最大值={max_contrib:.6f}, 非零={nonzero_count}")
        
        return {
            'by_multipole': eri_by_k,
            'cumulative': cumulative_eri,
            'convergence_analysis': self._analyze_convergence_pattern(eri_by_k)
        }
    
    def _compute_single_k_contribution(self, mu, nu, lam, sig, k):
        """计算单个k值的贡献"""
        k_integral = 0.0
        
        for mk in range(-k, k + 1):
            gaunt_key = (mu, nu, lam, sig, k, mk)
            if gaunt_key not in self.gaunt_cache:
                continue
            
            gaunt_coeff = self.gaunt_cache[gaunt_key]
            if abs(gaunt_coeff) < 1e-12:
                continue
            
            radial_integral = self._compute_radial_multipole_integral(mu, nu, lam, sig, k)
            multipole_coeff = (4.0 * math.pi) / (2.0 * k + 1.0)
            
            k_integral += multipole_coeff * gaunt_coeff * radial_integral
        
        return k_integral
    
    def _analyze_convergence_pattern(self, eri_by_k):
        """分析收敛模式"""
        analysis = {}
        
        k0_total = np.sum(np.abs(eri_by_k[0])) if 0 in eri_by_k else 1.0
        
        for k, eri_k in eri_by_k.items():
            total_k = np.sum(np.abs(eri_k))
            relative_contrib = total_k / k0_total if k0_total > 0 else 0
            
            analysis[k] = {
                'absolute_contribution': total_k,
                'relative_to_monopole': relative_contrib,
                'max_element': np.max(np.abs(eri_k)),
                'nonzero_count': np.sum(np.abs(eri_k) > 1e-12)
            }
        
        return analysis
    
    def compute_core_hamiltonian(self, nuclear_charge: int) -> np.ndarray:
        """
        计算核心哈密顿矩阵
        
        H_core = T + V_nuclear
        
        Args:
            nuclear_charge: 核电荷数
            
        Returns:
            np.ndarray: 核心哈密顿矩阵
        """
        T = self.compute_kinetic_matrix()
        V = self.compute_nuclear_attraction_matrix(nuclear_charge)
        return T + V
    
    def compute_all_integrals(self, nuclear_charge: int) -> Dict[str, np.ndarray]:
        """
        计算所有需要的积分
        
        Args:
            nuclear_charge: 核电荷数
            
        Returns:
            Dict: 包含所有积分矩阵的字典
        """
        print(f"开始计算所有积分 (核电荷 = {nuclear_charge}, ERI方法 = {self.eri_method})...")
        start_time = time.time()
        
        integrals = {}
        
        # 计算单电子积分
        integrals['overlap'] = self.compute_overlap_matrix()
        integrals['kinetic'] = self.compute_kinetic_matrix()
        integrals['nuclear'] = self.compute_nuclear_attraction_matrix(nuclear_charge)
        integrals['core_hamiltonian'] = integrals['kinetic'] + integrals['nuclear']
        
        # 计算双电子积分
        integrals['electron_repulsion'] = self.compute_electron_repulsion_integrals()
        
        total_time = time.time() - start_time
        print(f"积分计算完成，耗时: {total_time:.3f}s")
        
        return integrals
    
    def print_integral_summary(self, integrals: Dict[str, np.ndarray]):
        """打印积分结果摘要"""
        print("\n=== 积分计算结果摘要 ===")
        print(f"基组大小: {self.n_basis}x{self.n_basis}")
        print(f"ERI计算方法: {self.eri_method}")
        
        # 重叠矩阵
        S = integrals['overlap']
        print(f"\n重叠矩阵:")
        print(f"  对角元范围: {np.min(np.diag(S)):.6f} ~ {np.max(np.diag(S)):.6f}")
        print(f"  非对角元最大值: {np.max(np.abs(S - np.diag(np.diag(S)))):.6f}")
        
        # 核心哈密顿
        H_core = integrals['core_hamiltonian']
        eigenvals = np.linalg.eigvals(H_core)
        eigenvals.sort()
        print(f"\n核心哈密顿本征值:")
        for i, val in enumerate(eigenvals):
            orbital_name = self.orbital_names[i] if i < len(self.orbital_names) else f"轨道{i+1}"
            print(f"  {orbital_name}: {val:.6f} Hartree")
        
        # 电子排斥积分统计
        eri = integrals['electron_repulsion']
        nonzero_eri = eri[np.abs(eri) > 1e-10]
        print(f"\n电子排斥积分:")
        print(f"  非零元素数: {len(nonzero_eri)} / {eri.size}")
        if len(nonzero_eri) > 0:
            print(f"  数值范围: {np.min(nonzero_eri):.6f} ~ {np.max(nonzero_eri):.6f}")

# 便利函数
def calculate_atomic_integrals(basis_set, nuclear_charge: int, eri_method='simplified', max_multipole=2) -> Dict[str, np.ndarray]:
    """
    便利函数：计算原子的所有积分
    
    Args:
        basis_set: 原子轨道基组
        nuclear_charge: 核电荷数
        eri_method: ERI计算方法 ('simplified', 'efficient', 'complete')
        max_multipole: 最大多极展开项数（用于efficient方法）
        
    Returns:
        Dict: 所有积分矩阵
    """
    integral_calc = AtomicIntegrals(basis_set, eri_method=eri_method, max_multipole=max_multipole)
    integrals = integral_calc.compute_all_integrals(nuclear_charge)
    integral_calc.print_integral_summary(integrals)
    return integrals

def compare_eri_methods(basis_set, nuclear_charge: int):
    """比较不同ERI计算方法"""
    print("=== ERI方法对比 ===")
    
    methods = ['simplified', 'efficient', 'complete']
    results = {}
    times = {}
    
    for method in methods:
        try:
            print(f"\n测试方法: {method}")
            start_time = time.time()
            
            if method == 'efficient':
                calc = AtomicIntegrals(basis_set, eri_method=method, max_multipole=2)
            else:
                calc = AtomicIntegrals(basis_set, eri_method=method)
            
            eri = calc.compute_electron_repulsion_integrals()
            
            elapsed = time.time() - start_time
            times[method] = elapsed
            results[method] = eri
            
            nonzero_count = np.sum(np.abs(eri) > 1e-12)
            max_val = np.max(np.abs(eri))
            
            print(f"  耗时: {elapsed:.3f}s")
            print(f"  非零元素: {nonzero_count}")
            print(f"  最大值: {max_val:.6f}")
            
        except Exception as e:
            print(f"  方法 {method} 失败: {e}")
    
    # 比较结果差异
    if len(results) > 1:
        print(f"\n=== 结果对比 ===")
        ref_method = 'simplified'
        if ref_method in results:
            ref_result = results[ref_method]
            
            for method, result in results.items():
                if method != ref_method:
                    diff = np.max(np.abs(result - ref_result))
                    rel_diff = diff / np.max(np.abs(ref_result)) if np.max(np.abs(ref_result)) > 0 else 0
                    speedup = times[ref_method] / times[method] if method in times else 1
                    
                    print(f"{method} vs {ref_method}:")
                    print(f"  最大差异: {diff:.6f}")
                    print(f"  相对差异: {rel_diff*100:.2f}%")
                    print(f"  速度比: {speedup:.2f}x")
    
    return results
