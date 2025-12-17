"""
atomic_integrals.py - 优化的原子积分计算模块
使用向量化计算和scipy.integrate.simpson来加速积分计算
包含完整的多极展开电子排斥积分计算
"""

import numpy as np
from scipy.integrate import simpson
from scipy.integrate import cumulative_trapezoid
from typing import Dict, List, Tuple, Optional
import time
from functools import lru_cache
import math
import logging
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import gaunt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from .pseudo import Pseudopotential
from .basis_set import BasisSet, AtomicOrbital

class AtomicIntegrals:
    """
    原子积分计算类 - 完整版本
    
    使用向量化计算和scipy高效积分函数来计算原子轨道间的各种积分：
    - 重叠积分
    - 动能积分  
    - 核吸引积分
    - 电子排斥积分（支持完整多极展开）
    """

    def __init__(self, basis_set: BasisSet, nuclear_charge: int=None, pseudo: Pseudopotential=None,max_multipole=-1,real_basis=False):
        """
        初始化积分计算器
        
        Args:
            basis_set: BasisSet对象，包含原子轨道基组信息
            max_multipole: 多极展开的最大k值，k=-1表示完整展开
        """
        self.basis = basis_set
        self.n_basis = basis_set.get_orbital_count()
        self.max_multipole = max_multipole
        self.real_basis=real_basis
        if nuclear_charge is None and pseudo is None:
            raise ValueError("必须提供核电荷数或赝势")

        if nuclear_charge is not None:
            self.nuclear_charge = nuclear_charge
        else:
            self.nuclear_charge = int(pseudo.z_valence)
        self.pseudo = pseudo
        # 预计算所有需要的数据以加速后续计算
        self._precompute_orbital_data()
        self._setup_integration_grids()
        self._precompute_derivatives()
        self._precompute_gaunt_coefficients()
        
    
        logging.info(f"积分计算器初始化完成 - 基组大小: {self.n_basis}, ERI max multipole: {self.max_multipole}")
    
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
        # self.r_squared = self.r_grid**2
        self.r_safe = np.where(self.r_grid > 1e-12, self.r_grid, 1e-12)
        self.r_inv_squared = 1.0 / (self.r_safe**2)
        
        #TODO:这里可以优化到S积分计算时
        # 角动量选择矩阵：只有l和m都相同的轨道对才有非零积分
        self.angular_selection_matrix = (
            (self.orbital_l[:, np.newaxis] == self.orbital_l[np.newaxis, :]) & 
            (self.orbital_m[:, np.newaxis] == self.orbital_m[np.newaxis, :])
        )


    def _precompute_derivatives(self):
        """预计算径向函数的导数"""
        self.radial_first_derivatives = np.array([
            np.gradient(R, self.dr) for R in self.radial_functions
        ])
        
    
    def _precompute_gaunt_coefficients(self):
        """预计算Gaunt系数"""
        logging.info("计算Gaunt系数...")
        
        max_l = np.max(self.orbital_l)
  
        max_k = min(self.max_multipole, 2 * max_l) if self.max_multipole>=0 else 2 * max_l
        
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
                                if self.real_basis:
                                    gaunt_coeff = self._compute_gaunt_coefficient_product_real(
                                    l1, m1, l2, m2, l3, m3, l4, m4, k, mk
                                )
                                else:
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
        if not self._check_selection_rules(l1, -m1, l2, m2, k, -mk):
            return 0.0
        if not self._check_selection_rules(l3, -m3, l4, m4, k, mk):
            return 0.0
        
        # 计算两个Gaunt系数
        gaunt1 = self._compute_single_gaunt_coefficient(l1, -m1, l2, m2, k, -mk)
        gaunt2 = self._compute_single_gaunt_coefficient(l3, -m3, l4, m4, k, mk)
        
        return gaunt1 * gaunt2*((-1.0)**(m3+mk+m1))
    
    def _compute_gaunt_coefficient_product_real(self, l1, m1, l2, m2, l3, m3, l4, m4, k, mk):
        """
        计算 ERI 所需的 Gaunt 系数乘积 (实球谐函数版本)。
        公式：Sum_{q=-k}^{k} <l1 m1| l2 m2 | k q> * <l3 m3| l4 m4 | k q>
        mk 参数在循环外部传入，代表多极矩 k 的分量 q
        """
        # 左边: Integral(S_{l1,m1} S_{l2,m2} S_{k,mk})
        gaunt1 = self._compute_single_gaunt_coefficient_real(l1, m1, l2, m2, k, mk)
        
        if abs(gaunt1) < 1e-14:
            return 0.0
            
        # 右边: Integral(S_{l3,m3} S_{l4,m4} S_{k,mk})
        # 注意：实数展开中，中间项 S_{k,mk} 在左右两边是完全相同的实函数
        gaunt2 = self._compute_single_gaunt_coefficient_real(l3, m3, l4, m4, k, mk)
        
        return gaunt1 * gaunt2
    
    def _check_selection_rules(self, l1, m1, l2, m2, l3, m3):
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
    
    def _get_gaunt_coeffs_real(self, l, m):
        """
        根据 ABACUS 定义，获取实数球谐函数 S_{lm} 对应的复数球谐函数 Y_{lk} 展开系数。
        返回列表: [(k_index, coefficient), ...]
        """
        if m == 0:
            return [(0, 1.0)]
        
        # 预计算常数
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        
        if m > 0:
            # S_{l,m} = sqrt(2) Re(Y_{l,m})
            # = (1/sqrt(2)) * (Y_{l,m} + (-1)^m Y_{l,-m})
            phase = (-1)**m
            return [
                (m, inv_sqrt2), 
                (-m, phase * inv_sqrt2)
            ]
        else:
            # m < 0
            # S_{l,m} = sqrt(2) Im(Y_{l,|m|})
            # = (1/i*sqrt(2)) * (Y_{l,|m|} - (-1)^|m| Y_{l,-|m|})
            # = (-i/sqrt(2)) * Y_{l,|m|} + (i*(-1)^|m|/sqrt(2)) * Y_{l,-|m|}
            abs_m = abs(m)
            phase = (-1)**abs_m
            return [
                (abs_m, -1j * inv_sqrt2),
                (-abs_m, 1j * phase * inv_sqrt2)
            ]

    @lru_cache(maxsize=100000)
    def _compute_single_gaunt_coefficient_real(self, l1, m1, l2, m2, l3, m3):
        """
        计算实数球谐函数的 Gaunt 系数 (兼容 ABACUS)
        Integral(S_{l1,m1} S_{l2,m2} S_{l3,m3})
        """
        # 1. 获取三个轨道的展开系数
        coeffs1 = self._get_gaunt_coeffs_real(l1, m1)
        coeffs2 = self._get_gaunt_coeffs_real(l2, m2)
        coeffs3 = self._get_gaunt_coeffs_real(l3, m3)
        
        total_val = 0.0 + 0.0j
        
        # 2. 遍历所有组合进行线性叠加
        for (k1, c1) in coeffs1:
            for (k2, c2) in coeffs2:
                for (k3, c3) in coeffs3:
                    # 快速筛选：m量子数之和必须为0，否则积分为0
                    if k1 + k2 + k3 != 0:
                        continue
                        
                    # 调用 sympy 计算复数积分
                    # gaunt 返回的是 integral(Y_l1k1 * Y_l2k2 * Y_l3k3)
                    val = float(gaunt(l1, l2, l3, k1, k2, k3))
                    
                    if abs(val) > 1e-12:
                        total_val += c1 * c2 * c3 * val
        
        # 3. 结果必须是实数（数值误差导致的微小虚部应被忽略）
        if abs(total_val.imag) > 1e-10:
            logging.warning(f"Warning: Real Gaunt coeff has imaginary part: {total_val}")
        
        return total_val.real

    
    @lru_cache(maxsize=1000)
    def _compute_single_gaunt_coefficient(self, l1, m1, l2, m2, l3, m3):
        """计算单个Gaunt系数"""
        if not self._check_selection_rules(l1, m1, l2, m2, l3, m3):
            return 0.0
    
        if int(l1)!=l1 or int(l2)!=l2 or int(l3)!=l3 or int(m1)!=m1 or int(m2)!=m2 or int(m3)!=m3:
            logging.warning(f"gaunt系数遇到输入非整数值: l1={l1}, l2={l2}, l3={l3}, m1={m1}, m2={m2}, m3={m3}")
        else:
            l1=int(l1); l2=int(l2); l3=int(l3); m1=int(m1); m2=int(m2); m3=int(m3)
        
        return float(gaunt(l1, l2, l3,m1,m2, m3))
            
    

    
    def compute_overlap_matrix(self) -> np.ndarray:
        """
        计算重叠积分矩阵
        
        S_μν = ⟨φ_μ|φ_ν⟩ = ∫ R_μ(r) R_ν(r) r² dr
        
        Returns:
            np.ndarray: 重叠矩阵 [n_basis, n_basis]
        """
        logging.info("计算重叠积分矩阵...")
        
        # 构建被积函数矩阵 [n_basis, n_basis, n_grid]
        integrand_matrix = (
            self.radial_functions[:, np.newaxis, :] * 
            self.radial_functions[np.newaxis, :, :]
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
        
        T_μν = ⟨φ_μ|-½∇²|φ_ν⟩ = -½ ∫ R_μ(r) [d²R_ν/dr² + (2/r)dR_ν/dr - l_ν(l_ν+1)R_ν/r²] r² dr = ∫ ½ drR_ν/dr drR_μ/dr - l_ν(l_ν+1)R_ν R_μdr
        
        Returns:
            np.ndarray: 动能矩阵 [n_basis, n_basis]
        """
        logging.info("计算动能积分矩阵...")
        
        T = np.zeros((self.n_basis, self.n_basis))
        
        # 对于每个轨道对计算动能积分
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                if self.angular_selection_matrix[mu, nu]:
                    l_nu = self.orbital_l[nu]

                    kinetic_integrand=(self.radial_first_derivatives[mu]*self.radial_first_derivatives[nu]*0.5+
                                      0.5*l_nu * (l_nu + 1) *self.radial_functions[mu]* self.radial_functions[nu]*self.r_inv_squared
                                      )
                    T[mu, nu] = simpson(kinetic_integrand, x=self.r_grid)
                    # # 动能算符作用在径向波函数上
                    # kinetic_operator = (
                    #     self.radial_second_derivatives[nu] + 
                    #     (2.0 / self.r_safe) * self.radial_first_derivatives[nu] - 
                    #     l_nu * (l_nu + 1) * self.radial_functions[nu] / (self.r_safe**2)
                    # )
                    
                    # # 计算积分
                    # integrand = -0.5 * self.radial_functions[mu] * kinetic_operator
                    # T[mu, nu] = simpson(integrand, x=self.r_grid)
        return T
    
    def compute_nuclear_attraction_matrix(self) -> np.ndarray:
        """
        计算核吸引积分矩阵
        
        V_μν = ⟨φ_μ|-Z/r|φ_ν⟩ = -Z ∫ R_μ(r) R_ν(r) r dr
        
        Args:
            nuclear_charge: 核电荷数
            
        Returns:
            np.ndarray: 核吸引矩阵 [n_basis, n_basis]
        """
        logging.info("计算核吸引积分矩阵...")
        
        if self.pseudo is None:
            return self._compute_nuclear_attraction_matrix_all_electrons(self.nuclear_charge)
        else:
            return self._compute_nuclear_attraction_matrix_pseudo()+self._compute_nuclear_attraction_matrix_pseudo_nonlocal()

    def _compute_nuclear_attraction_matrix_all_electrons(self, nuclear_charge: int) -> np.ndarray:
        """
        计算全电子核吸引积分矩阵
        
        V_μν = ⟨φ_μ|-Z/r|φ_ν⟩ = -Z ∫ R_μ(r) R_ν(r) r dr
        
        Args:
            nuclear_charge: 核电荷数
            
        Returns:
            np.ndarray: 核吸引矩阵 [n_basis, n_basis]
        """
        logging.info("计算核吸引积分矩阵...")
        
        # 构建被积函数矩阵：R_μ(r) * R_ν(r) * r  
        integrand_matrix = (
            self.radial_functions[:, np.newaxis, :] * 
            self.radial_functions[np.newaxis, :, :] 
            /self.r_safe[np.newaxis, np.newaxis, :]
        )
        
        # 批量计算积分
        V = np.array([
            [simpson(integrand_matrix[mu, nu], x=self.r_grid) 
             for nu in range(self.n_basis)]
            for mu in range(self.n_basis)
        ])
        
        # 应用核电荷和角动量选择规则
        return  -nuclear_charge * V * self.angular_selection_matrix
    
    
    
    def _compute_nuclear_attraction_matrix_pseudo(self) -> np.ndarray:
        """
        计算全电子核吸引积分矩阵
        
        V_μν = ∫ R_μ(r)Vlocal(r) R_ν(r) r^2 dr

        Args:
            nuclear_charge: 核电荷数
        Returns:
            np.ndarray: 核吸引矩阵 [n_basis, n_basis]
        """
        logging.info("计算核吸引积分矩阵...")
        
        # 构建被积函数矩阵：R_μ(r)*Vlocal(r)* R_ν(r) * r^2
        integrand_matrix = (
            self.radial_functions[:, np.newaxis, :] * 
            self.radial_functions[np.newaxis, :, :] * 
            self.pseudo.v_local(self.r_grid)[np.newaxis, np.newaxis, :]
        )
        
        # 批量计算积分
        V = np.array([
            [simpson(integrand_matrix[mu, nu], x=self.r_grid) 
             for nu in range(self.n_basis)]
            for mu in range(self.n_basis)
        ])
        
        # 应用核电荷和角动量选择规则
        return V * self.angular_selection_matrix

    
    def _compute_nuclear_attraction_matrix_pseudo_nonlocal(self) -> np.ndarray:
        """
        计算赝势非局域部分的核吸引积分矩阵
        
        V_μν^NL = Σ_ij ⟨φ_μ|β_i⟩ D_ij ⟨β_j|φ_ν⟩
        
        Returns:
            np.ndarray: 非局域核吸引矩阵 [n_basis, n_basis]
        """
        logging.info("计算赝势非局域核吸引积分矩阵...")
        
        
        n_basis = self.n_basis
        n_projectors = len(self.pseudo.beta_projectors)
        
        # S_proj (n_basis x n_projectors) 矩阵，S_{μi} = ⟨φ_μ|β_i⟩
        
        S_proj = np.zeros((n_basis, n_projectors))
        
        # 遍历所有基函数 μ
        for mu in range(n_basis):
            orb_l = self.orbital_l[mu]
            orb_m = self.orbital_m[mu]
            
            # 遍历所有投影函数 i
            for i in range(n_projectors):
                proj = self.pseudo.beta_projectors[i]
                if orb_l == proj.l :
                    # 径向积分: ∫ rR_μ(r) * rβ_i(r) dr
                    integrand = self.radial_functions[mu] * proj.radial_function(self.r_grid)
                    # 将计算结果存入 S_proj 矩阵
                    S_proj[mu, i] = simpson(integrand, x=self.r_grid)


        D_matrix = self.pseudo.d_matrix
        print(D_matrix.shape)
        print(S_proj.shape)
    
        #通过矩阵乘法构建完整的非局域矩阵 V_non_local ---
        # V_NL = S_proj * D * S_proj^T
        # (n_basis, n_proj) @ (n_proj, n_proj) -> (n_basis, n_proj)
        # (n_basis, n_proj) @ (n_proj, n_basis) -> (n_basis, n_basis)
        V_non_local_matrix = S_proj @ D_matrix @ S_proj.T
        
        return V_non_local_matrix

    def _compute_eri_element_k(self, mu, nu, lam, sig, max_k):
        """计算单个ERI元素，到max_k截断"""
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
    
    def _compute_eri_k(self, max_k):
        """计算所有ERI元素"""
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
                        integral_value = self._compute_eri_element_k(mu, nu, lam, sig, max_k)
                        eri[mu, nu, lam, sig] = integral_value
                        
                        if abs(integral_value) > 1e-12:
                            computed_integrals += 1

        logging.debug(f"ERI Integral:maxk={max_k},完成：{computed_integrals}/{total_integrals} 个非零积分")
        return eri
    

    def compute_electron_repulsion_integrals(self) -> np.ndarray:
        """
        计算电子排斥积分
        
        (μν|λσ) = ⟨φ_μφ_ν|1/r₁₂|φ_λφ_σ⟩=(μν|λσ) = Σ_{k=0..∞} Σ_{mk=-k..k} (4π / (2k+1)) * A₁ * A₂ * Rₖ
        Rₖ = ∫∫ R_μ(r₁)R_ν(r₁) * [r_<ᵏ / r_>ᵏ⁺¹] * R_λ(r₂)R_σ(r₂) * r₁²dr₁ * r₂²dr₂
        A₁ = ∫ Y*(lμ,mμ) Y(lν,mν) Y*(k,mk) dΩ₁
        A₂ = ∫ Y*(lλ,mλ) Y(lσ,mσ) Y(k,mk) dΩ₂
        
        - 'simplified': 只计算垄断项k=0
        - 'efficient': 计算到指定的最大k值
        - 'complete': 完整的多极展开

        Returns:
            np.ndarray: 电子排斥积分张量 [n_basis, n_basis, n_basis, n_basis]
        """
        maxk=min(self.max_multipole, 2 * np.max(self.orbital_l)) if self.max_multipole>=0 else 2 * np.max(self.orbital_l)
        return self._compute_eri_k(maxk)



    @lru_cache(maxsize=100000)
    def _compute_radial_multipole_integral(self, mu, nu, lam, sig, k):
        """
        计算k阶多极展开的径向积分
        Rₖ=∫dr₁ ∫dr₂ ρ₁(r₁) [r_<ᵏ / r_>ᵏ⁺¹] ρ₂(r₂)=(1/r₁ᵏ⁺¹) ∫_0^r₁ r₂ᵏ ρ₂(r₂) dr₂ + r₁ᵏ ∫_r₁^∞ [ρ₂(r₂)/r₂ᵏ⁺¹] dr₂
        """
        # 1. 获取径向密度函数 rho(r) = R(r1)R(r2)r^2
        # rho1对应(mu, nu)，rho2对应(lam, sig)
        rho1 = self.radial_functions[mu] * self.radial_functions[nu] 
        rho2 = self.radial_functions[lam] * self.radial_functions[sig] 

        # 使用预先计算的安全r网格，避免除零
        r_safe = self.r_safe
        r_grid = self.r_grid
        
        # I(r1) = (1/r1**(k+1)) * integral_0^r1(r2**k * rho2(r2) dr2) + 
        #         r1**k * integral_r1^inf(rho2(r2) / r2**(k+1) dr2)

        # F(r1) = integral_0^r1(r2**k * rho2(r2) dr2)
        integrand_F = r_safe**k * rho2

        # cumulative_trapezoid返回的数组比输入短1
        F = np.zeros_like(r_grid)
        F[1:] = cumulative_trapezoid(integrand_F, x=r_grid)

      
        # G(r1) = integral_r1^inf(rho2(r2) / r2**(k+1) dr2)
        integrand_G = rho2 / r_safe**(k+1)
        
        G_total = simpson(integrand_G, x=r_grid)
        
        G_cumulative = np.zeros_like(r_grid)
        G_cumulative[1:] = cumulative_trapezoid(integrand_G, x=r_grid)
        
        G = G_total - G_cumulative

        # I = F / r1**(k+1) + r1**k * G
        I = F / r_safe**(k+1) + r_safe**k * G

        #R_k = integral( rho1(r1) * I(r1) dr1 )
        final_integrand = rho1 * I
        
        integral_value = simpson(final_integrand, x=r_grid)

        return integral_value
    
    def analyze_eri_convergence(self) -> Dict:
        """分析ERI多极展开的收敛性"""

        
        logging.info("=== ERI多极展开收敛性分析 ===")
        
        max_l = np.max(self.orbital_l)
        max_k_analysis = min(2 * max_l, 4)  # 限制分析范围
        
        eri_by_k = {}
        cumulative_eri = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))
        
        for k in range(max_k_analysis + 1):
            logging.info(f"分析k={k}项...")
            
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
            
            logging.info(f"  k={k}: 总贡献={total_contrib:.6f}, 最大值={max_contrib:.6f}, 非零={nonzero_count}")
        
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
    
    def compute_core_hamiltonian(self) -> np.ndarray:
        """
        计算核心哈密顿矩阵
        
        H_core = T + V_nuclear
        
        Args:
            nuclear_charge: 核电荷数
            
        Returns:
            np.ndarray: 核心哈密顿矩阵
        """
        T = self.compute_kinetic_matrix()
        V = self.compute_nuclear_attraction_matrix()
        return T + V
    
    def compute_all_integrals(self) -> Dict[str, np.ndarray]:
        """
        计算所有需要的积分
        
        Args:
            nuclear_charge: 核电荷数
            
        Returns:
            Dict: 包含所有积分矩阵的字典
        """
        logging.info(f"开始计算所有积分 (核电荷 = {self.nuclear_charge}, ERI max multipole = {self.max_multipole})...")
        start_time = time.time()
        
        integrals = {}
        
        # 计算单电子积分
        integrals['overlap'] = self.compute_overlap_matrix()
        integrals['kinetic'] = self.compute_kinetic_matrix()
        integrals['nuclear'] = self.compute_nuclear_attraction_matrix()
        integrals['core_hamiltonian'] = integrals['kinetic'] + integrals['nuclear']
        
        # 计算双电子积分
        integrals['electron_repulsion'] = self.compute_electron_repulsion_integrals()
        
        total_time = time.time() - start_time
        logging.info(f"积分计算完成，耗时: {total_time:.3f}s")
        
        return integrals
    
    def print_integral_summary(self, integrals: Dict[str, np.ndarray]):
        """打印积分结果摘要"""
        print("\n=== 积分计算结果摘要 ===")
        print(f"基组大小: {self.n_basis}x{self.n_basis}")
        print(f"ERI max multipole: {self.max_multipole if self.max_multipole>=0 else '完整展开'}")
        
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

def calculate_atomic_integrals(basis_set, max_multipole=-1) -> Dict[str, np.ndarray]:
    """
    计算原子的所有积分
    
    Args:
        basis_set: 原子轨道基组
        nuclear_charge: 核电荷数
        max_multipole: 最大多极展开项数, -1表示完整展开
        
    Returns:
        Dict: 所有积分矩阵
    """
    integral_calc = AtomicIntegrals(basis_set, max_multipole=max_multipole)
    integrals = integral_calc.compute_all_integrals()
    integral_calc.print_integral_summary(integrals)
    return integrals

