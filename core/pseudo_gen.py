import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.optimize import root
import logging
from typing import Dict, Tuple

# 导入现有模块的类和函数
from core.pseudo import Pseudopotential, BetaProjector
from core.basis_set import BasisSet, AtomicOrbital
from core.LSDA import KSResults, get_slater_vwn5

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PseudopotentialGenerator:
    """
    从全电子 LSDA 计算结果生成模守恒赝势 (Norm-Conserving Pseudopotential)
    完全适配非均匀网格（如对数网格）。
    """

    def __init__(self, ks_results: KSResults, r_cuts: Dict[str, float] = None, spin='alpha', v_shift: float = 0.0, auto_v_shift=True):
        """
        Args:
            ks_results: 全电子 LSDA 计算结果对象
            r_cuts: 截断半径字典, e.g. {'s': 1.2, 'p': 1.5}。
        """
        self.results = ks_results
        self.atom_element = ks_results.integral_calc.basis.element
        self.Z = ks_results.integral_calc.nuclear_charge
        self.r_grid = ks_results.integral_calc.r_grid
        self.spin = spin
        self.v_shift = v_shift
        
        # 确保 r_grid 不含 0 点以避免除零 (通常对数网格第一个点也不是0，但防万一)
        self.r_safe = np.where(self.r_grid < 1e-12, 1e-12, self.r_grid)
        
        # [修改 1] 移除了 self.dr。
        # 在非均匀网格中，dr 是随 r 变化的，不能使用标量。
        
        # 默认截断半径
        default_rc = 1.5
        self.r_cuts = {'s': default_rc, 'p': default_rc, 'd': default_rc, 'f': default_rc}
        if r_cuts:
            self.r_cuts.update(r_cuts)

        self.l_max = 0
        self.valence_config = {} 
        self.v_ae_alpha = ks_results.total_potential_alpha
        self.v_ae_beta = ks_results.total_potential_beta
        self.auto_v_shift = auto_v_shift
        
        # 自动确定真空能级对齐偏移量
        if self.auto_v_shift:
            if self.spin == 'alpha':
                self.v_shift = self.v_ae_alpha[-1]
            else:
                self.v_shift = self.v_ae_beta[-1]

    def generate(self, local_channel: str = 'p') -> Tuple[Pseudopotential, BasisSet]:
        """
        生成赝势的主流程，同时返回配套的最小数值基组 (Minimal Basis Set)。
        
        Returns:
            (pseudo, basis_set): 赝势对象和对应的基组对象
        """
        logging.info(f"开始为元素 {self.atom_element} 生成赝势...")
        
        # 1. 准备势函数
        if self.spin == 'alpha':
            self.v_ae_total = self.v_ae_alpha.copy()
        else:
            self.v_ae_total = self.v_ae_beta.copy()
            
        logging.info(f"应用真空能级偏移 v_shift: {self.v_shift:.6f} Ha")
        self.v_ae_total -= self.v_shift
        
        # 2. 提取全电子价态波函数 (现在包含了正确的总占据数)
        self._extract_ae_valence_states()
        
        # 3. 为每个通道生成屏蔽赝势 V_scr
        pseudo_data = {} 
        rho_val_pseudo_radial = np.zeros_like(self.r_grid)

        if not self.valence_config:
            raise ValueError("未找到价电子态，无法生成赝势。")
        self.l_max = max(self.valence_config.keys())
        
        # --- 准备基组对象 ---
        basis_set = BasisSet(element=self.atom_element)
        basis_set.set_basis_parameters(
            radius_cutoff=self.r_grid[-1]
        )
        
        for l, state in self.valence_config.items():
            l_str = self._l_to_str(l)
            rc = self.r_cuts.get(l_str, 1.5)
            logging.info(f"处理 L={l} ({l_str}) 通道 (rc={rc:.2f} au, ε={state['energy']:.4f} Ha)...")
            
            # 生成赝波函数 phi_pp 和 屏蔽势 V_scr
            try:
                phi_pp, v_scr = self._pseudize_channel(l, state['radial_func'], state['energy'], rc)
            except Exception as e:
                logging.error(f"L={l} 通道生成失败: {e}")
                raise e
            
            pseudo_data[l] = {
                'phi_pp': phi_pp, 
                'v_scr': v_scr,
                'energy': state['energy']
            }
            
            # --- 累加赝价电子密度 ---
            # 使用提取到的总占据数 (Alpha+Beta)
            occ_total = state['occupation']
            rho_val_pseudo_radial += occ_total * (phi_pp ** 2)
            
            # --- 构建 AtomicOrbital 并添加到基组 ---
            n_principal = state['n'] 
            
            # 为该 l 下的所有 m 分量创建轨道 (-l ... +l)
            # 这样对于 p 轨道会创建 px, py, pz 三个对象
            for m in range(-l, l + 1):
                orb = AtomicOrbital(
                    n=n_principal,
                    l=l,
                    m=m,
                    orbital_type=l, 
                    n_index=n_principal,
                    r_grid=self.r_grid.copy(),
                    radial_function=phi_pp.copy() 
                )
                basis_set.add_orbital(orb)

        # 4. 去屏蔽 (使用总密度计算 Hartree)
        logging.info("计算赝价电荷密度的 Hartree 和 XC 势...")
        rho_val_pseudo_3d = rho_val_pseudo_radial / (4.0 * np.pi * self.r_safe**2)
        
        v_hartree = self._solve_poisson(rho_val_pseudo_radial)
        
        rho_half = rho_val_pseudo_3d / 2.0
        v_xc_alpha, _, _ = get_slater_vwn5(rho_half, rho_half) 
        v_xc = v_xc_alpha

        # 5. 构建离子赝势 V_ion
        v_ion_channels = {}
        for l, data in pseudo_data.items():
            v_ion = data['v_scr'] - v_hartree - v_xc
            v_ion_channels[l] = v_ion

        # 6. 分离局域和非局域部分 (Kleinman-Bylander)
        l_loc_idx = self._str_to_l(local_channel)
        if l_loc_idx not in v_ion_channels:
            l_loc_idx = max(v_ion_channels.keys())
            logging.warning(f"指定的局域通道 {local_channel} 无数据，自动切换为 L={l_loc_idx}")

        v_local_grid = v_ion_channels[l_loc_idx]
        
        beta_projectors = []
        d_matrix_blocks = [] 
        
        z_valence = np.sum([s['occupation'] for s in self.valence_config.values()])
        
        logging.info("构建 Kleinman-Bylander 投影算符 (强制归一化 Beta)...")
        
        proj_count = 0
        for l, v_ion in v_ion_channels.items():
            if l == l_loc_idx:
                continue
            
            delta_v = v_ion - v_local_grid
            chi_pp = pseudo_data[l]['phi_pp'] 
            raw_beta = delta_v * chi_pp 
            
            integrand_ekb = chi_pp * raw_beta
            e_kb = simpson(integrand_ekb, x=self.r_grid)
            
            if abs(e_kb) < 1e-10:
                logging.warning(f"L={l} 通道的 KB 系数过小 ({e_kb})，跳过投影。")
                continue
            
            beta_norm_sq = simpson(raw_beta**2, x=self.r_grid)
            
            if beta_norm_sq < 1e-20:
                continue

            norm_factor = 1.0 / np.sqrt(beta_norm_sq)
            beta_normalized = raw_beta * norm_factor
            
            d_val = beta_norm_sq / e_kb
            
            beta_interp = interp1d(self.r_grid, beta_normalized, kind='cubic', bounds_error=False, fill_value=0.0)
            
            beta_proj = BetaProjector(l=l, index=proj_count, radial_function=beta_interp)
            beta_projectors.append(beta_proj)
            d_matrix_blocks.append(d_val)
            proj_count += 1

        n_proj = len(beta_projectors)
        d_matrix = np.diag(d_matrix_blocks) if n_proj > 0 else np.zeros((0,0))

        v_local_interp = interp1d(self.r_grid, v_local_grid, kind='cubic', bounds_error=False, fill_value=(v_local_grid[0], 0.0))
        self.pseudo_data = pseudo_data

        # 7. 封装赝势
        pseudo = Pseudopotential(
            element=self.atom_element,
            z_valence=float(z_valence),
            l_max=self.l_max,
            mesh_size=len(self.r_grid),
            functional="LDA-PZ/VWN",
            is_norm_conserving=True,
            r_grid=self.r_grid,
            v_local=v_local_interp,
            d_matrix=d_matrix,
            beta_projectors=beta_projectors,
            rho_atomic=rho_val_pseudo_radial
        )
        
        logging.info("赝势及配套基组生成完成！")
        return pseudo, basis_set
    def _extract_ae_valence_states(self):
        """
        处理简并轨道的占据数累加，并统计所有自旋通道的总电荷。
        """
        res = self.results
        n_arr = res.integral_calc.orbital_n
        l_arr = res.integral_calc.orbital_l
        basis_funcs = res.integral_calc.radial_functions 
        
        # 1. 准备数据
        # 无论 spin 是什么，我们都同时获取 alpha 和 beta 的占据数
        occ_alpha = res.occ_alpha
        occ_beta = res.occ_beta
        
        # 确定用于提取波函数形状的“主通道”
        if self.spin == 'alpha':
            C_main = res.coefficients_alpha 
            occ_main = occ_alpha
            eps_main = res.orbital_energies_alpha - self.v_shift 
        else:
            C_main = res.coefficients_beta 
            occ_main = occ_beta
            eps_main = res.orbital_energies_beta - self.v_shift

        # 2. 分组与累加
        groups = {}
        
        for i in range(res.integral_calc.n_basis):
            # 只要任意一个自旋通道有占据，就应该考虑
            # 或者是主通道有显著占据 (有时 beta 可能为空但我们需要它的形状，这种情况较少见)
            total_occ_i = occ_alpha[i] + occ_beta[i]
            
            # 过滤完全空的轨道 (阈值设低一点以防微量占据)
            if total_occ_i < 1e-4 and occ_main[i] < 1e-4: continue 

            # 确定当前轨道的 l 和 n
            weights = C_main[:, i]**2
            main_idx = np.argmax(weights)
            l_curr = l_arr[main_idx]
            n_curr = n_arr[main_idx]
            
            # 能量 Key
            e_curr = eps_main[i]
            e_key = round(e_curr, 4) 
            group_key = (l_curr, e_key)
            
            # 重构波函数 (使用主通道系数)
            u_recon = np.zeros_like(self.r_grid)
            relevant_indices = [k for k in range(len(l_arr)) if l_arr[k] == l_curr]
            for k in relevant_indices:
                u_recon += C_main[k, i] * basis_funcs[k]
            
            # 积分归一化
            norm = simpson(u_recon**2, x=self.r_grid)
            if norm < 1e-10: continue
            u_recon /= np.sqrt(norm)

            if group_key not in groups:
                groups[group_key] = {
                    'n': n_curr,
                    'l': l_curr,
                    'energy': e_curr,
                    'occupation': 0.0, # 初始化总占据数
                    'radial_func': u_recon
                }
            
            # 累加总占据数 (Alpha + Beta)
            groups[group_key]['occupation'] += total_occ_i

        # 3. 筛选价态
        final_valence = {}
        for key, data in groups.items():
            l = data['l']
            e = data['energy']
            if l not in final_valence or e > final_valence[l]['energy']:
                final_valence[l] = data

        self.valence_config = final_valence
        
        log_msg = "提取到的价态(修正后): "
        for l, v in self.valence_config.items():
            log_msg += f"[L={l} n={v['n']} e={v['energy']:.4f} Z_val={v['occupation']:.2f}] "
        logging.info(log_msg)
    def _pseudize_channel(self, l: int, u_ae_in: np.ndarray, energy: float, rc: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extended Troullier-Martins 赝化流程 (包含 r^8 项以匹配三阶导数)
        """
        u_ae = u_ae_in.copy()
        idx_rc = np.searchsorted(self.r_grid, rc)
        r_vals = self.r_grid[:idx_rc+1]
        rc_safe = r_vals[-1]
        
        val_c = u_ae[idx_rc]
        # 符号修正
        if val_c < 0:
            u_ae *= -1.0
            val_c = u_ae[idx_rc]
        
        if abs(val_c) < 1e-12:
            raise ValueError(f"L={l}: 波函数在 rc 处过小，请更改截断半径。")

        # =======================================================
        # 1. 计算全电子波函数在 rc 处的高阶导数 (使用对数网格)
        # =======================================================
        d1_array = np.gradient(u_ae, self.r_grid, edge_order=2)
        d2_array = np.gradient(d1_array, self.r_grid, edge_order=2)
        # 【新增】计算三阶导数
        d3_array = np.gradient(d2_array, self.r_grid, edge_order=2)
        
        der1_c = d1_array[idx_rc]
        der2_c = d2_array[idx_rc]
        der3_c = d3_array[idx_rc]
        
        # 模守恒目标值
        norm_ae = simpson(u_ae[:idx_rc+1]**2, x=r_vals)
        
        # =======================================================
        # 2. 计算目标 P(r) 函数及其导数
        # =======================================================
        # 比值定义
        d_ratio1 = der1_c / val_c
        d_ratio2 = der2_c / val_c
        d_ratio3 = der3_c / val_c
        
        # P(r) = ln(u / r^(l+1))
        ln_part = np.log(abs(val_c) / (rc_safe**(l+1))) 
        
        # P'(r)
        p_der1_target = d_ratio1 - (l+1)/rc_safe
        
        # P''(r) = (u''/u) - (u'/u)^2 + (l+1)/r^2
        p_der2_target = d_ratio2 - d_ratio1**2 + (l+1)/(rc_safe**2)
        
        # 【新增】P'''(r) 目标值
        # P''' = (u'''/u) - 3(u''/u)(u'/u) + 2(u'/u)^3 - 2(l+1)/r^3
        p_der3_target = (d_ratio3 - 3 * d_ratio2 * d_ratio1 + 
                         2 * d_ratio1**3 - 2 * (l+1) / (rc_safe**3))

        # =======================================================
        # 3. 定义拟合多项式 (扩展到 c4 * r^8)
        # c 向量现在有 5 个元素: [c0, c1, c2, c3, c4]
        # p(r) = c0 + c1*r^2 + c2*r^4 + c3*r^6 + c4*r^8
        # =======================================================
        def poly_p(r, c):
            r2 = r**2
            return c[0] + c[1]*r2 + c[2]*r2**2 + c[3]*r2**3 + c[4]*r2**4
            
        def get_u_trial(c):
            p = poly_p(r_vals, c)
            p = np.clip(p, -200, 200) # 数值稳定性
            return (r_vals**(l+1)) * np.exp(p)

        # =======================================================
        # 4. 定义优化目标函数 (5个方程)
        # =======================================================
        def objective(c):
            r = rc_safe
            r2 = r*r; r3 = r2*r; r4 = r2*r2; r5 = r4*r; r6 = r4*r2; r7 = r6*r; r8 = r4*r4
            
            # Eq1: 匹配 P(rc)
            p_val_trial = c[0] + c[1]*r2 + c[2]*r4 + c[3]*r6 + c[4]*r8
            eq1 = p_val_trial - ln_part
            
            # Eq2: 匹配 P'(rc)
            p_der1_trial = 2*c[1]*r + 4*c[2]*r3 + 6*c[3]*r5 + 8*c[4]*r7
            eq2 = p_der1_trial - p_der1_target
            
            # Eq3: 匹配 P''(rc)
            p_der2_trial = 2*c[1] + 12*c[2]*r2 + 30*c[3]*r4 + 56*c[4]*r6
            eq3 = p_der2_trial - p_der2_target
            
            # 【新增】Eq4: 匹配 P'''(rc)
            p_der3_trial = 24*c[2]*r + 120*c[3]*r3 + 336*c[4]*r5
            eq4 = p_der3_trial - p_der3_target
            
            # Eq5: 模守恒
            u_trial = get_u_trial(c)
            norm_trial = simpson(u_trial**2, x=r_vals)
            eq5 = norm_trial - norm_ae 
            
            return [eq1, eq2, eq3, eq4, eq5]
        
        # 初始猜测 (增加 c4=0.0)
        c_guess = [ln_part, -0.5, 0.0, 0.0, 0.0] 
        # 使用 'lm' 方法求解非线性方程组
        sol = root(objective, c_guess, method='lm', options={'maxiter': 10000, 'ftol': 1e-10, 'xtol': 1e-10})
        
        if not sol.success:
            logging.warning(f"L={l} 扩展TM拟合未完全收敛: {sol.message}, 残差: {np.linalg.norm(sol.fun):.4e}")
            # 可以尝试其他初始猜测或算法，这里暂时继续

        c_opt = sol.x
        u_pp = u_ae.copy()
        u_pp[:idx_rc+1] = get_u_trial(c_opt)
        
        # =======================================================
        # 5. 反推屏蔽势 V_scr
        # V_scr = E - l(l+1)/2r^2 + (1/2) * [P'' + (P')^2 + 2(l+1)P'/r]
        # =======================================================
        v_scr = np.zeros_like(self.r_grid)
        
        c = c_opt
        r_in = self.r_safe[:idx_rc+1]
        r2 = r_in**2; r3 = r2*r_in; r4 = r2**2; r5 = r4*r_in; r6 = r4*r2; r7 = r6*r_in
        
        # 计算拟合多项式的导数 P' 和 P'' 在整个内部区域的值
        Pp = 2*c[1]*r_in + 4*c[2]*r3 + 6*c[3]*r5 + 8*c[4]*r7
        Ppp = 2*c[1] + 12*c[2]*r2 + 30*c[3]*r4 + 56*c[4]*r6
        
        # TM 势函数公式
        term = Ppp + Pp**2 + 2*(l+1)*Pp/r_in
        v_scr[:idx_rc+1] = energy + 0.5 * term
        
        # 外部区域直接使用全电子势
        v_scr[idx_rc+1:] = self.v_ae_total[idx_rc+1:]
        
        return u_pp, v_scr

    def _solve_poisson(self, rho_radial):
        """
        求解泊松方程 (适配非均匀网格)
        rho_radial: 4*pi*r^2 * n(r)
        """
        r = self.r_safe
        # [修改 4] 确保积分函数使用 r 坐标数组
        Q_r = cumulative_trapezoid(rho_radial, r, initial=0)
        integrand2 = rho_radial / r
        
        total_int = simpson(integrand2, x=r)
        cum_int = cumulative_trapezoid(integrand2, r, initial=0)
        integral2 = total_int - cum_int
        
        v_h = Q_r / r + integral2
        return v_h

    def _l_to_str(self, l):
        return {0:'s', 1:'p', 2:'d', 3:'f'}.get(l, 'x')
    
    def _str_to_l(self, s):
        return {'s':0, 'p':1, 'd':2, 'f':3}.get(s, 0)