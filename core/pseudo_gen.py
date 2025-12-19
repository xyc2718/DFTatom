import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.optimize import root
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# 导入现有模块的类和函数
from core.pseudo import Pseudopotential, BetaProjector
from core.LSDA import KSResults, get_slater_vwn5

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PseudopotentialGenerator:
    """
    从全电子 LSDA 计算结果生成模守恒赝势 (Norm-Conserving Pseudopotential)
    """

    def __init__(self, ks_results: KSResults, r_cuts: Dict[str, float] = None):
        """
        Args:
            ks_results: 全电子 LSDA 计算结果对象
            r_cuts: 截断半径字典, e.g. {'s': 1.2, 'p': 1.5}。
        """
        self.results = ks_results
        self.atom_element = ks_results.integral_calc.basis.element
        self.Z = ks_results.integral_calc.nuclear_charge
        self.r_grid = ks_results.integral_calc.r_grid
        
        # 确保 r_grid 不含 0 点以避免除零
        self.r_safe = np.where(self.r_grid < 1e-12, 1e-12, self.r_grid)
        
        # --- 修复点 1: 强制 self.dr 为标量 ---
        # 假设是均匀网格，直接取前两点的差
        self.dr = self.r_grid[1] - self.r_grid[0]
        # 如果 integral_calc 中存了 dr，也可以直接用
        if hasattr(ks_results.integral_calc, 'dr'):
            self.dr = ks_results.integral_calc.dr
        
        # 默认截断半径
        default_rc = 1.5
        self.r_cuts = {'s': default_rc, 'p': default_rc, 'd': default_rc, 'f': default_rc}
        if r_cuts:
            self.r_cuts.update(r_cuts)

        self.l_max = 0
        self.valence_config = {} 

    def generate(self, local_channel: str = 'p') -> Pseudopotential:
        """生成赝势的主流程"""
        logging.info(f"开始为元素 {self.atom_element} 生成赝势...")
        
        # 1. 提取全电子价态波函数
        self._extract_ae_valence_states()
        
        # 2. 为每个通道生成屏蔽赝势 V_scr
        pseudo_data = {} 
        rho_val_pseudo_radial = np.zeros_like(self.r_grid)

        if not self.valence_config:
            raise ValueError("未找到价电子态，无法生成赝势。")
        self.l_max = max(self.valence_config.keys())
        
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
            
            # 累加赝价电子密度
            occ_total = state['occupation']
            rho_val_pseudo_radial += occ_total * (phi_pp ** 2)

        # 3. 去屏蔽 (Unscreening)
        logging.info("计算赝价电荷密度的 Hartree 和 XC 势...")
        rho_val_pseudo_3d = rho_val_pseudo_radial / (4.0 * np.pi * self.r_safe**2)
        
        v_hartree = self._solve_poisson(rho_val_pseudo_radial)
        
        rho_half = rho_val_pseudo_3d / 2.0
        v_xc_alpha, _, _ = get_slater_vwn5(rho_half, rho_half) 
        v_xc = v_xc_alpha 

        # 4. 构建离子赝势 V_ion
        v_ion_channels = {}
        for l, data in pseudo_data.items():
            v_ion = data['v_scr'] - v_hartree - v_xc
            v_ion_channels[l] = v_ion

        # 5. 分离局域和非局域部分 (Kleinman-Bylander)
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
            
            # delta V = V_l - V_loc
            delta_v = v_ion - v_local_grid
            
            # beta = delta_v * u_pseudo
            chi_pp = pseudo_data[l]['phi_pp'] 
            raw_beta = delta_v * chi_pp 
            
            # E_KB = < chi | beta_raw >
            integrand_ekb = chi_pp * raw_beta
            e_kb = simpson(integrand_ekb, x=self.r_grid)
            
            if abs(e_kb) < 1e-10:
                logging.warning(f"L={l} 通道的 KB 系数过小 ({e_kb})，跳过投影。")
                continue
            
            # 归一化处理
            beta_norm_sq = simpson(raw_beta**2, x=self.r_grid)
            
            if beta_norm_sq < 1e-20:
                continue

            norm_factor = 1.0 / np.sqrt(beta_norm_sq)
            beta_normalized = raw_beta * norm_factor
            
            # 修正 D 矩阵: D = N / E_KB
            d_val = beta_norm_sq / e_kb
            
            beta_interp = interp1d(self.r_grid, beta_normalized, kind='cubic', bounds_error=False, fill_value=0.0)
            
            beta_proj = BetaProjector(l=l, index=proj_count, radial_function=beta_interp)
            beta_projectors.append(beta_proj)
            d_matrix_blocks.append(d_val)
            proj_count += 1

        n_proj = len(beta_projectors)
        d_matrix = np.diag(d_matrix_blocks) if n_proj > 0 else np.zeros((0,0))

        v_local_interp = interp1d(self.r_grid, v_local_grid, kind='cubic', bounds_error=False, fill_value=(v_local_grid[0], 0.0))

        # 6. 封装
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
        
        logging.info("赝势生成完成！")
        return pseudo

    def _extract_ae_valence_states(self):
        res = self.results
        n_arr = res.integral_calc.orbital_n
        l_arr = res.integral_calc.orbital_l
        basis_funcs = res.integral_calc.radial_functions 
        
        C = res.coefficients_alpha 
        occ = res.occ_alpha + res.occ_beta 
        eps = res.orbital_energies_alpha

        found_states = {} 
        
        for i in range(res.integral_calc.n_basis):
            if occ[i] < 1e-3: continue 
            
            weights = C[:, i]**2
            main_idx = np.argmax(weights)
            l_curr = l_arr[main_idx]
            n_curr = n_arr[main_idx] 
            
            if l_curr not in found_states or eps[i] > found_states[l_curr]['energy']:
                u_recon = np.zeros_like(self.r_grid)
                relevant_indices = [k for k in range(len(l_arr)) if l_arr[k] == l_curr]
                
                for k in relevant_indices:
                    u_recon += C[k, i] * basis_funcs[k]
                
                norm = simpson(u_recon**2, self.r_grid)
                u_recon /= np.sqrt(norm)
                
                found_states[l_curr] = {
                    'n': n_curr,
                    'energy': eps[i],
                    'occupation': occ[i],
                    'radial_func': u_recon
                }
        
        self.valence_config = found_states
        logging.info(f"提取到的价态: { {k: f'n={v['n']}, e={v['energy']:.3f}' for k,v in found_states.items()} }")

    def _pseudize_channel(self, l: int, u_ae_in: np.ndarray, energy: float, rc: float) -> Tuple[np.ndarray, np.ndarray]:
        u_ae = u_ae_in.copy()
        idx_rc = np.searchsorted(self.r_grid, rc)
        r_vals = self.r_grid[:idx_rc+1]
        
        val_c = u_ae[idx_rc]
        
        # 符号修正
        if val_c < 0:
            u_ae *= -1.0
            val_c = u_ae[idx_rc]
        
        if val_c < 1e-12:
            raise ValueError(f"L={l}: 波函数在 rc 处过小，请更改截断半径。")

        # --- 修复点 2: 使用标量 self.dr 计算导数 ---
        der1_c = (u_ae[idx_rc+1] - u_ae[idx_rc-1]) / (2*self.dr) 
        der2_c = (u_ae[idx_rc+1] - 2*u_ae[idx_rc] + u_ae[idx_rc-1]) / (self.dr**2)
        
        norm_ae = simpson(u_ae[:idx_rc+1]**2, x=r_vals)
        
        rc_safe = r_vals[-1]
        ln_part = np.log(val_c / (rc_safe**(l+1))) 
        
        d1 = der1_c / val_c
        d2 = der2_c / val_c
        
        p_val = ln_part
        p_der1 = d1 - (l+1)/rc_safe
        p_der2 = d2 - d1**2 + (l+1)/(rc_safe**2)
        
        def poly_p(r, c):
            return c[0] + c[1]*r**2 + c[2]*r**4 + c[3]*r**6
            
        def get_u_trial(c):
            p = poly_p(r_vals, c)
            p = np.clip(p, -100, 100) 
            return (r_vals**(l+1)) * np.exp(p)

        def objective(c):
            r = rc_safe
            r2 = r*r; r4 = r2*r2; r6 = r4*r2
            
            eq1 = (c[0] + c[1]*r2 + c[2]*r4 + c[3]*r6) - p_val
            eq2 = (2*c[1]*r + 4*c[2]*r*r2 + 6*c[3]*r*r4) - p_der1
            eq3 = (2*c[1] + 12*c[2]*r2 + 30*c[3]*r4) - p_der2
            
            u_trial = get_u_trial(c)
            norm_trial = simpson(u_trial**2, x=r_vals)
            eq4 = norm_trial - norm_ae 
            
            return [float(eq1), float(eq2), float(eq3), float(eq4)]
        
        c_guess = [p_val, -0.5, 0.0, 0.0] 
        
        sol = root(objective, c_guess, method='lm', options={'maxiter': 2000})
        
        if not sol.success:
            logging.warning(f"L={l} 拟合未完全收敛: {sol.message}")
            
        c_opt = sol.x
        u_pp = u_ae.copy()
        u_pp[:idx_rc+1] = get_u_trial(c_opt)
        
        v_scr = np.zeros_like(self.r_grid)
        
        # A. 内部区域
        c = c_opt
        r_in = self.r_safe[:idx_rc+1]
        r2 = r_in**2; r4 = r2**2
        
        Pp = 2*c[1]*r_in + 4*c[2]*r_in*r2 + 6*c[3]*r_in*r4
        Ppp = 2*c[1] + 12*c[2]*r2 + 30*c[3]*r4
        
        term = Ppp + Pp**2 + 2*(l+1)*Pp/r_in
        v_scr[:idx_rc+1] = energy + 0.5 * term
        
        # B. 外部区域
        u_out = u_pp
        d2u_out = np.gradient(np.gradient(u_out, self.dr), self.dr)
        denom = np.where(abs(u_out) > 1e-12, u_out, 1e-12)
        v_scr_out = energy + 0.5 * d2u_out / denom - 0.5 * l*(l+1) / (self.r_safe**2)
        
        v_scr[idx_rc+1:] = v_scr_out[idx_rc+1:]
        
        return u_pp, v_scr

    def _solve_poisson(self, rho_radial):
        r = self.r_safe
        Q_r = cumulative_trapezoid(rho_radial, r, initial=0)
        integrand2 = rho_radial / r
        
        total_int = simpson(integrand2, r)
        cum_int = cumulative_trapezoid(integrand2, r, initial=0)
        integral2 = total_int - cum_int
        
        v_h = Q_r / r + integral2
        return v_h

    def _l_to_str(self, l):
        return {0:'s', 1:'p', 2:'d', 3:'f'}.get(l, 'x')
    
    def _str_to_l(self, s):
        return {'s':0, 'p':1, 'd':2, 'f':3}.get(s, 0)