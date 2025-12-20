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

    def __init__(self, ks_results: KSResults, r_cuts: Dict[str, float] = None,spin='alpha',v_shift: float=0.0,auto_v_shift=True):
        """
        Args:
            ks_results: 全电子 LSDA 计算结果对象
            r_cuts: 截断半径字典, e.g. {'s': 1.2, 'p': 1.5}。
        """
        self.results = ks_results
        self.atom_element = ks_results.integral_calc.basis.element
        self.Z = ks_results.integral_calc.nuclear_charge
        self.r_grid = ks_results.integral_calc.r_grid
        self.spin=spin
        self.v_shift = v_shift
        # 确保 r_grid 不含 0 点以避免除零
        self.r_safe = np.where(self.r_grid < 1e-12, 1e-12, self.r_grid)
        
        # 假设是均匀网格，直接取前两点的差
        self.dr = self.r_grid[1] - self.r_grid[0]
        
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
        if self.auto_v_shift:
            if self.spin=='alpha':
                self.v_shift=self.v_ae_alpha[-1]
            else:
                self.v_shift=self.v_ae_beta[-1]

    def generate(self, local_channel: str = 'p') -> Pseudopotential:
        """生成赝势的主流程"""
        logging.info(f"开始为元素 {self.atom_element} 生成赝势...")
        # 根据请求的自旋选择对应的全电子势
        if self.spin == 'alpha':
            self.v_ae_total = self.v_ae_alpha.copy()
        else:
            self.v_ae_total = self.v_ae_beta.copy()
        self.v_ae_total -= self.v_shift
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
        self.pseudo_data = pseudo_data

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
        """
        修正版：正确处理简并轨道的占据数累加。
        逻辑：
        1. 遍历所有轨道，根据 (l, energy) 进行分组。
        2. 在组内累加占据数 (解决 0.33+0.33+0.33=1.0 的问题)。
        3. 对于每个 l，只保留能量最高的那个组（价态）。
        """
        res = self.results
        n_arr = res.integral_calc.orbital_n
        l_arr = res.integral_calc.orbital_l
        basis_funcs = res.integral_calc.radial_functions 
        
        # 1. 准备数据
        if self.spin == 'alpha':
            C = res.coefficients_alpha 
            occ = res.occ_alpha 
            # 注意：这里立刻应用 v_shift，后续所有逻辑都基于修正后的能量
            eps = res.orbital_energies_alpha - self.v_shift 
        else:
            C = res.coefficients_beta 
            occ = res.occ_beta 
            eps = res.orbital_energies_beta - self.v_shift

        # 2. 【核心步骤】分组与累加
        # 字典结构: groups[(l, energy_rounded)] = { ... state_data ... }
        groups = {}
        
        for i in range(res.integral_calc.n_basis):
            # 忽略完全空的轨道 (甚至忽略 < 0.01 的，因为我们要找主价态)
            # 但对于 0.33 这种占据，必须保留
            if occ[i] < 1e-3: continue 

            # 确定当前轨道的 l 和 n (基于最大贡献的基函数)
            weights = C[:, i]**2
            main_idx = np.argmax(weights)
            l_curr = l_arr[main_idx]
            n_curr = n_arr[main_idx]
            
            # 能量四舍五入，用于识别简并轨道
            # 例如 -0.033037 和 -0.033038 会被视为同一个 key
            e_curr = eps[i]
            e_key = round(e_curr, 4) 
            group_key = (l_curr, e_key)
            
            # 重构波函数 (只需要做一次，但为了简单先在这里做)
            # 实际上对于简并轨道，径向部分是一样的，存哪一个都行
            u_recon = np.zeros_like(self.r_grid)
            relevant_indices = [k for k in range(len(l_arr)) if l_arr[k] == l_curr]
            for k in relevant_indices:
                u_recon += C[k, i] * basis_funcs[k]
            norm = simpson(u_recon**2, self.r_grid)
            u_recon /= np.sqrt(norm)

            if group_key not in groups:
                # 新发现的能级组
                groups[group_key] = {
                    'n': n_curr,
                    'l': l_curr,
                    'energy': e_curr,       # 存精确能量
                    'occupation': occ[i],   # 初始化占据数
                    'radial_func': u_recon
                }
            else:
                # 已存在的能级组 -> 累加占据数！
                # 这就是解决问题的关键：0.33 -> 0.66 -> 0.99
                groups[group_key]['occupation'] += occ[i]
                
                # 可选：如果新轨道的能量略高（微小差异），更新为更精确的能量
                # 或者保持平均，这里为了简单保留第一个遇到的即可

        # 3. 【筛选步骤】挑选价态
        # 现在 groups 里可能有：
        # (l=0, e=-55): occ=2.0 (1s)
        # (l=0, e=-3.9): occ=2.0 (2s)
        # (l=0, e=-0.2): occ=2.0 (3s)
        # (l=1, e=-2.4): occ=6.0 (2p)
        # (l=1, e=-0.03): occ=1.0 (3p)  <-- 我们要这个！
        
        final_valence = {}
        
        for key, data in groups.items():
            l = data['l']
            e = data['energy']
            
            # 如果是同 l 的新能级，检查是否能量更高
            if l not in final_valence or e > final_valence[l]['energy']:
                final_valence[l] = data

        self.valence_config = final_valence
        
        # 4. 打印结果供检查
        log_msg = "提取到的价态(修正后): "
        for l, v in self.valence_config.items():
            log_msg += f"[L={l} n={v['n']} e={v['energy']:.4f} occ={v['occupation']:.2f}] "
        logging.info(log_msg)
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
        v_scr[idx_rc+1:] = self.v_ae_total[idx_rc+1:]
        
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