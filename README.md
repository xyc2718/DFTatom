# 原子Hartree-Fock与LSDA计算程序

## 简介

本程序为复旦大学电子结构计算课程作业，实现了自旋极化的原子Hartree-Fock (HF)和局域自旋密度近似(LSDA)计算。支持STO-3G/6G,aug-cc-pvtz/pvdz等高斯基组和ABACUS格式的数值原子轨道基组+赝势。支持TM方法的赝势生成，RT-TDDFT/LR-TDDFT的计算

## 主要功能

- 自旋极化Hartree-Fock (UHF)计算
- 自旋极化LSDA计算（SVWN5交换关联泛函）
- 高斯基组/数值原子轨道基组+模守恒赝势支持
- 波函数可视化
- 对数网格TM方法生成赝势
- RT-TDDFT计算原子光谱和拉比振荡
- LR-TDDFT计算原子光谱

## 目录结构

```
DFTatom/
├── core/                         # 核心计算模块
│   ├── __init__.py               # 包初始化文件
│   ├── atomic_integrals.py       # 单/双电子积分计算
│   ├── basis_set.py              # 基组基类
│   ├── pseudo.py                 # 赝势基类
│   ├── pseudo_gen.py             # 赝势生成逻辑核心
│   ├── sto3g.py                  # 高斯基组
│   ├── HF.py                     # Hartree-Fock 求解器
│   ├── LSDA.py                   # LSDA 交换相关泛函实现
│   ├── lr_tddft.py               # 线性响应 TDDFT 逻辑
│   ├── rt_tddft.py               # 实时 TDDFT 逻辑
│   └── visualization.py          # 电子密度/轨道可视化
│
├── main/                         # 执行脚本目录
│   ├── run_dft.py                # 主程序：执行常规 DFT 计算
│   ├── gen_pseudo.py             # 脚本：调用核心模块生成赝势文件
│   ├── run_rttddft.py            # 脚本：执行实时时间相关 DFT 模拟
│   └── run_lrtddft.py            # 脚本：执行线性响应激发的计算
│
└── README.md                     # 项目说明文档
```

## 快速开始

### 1. 运行示例计算

```bash
# 氢原子和碳原子的HF/LSDA计算(NAO/STO-3G)
python main/run_dft.py

#unc-STO-6G生成Al原子赝势
python main/gen_pseudo.py

#He原子的RT-TDDFT计算光谱
python main/run_rttddft.py

#He原子拉比振荡
python main/run_Rabi.py

#He原子LR-TDDFT计算光谱
python main/run_lrtddft.py
```

## 使用示例

### HF/DFT计算
```python
from core import *
# === 1. 全电子计算 (STO-3G) ===

# 示例 A: 氢原子 Hartree-Fock
calc_hf = AtomicHartreeFock(AtomicIntegrals(get_sto3g_basis('H'), nuclear_charge=1))
res_hf = calc_hf.run_scf()

# 示例 B: 碳原子 LSDA (自旋多重度=3)
# uncontracted=False 表示使用收缩基组
calc_lsda = AtomicLSDA(
    AtomicIntegrals(get_sto3g_basis('C',uncontracted=False), nuclear_charge=6),
    multiplicity=3
)
res_lsda = calc_lsda.run_scf()
# === 2. 赝势计算 (数值原子轨道) ===
basis = load_basis_set_from_file("C_gga_7au_100Ry_2s2p1d.orb")
pseudo = load_pseudopotential_from_upf("C_ONCV_PBE-1.0.upf")

# 传入 pseudo 参数即开启赝势模式
calc_psp = AtomicHartreeFock(AtomicIntegrals(basis, pseudo=pseudo))
res_psp = calc_psp.run_scf()
```

### 结果分析与可视化
```python
print_ks_results(res_lsda) #打印LDA计算基本信息
analyze_orbital_compositions(res_lsda) #分析轨道成分
plot_radial_wavefunctions(results, spin='alpha', max_r=4.0)
# 绘制二维等值线图
plot_orbital_contour(results, orbital_index=5, spin='alpha', 
                     plane='xz', range_au=5.0)
```

### 赝势生成

```python
# === 1. 全电子计算 ===
#使用精度更高的unc-sto6g基组,务必使用对数网格
basis_set = get_sto6g_basis('Al',radius_cutoff=8.0,grid_type='log',mesh_size=2000,uncontracted)
integral=AtomicIntegrals(basis_set,13,real_basis=True)
lsda_calc=AtomicLSDA(integral)

# === 2.生成赝势===
r_cuts = {'s': 1.6, 'p': 1.6} 
generator = PseudopotentialGenerator(lsda_results, r_cuts=r_cuts)
# 选择 'p' 通道作为局域势
pseudo,basis_set_ps = generator.generate(local_channel="p")
```
### TDDFT
```python
basis = get_aug_cc_pvtz_basis('Li',grid_type='log',radius_cutoff=30.0,mesh_size=2000) #使用扩散基组
integral = AtomicIntegrals(basis,nuclear_charge=4, real_basis=True,eri_cache=True)
# 静态 LSDA
lsda = AtomicLSDA(integral,strict_diagonalize=True) #大型基组推荐开启strict_diagonalize
gs_results = lsda.run_scf()
# RT-TDDFT计算
tddft = RealTimeTDDFT(gs_results,lsda,dt=0.05, strict_diagonalize=False,threshold=1e-12)
tddft.propagate(total_time=500.0, print_interval=100,kick_params={'strength': 0.001, 'direction': z},field_func=lambda t : lambda x: 0.1* np.sin(0.1* t)}
spectrum = tddft.calculate_spectrum(kick_strength=0.001, damping=0.003)
plt.plot(spectrum["energy_ev"],spectrum["intensity"])
plt.plot(np.array(tddft.time_history), np.array(tddft.dipole_history)[:,2])
```

### LR-TDDFT计算
```python
basis = get_aug_cc_pvtz_basis('Li',grid_type='log',radius_cutoff=30.0,mesh_size=2000) #使用扩散基组
integral = AtomicIntegrals(basis,nuclear_charge=4, real_basis=True,eri_cache=True)
# 静态 LSDA
lsda = AtomicLSDA(integral,strict_diagonalize=True,functional_type="lb94") LR-TDDFT推荐使用LB94泛函
gs_results = lsda.run_scf()
lr_tddft = LinearResponseTDDFT(gs_results, lsda)
excitations = lr_tddft.solve(n_states=10)
spectrum =lr_tddft.get_absorption_spectrum(0,20.0)
```
## License

GPL3.0 License
