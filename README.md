# 原子Hartree-Fock与LSDA计算程序

## 简介

本程序实现了自旋极化的原子Hartree-Fock (HF)和局域自旋密度近似(LSDA)计算。支持STO-3G基组和ABACUS格式的数值原子轨道基组+赝势。

## 主要功能

- ✅ 自旋极化Hartree-Fock (UHF)计算
- ✅ 自旋极化LSDA计算（SVWN5交换关联泛函）
- ✅ STO-3G基组支持
- ✅ 数值原子轨道基组+模守恒赝势支持
- ✅ 完整的双电子积分（多极展开）
- ✅ Gaunt系数计算
- ✅ 波函数可视化（径向波函数和二维密度图）

## 目录结构

```
DFTatom/
├── core/                          # 核心计算模块
│   ├── __init__.py                # 包初始化文件
│   ├── atomic_integrals.py        # 单/双电子积分计算（重叠、动能、库仑）
│   ├── basis_set.py               # 基组基类
│   ├── pseudo.py                  # 赝势基类（模守恒赝势接口）
│   ├── pseudo_gen.py              # 赝势生成工具
│   ├── sto3g.py                   # STO-3G基组
│   ├── HF.py                      # Hartree-Fock
│   ├── LSDA.py                    # LSDA
│   ├── lr_tddft.py                # 线性响应TDDFT
│   ├── rt_tddft.py                # 实时TDDFT
│   └── visualization.py           # 电子密度/轨道可视化
```

## 快速开始

### 1. 运行示例计算

```bash
# 氢原子和碳原子的HF计算（STO-3G）
python run_HF.py

# 氢原子和碳原子的LSDA计算（STO-3G）
python run_LSDA.py

# 使用NAO+赝势的计算
python run_HF_ps.py
python run_LSDA_ps.py
```

### 2. 生成报告所需的所有数据和图片

```bash
# 第一步：生成计算数据
python main/generate_all_results.py

# 第二步：生成波函数图片
python main/plot_wavefunctions.py

# 第三步：编译LaTeX文档
cd readme
pdflatex readme.tex
pdflatex readme.tex  # 运行两次确保交叉引用正确
```

## 使用示例

### STO-3G基组计算

```python
from core import *

# 氢原子HF计算
basis = get_sto3g_basis('H')
integral = AtomicIntegrals(basis, nuclear_charge=1, real_basis=True)
hf_calc = AtomicHartreeFock(integral)
hf_results = hf_calc.run_scf()
print_hf_results(hf_results)

# 碳原子LSDA计算
basis = get_sto3g_basis('C')
integral = AtomicIntegrals(basis, nuclear_charge=6, real_basis=True)
lsda_calc = AtomicLSDA(integral, multiplicity=3)
lsda_results = lsda_calc.run_scf()
print_ks_results(lsda_results)
```

### 数值原子轨道基+赝势计算

```python
from core import *

# 读取基组和赝势
basis = load_basis_set_from_file(
    "SG15-Version1p0__StandardOrbitals-Version2p0/C_gga_7au_100Ry_2s2p1d.orb"
)
pseudo = load_pseudopotential_from_upf(
    "SG15-Version1p0_Pseudopotential/SG15_ONCV_v1.0_upf/C_ONCV_PBE-1.0.upf"
)

# HF计算
integral = AtomicIntegrals(basis, pseudo=pseudo, real_basis=True)
hf_calc = AtomicHartreeFock(integral)
hf_results = hf_calc.run_scf()
```

### 波函数可视化

```python
from core.visualization import *

# 绘制径向波函数
plot_radial_wavefunctions(results, spin='alpha', max_r=4.0)

# 绘制二维等值线图
plot_orbital_contour(results, orbital_index=5, spin='alpha',
                     plane='xz', range_au=5.0)
```

## 依赖库

- Python 3.7+
- NumPy
- SciPy
- matplotlib
- sympy (用于Gaunt系数计算)

## 算法详情

详细的算法原理、公式推导和结果请参阅 `readme/readme.pdf`。

主要算法包括：

1. **Hartree-Fock方法**
   - 自旋极化Fock算符
   - 多极展开计算双电子积分
   - Gaunt系数计算（复球谐函数和实球谐函数）
   - 自洽场迭代

2. **LSDA方法**
   - Slater交换泛函
   - VWN5关联泛函
   - 解析导数计算交换关联势
   - Fermi-Dirac分数占据数

3. **赝势**
   - 局域赝势（local）
   - 非局域赝势（Kleinman-Bylander形式）

## 注意事项

1. NAO+赝势计算需要在当前目录下有对应的基组和赝势文件
2. LaTeX编译需要支持中文（使用ctex宏包）
3. 生成图片需要matplotlib支持中文字体
4. 完整的计算可能需要几分钟到十几分钟

## 作者

课程作业项目

## License

MIT License
