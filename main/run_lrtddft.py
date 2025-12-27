import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import *
basis = get_aug_cc_pvtz_basis('He',grid_type='log',radius_cutoff=30.0)
integral = AtomicIntegrals(basis,nuclear_charge=2, real_basis=True,eri_cache=True)
print(">>> 1. 静态计算 (He Atom, aug-cc-pvdz)")
# 静态 LSDA
lsda = AtomicLSDA(integral,strict_diagonalize=True,functional_type="lb94")
gs_results = lsda.run_scf()

lr_tddft = LinearResponseTDDFT(gs_results, lsda)
excitations = lr_tddft.solve(n_states=10)

print("\n>>> 3. 激发态结果")
print(f"{'State':<5} {'Energy(eV)':<12} {'Osc.Str':<10} {'Composition'}")
print("-" * 60)
for i, exc in enumerate(excitations):
    print(f"{i+1:<5} {exc.energy_ev:<12.4f} {exc.oscillator_strength:<10.4f} {', '.join(exc.contributions)}")
lr_tddft.plot_absorption_spectrum(
    filename="He_UV_Vis_Spectrum.png",
    title="He Absorption Spectrum",
    sigma=0.2,       # 展宽参数
    kind='gaussian',  # 使用高斯展宽
    start_ev=0.0,
    end_ev=55.0
)