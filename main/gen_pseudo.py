from pathlib import Path
import matplotlib.pyplot as plt
output_dir=Path(__file__).parent.parent /"data/psd_results"
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(output_dir,exist_ok=True)
from core import *
all_results={}
#basis_set = get_aug_cc_pvtz_basis('Al',radius_cutoff=10.0,mesh_size=1000,grid_type='log')
basis_set = get_sto3g_basis('Al',radius_cutoff=8.0,grid_type='log',mesh_size=2000)
#basis_set = get_sto3g_basis('Al',radius_cutoff=8.0,grid_type='log',mesh_size=3000)
integral=AtomicIntegrals(basis_set,13,real_basis=True, eri_cache=True)
lsda_calc=AtomicLSDA(integral,damping_factor=0.7,max_iterations=200,strict_diagonalize=True)
#lsda_calc=AtomicHartreeFock(integral,damping_factor=0.7,max_iterations=200)
lsda_results=lsda_calc.run_scf()
# 1. 初始化生成器
# 为 Al 原子设置截断半径，3s 和 3p 轨道
r_cuts = {'s': 1.6, 'p': 1.6} 
generator = PseudopotentialGenerator(lsda_results, r_cuts=r_cuts,v_shift=0.0,auto_v_shift=False)
# 2. 生成赝势 (选择 'p' 通道作为局域势)
pseudo,basis_set_ps = generator.generate(local_channel="d")

# 3. 打印信息或保存
pseudo.print_info()
# 4. 可视化检查 (可选)
r = pseudo.r_grid
fig=plt.figure(figsize=(10, 6))
plt.plot(r, pseudo.v_local(r), label='V_local', linewidth=2)
for proj in pseudo.beta_projectors:
    l_char = ['s','p','d'][proj.l]
    # 注意 BetaProjector 存储的是 beta(r)
    # 物理量级可能很大，可以画来看看
    plt.plot(r, proj.radial_function(r), label=f'Projector {l_char}', linestyle='--')

plt.xlim(0, 8)
plt.ylim(-10, 5)
plt.xlabel('Radius (Bohr)')
plt.ylabel('Potential / Projector (au)')
plt.legend()
plt.title(f'{pseudo.element} Generated Pseudopotential')
plt.grid(True)
plt.savefig(output_dir / 'Al_pseudopotential.png', dpi=300)
plt.close()



int_psd=AtomicIntegrals(basis_set_ps,pseudo=pseudo,real_basis=True)
lsda_calc_psd=AtomicLSDA(int_psd,damping_factor=0.7,max_iterations=200)
lsda_results_psd=lsda_calc_psd.run_scf()

fig3s=plt.figure(figsize=(8,6))
xpsd,ypsd=get_radial_data(lsda_results_psd,"alpha",0)
x,y=get_radial_data(lsda_results,"alpha",5)
if y[-1]*ypsd[-1]<0:
    ypsd=-ypsd
plt.plot(x,y,label='All-electron 3s',linewidth=2)
plt.plot(xpsd,ypsd,label='Pseudopotential 3s',linewidth=2)
plt.xlabel('Radius (Bohr)')
plt.ylabel('Radial Wavefunction r*R(r)')
plt.title('Al 3s Orbital Comparison')
plt.legend()
plt.savefig(output_dir / 'Al_3s_orbital_comparison.png', dpi=300)
plt.close()

fig3p=plt.figure(figsize=(8,6))
xpsd,ypsd=get_radial_data(lsda_results_psd,"alpha",1)
x,y=get_radial_data(lsda_results,"alpha",6)
plt.plot(x,y,label='All-electron 3p',linewidth=2)
if ypsd[-1]*y[-1]<0:
    ypsd=-ypsd
plt.plot(xpsd,ypsd,label='Pseudopotential 3p',linewidth=2)
plt.legend()
plt.xlabel('Radius (Bohr)')
plt.ylabel('Radial Wavefunction r*R(r)')
plt.title('Al 3p Orbital Comparison')
plt.savefig(output_dir / 'Al_3p_orbital_comparison.png', dpi=300)
plt.close()
save_table( "Al_all_electron.txt",lsda_results, output_dir,10 )
save_table( "Al_pseudo.txt",lsda_results_psd, output_dir,5 )
