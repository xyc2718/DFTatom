from core import *

# 假设 lsda_results 是您之前运行 STO-3G 全电子计算的结果
#basis_set = get_aug_cc_pvtz_basis('C',radius_cutoff=10.0,mesh_size=1000)
#basis_set = get_ano_rcc_vqzp_basis('Al',radius_cutoff=20.0,mesh_size=1000)
basis_set = get_sto3g_basis('Al',radius_cutoff=6.0,mesh_size=601)
integral=AtomicIntegrals(basis_set,13,real_basis=False)
lsda_calc=AtomicLSDA(integral,damping_factor=0.7,max_iterations=200,strict_diagonalize=True)
lsda_results=lsda_calc.run_scf()
# 1. 初始化生成器
# 为 Al 原子设置截断半径，3s 和 3p 轨道
r_cuts = {'s': 1.6, 'p': 1.6} 
generator = PseudopotentialGenerator(lsda_results, r_cuts=r_cuts)

# 2. 生成赝势 (选择 'p' 通道作为局域势)
pseudo = generator.generate(local_channel='p')

# 3. 打印信息或保存
pseudo.print_info()

# 4. 可视化检查 (可选)
r = pseudo.r_grid
plt.figure(figsize=(10, 6))
plt.plot(r, pseudo.v_local(r), label='V_local (p)', linewidth=2)
for proj in pseudo.beta_projectors:
    l_char = ['s','p','d'][proj.l]
    # 注意 BetaProjector 存储的是 beta(r)
    # 物理量级可能很大，可以画来看看
    plt.plot(r, proj.radial_function(r), label=f'Projector {l_char}', linestyle='--')

plt.xlim(0, 25)
plt.ylim(-10, 5)
plt.xlabel('Radius (Bohr)')
plt.ylabel('Potential / Projector (au)')
plt.legend()
plt.title(f'{pseudo.element} Generated Pseudopotential')
plt.grid(True)
plt.show()