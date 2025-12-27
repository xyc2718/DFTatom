import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import *
import numpy as np
import matplotlib.pyplot as plt
output_path="Rabi_He"
from pathlib import Path
output_path=Path(output_path)
os.makedirs(output_path,exist_ok=True)


basis = get_aug_cc_pvtz_basis('He',grid_type='log',radius_cutoff=30.0,mesh_size=1000)
basis.print_basis_info()
integral = AtomicIntegrals(basis,nuclear_charge=2, real_basis=True,eri_cache=True)

# 静态 LSDA (Ground State)
lsda = AtomicLSDA(integral,strict_diagonalize=True) # Lithium doublet
gs_results = lsda.run_scf()

# RT-TDDFT 初始化
tddft = RealTimeTDDFT(gs_results, lsda,dt=0.05, strict_diagonalize=True,threshold=1e-6)

# 定义激光函数 E(t) = E0 * sin(w * t)
omega = 0.9 # a.u.
E0 = 0.002

def cw_laser(t):
    return E0 * np.sin(omega * t)

# 传入 field_func，不传 kick_params
tddft.propagate(
    total_time=1000.0,
    field_func=cw_laser,
    field_direction='z'
)

fig=plt.figure()
plt.plot(np.array(tddft.time_history), np.array(tddft.dipole_history)[:,2])
plt.xlabel('Time (a.u.)')
plt.ylabel('Dipole Moment (a.u.)')
plt.title('Li Dipole Moment under CW Laser')
plt.savefig(output_path/'li_cw_laser_dipole_time.png', dpi=300)
