import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
output_path="He_RT_TDDFT_Spectrum"
os.makedirs(output_path, exist_ok=True)
output_path=Path(output_path)
basis = get_aug_cc_pvtz_basis('He',grid_type='log',radius_cutoff=30.0,mesh_size=2000)
basis.print_basis_info()
integral = AtomicIntegrals(basis,nuclear_charge=2, real_basis=True,eri_cache=True)

# 静态 LSDA (Ground State)
lsda = AtomicLSDA(integral,strict_diagonalize=True,diagonalization_threshold=1e-12) 

gs_results = lsda.run_scf()
print_ks_results(gs_results)
analyze_orbital_compositions(gs_results)
tddft = RealTimeTDDFT(gs_results,lsda,dt=0.05, strict_diagonalize=False)

strength = 0.001
tddft.propagate(total_time=500.0, print_interval=100, kick_params={
    'strength': strength,'direction': 'z'
})
# 5. 分析光谱
print("\n>>> Step 4: Spectrum Analysis")
spectrum = tddft.calculate_spectrum(kick_strength=strength, damping=0.003)
fig=plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(np.array(tddft.time_history), np.array(tddft.dipole_history)[:,0])
plt.title('Dipole Moment x')
plt.xlabel('Time (a.u.)')
plt.subplot(1,3,2)
plt.plot(np.array(tddft.time_history), np.array(tddft.dipole_history)[:,1])
plt.title('Dipole Moment y')
plt.xlabel('Time (a.u.)')
plt.subplot(1,3,3)
plt.plot(np.array(tddft.time_history), np.array(tddft.dipole_history)[:,2])
plt.title('Dipole Moment z')
plt.xlabel('Time (a.u.)')
plt.savefig(output_path/'lithium_rt_tddft_dipole_time.png', dpi=300)

sp=tddft.calculate_spectrum(kick_strength=strength, damping=0.01)
fig=plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(tddft.time_history, np.array(tddft.dipole_history)[:,2],c="b")
plt.grid()
plt.xlabel('Time (a.u.)')
plt.ylabel('Dipole Moment (a.u.)')
plt.subplot(2,1,2)
plt.plot(sp["energy_ev"], sp['intensity'], label='RT-TDDFT Spectrum',c="r")
plt.xlim(0,50)
plt.grid()
plt.xlabel('Energy (eV)')
plt.ylabel('Absorption Intensity (a.u.)')
plt.savefig(output_path/'rt_tddft_spectrum.png',dpi=300)
