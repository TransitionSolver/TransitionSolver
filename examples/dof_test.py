from models.real_scalar_singlet_model import RealScalarSingletModel
from analysis import phase_structure
import json
import numpy as np
import sys
import matplotlib.pyplot as plt

potentialClass = RealScalarSingletModel
outputFolder = 'output/RSS/RSS_BP1/'

with open(outputFolder + 'phase_history.json', 'r') as f:
    phaseHistoryReport = json.load(f)

bSuccess, phaseStructure = phase_structure.load_data(outputFolder + 'phase_structure.dat')

if not bSuccess:
    print('Failed to load phase structure.')
    sys.exit(1)

potential = potentialClass(*np.loadtxt(outputFolder + 'parameter_point.txt'))

fig, ax = plt.subplots(figsize=(12, 8))

for phase in phaseStructure.phases:
    #T = np.linspace(phase.T[0], phase.T[-1], 100)
    T = np.logspace(-2, np.log10(phase.T[-1])-0.01, 100)*1000
    dof = [potential.getDegreesOfFreedomInPhase(phase, t/1000) for t in T]

    plt.plot(T, dof, linewidth=2.5, label=f'Phase {phase.key}')

#ax = plt.gca()
ax.invert_xaxis()
plt.xscale('log')
plt.xlabel('$T \;\; \mathrm{[MeV]}$', fontsize=24)
plt.ylabel('$g_{\mathrm{eff}}$', fontsize=24)
plt.tick_params(size=10, labelsize=20)
plt.ylim(0, potential.ndof*1.05)
plt.margins(0, 0)
plt.legend(fontsize=20)
plt.show()
