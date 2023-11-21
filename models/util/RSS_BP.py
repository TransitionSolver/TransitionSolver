import numpy as np
from models.real_scalar_singlet_model_boltz import RealScalarSingletModel_Boltz
import sys

number = 5

if number == 1:
    theta = 0.24
elif number == 2:
    theta = 0.258
elif number == 3:
    theta = 0.262
elif number == 4:
    theta = 0.2623
elif number == 5:
    theta = 0.1
else:
    print('Unsupported benchmark number:', number)
    sys.exit(1)

BP = [-6.299125406135615322e+02, -9.096911651393214981e+01, theta, 6.637453498848074105e+02, 3.511829806788588257e+02]

potential = RealScalarSingletModel_Boltz(*BP)
BP = potential.getParameterPoint()

np.savetxt(f'input/RSS/RSS_new_BP{number}.txt', BP, newline=' ')
