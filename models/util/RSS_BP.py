import numpy as np
from models.real_scalar_singlet_model_boltz import RealScalarSingletModel_Boltz

number = 4
theta = 0.2623
BP = [-6.299125406135615322e+02, -9.096911651393214981e+01, theta, 6.637453498848074105e+02, 3.511829806788588257e+02]

potential = RealScalarSingletModel_Boltz(*BP)
BP = potential.getParameterPoint()

np.savetxt(f'input/RSS/RSS_new_BP{number}.txt', BP, newline=' ')
