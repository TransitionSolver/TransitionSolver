from __future__ import annotations
from models.ToyModel import ToyModel
from models.SingletModel import SingletModel
from models.RealScalarSingletModel import RealScalarSingletModel
from analysis import PhaseStructure, PhaseHistoryAnalysis, TransitionGraph
import numpy as np
import subprocess
import json
import time
import pathlib
from gws import GieseKappa, Hydrodynamics
from gws.Hydrodynamics import HydroVars
import math
import matplotlib.pyplot as plt



# The file path to PhaseTracer. This is user specific.
#PHASETRACER_DIR = '/home/xuzhongxiu/PhaseTracer/'

import traceback
import sys


Treh_p =  42.920478890565676
vwTp =  0.95
mbsTp =  354990597381582.9
K =  0.3002456512638232
tau_sw =  748079675255923.1
Hp =  2.447574506784356e-15
betaTp =  -2.9933190304214253e-13
gp =  107.75
alphap =  1.0986892685273835
kappap =  0.5735218721795461
upsilon_old =  1
upsilon_new =  0.5368564081941845
epsilon_old =  0.0
epsilon_new =  0.5986097316621057


def fsw(mbs, H, T, g):
    fsw = 8.9 * 10**(-6) * (g/100)**(1/6) * (T/100) * ((8 * np.pi)**(1/3)) / (H * mbs)
    return fsw

def Omegaswh2(mbs, H, alpha, g, T, kappa, f):
    Omegaswh2 = 2.59 * 10**(-6) * (100/g)**(1/3) * (kappa * alpha/(1+alpha))**2 * H * mbs / ((8 * np.pi)**(1/3)) * (f/fsw(mbs, H, T, g))**3 * (7/(4 + 3*(f/fsw(mbs, H, T, g))**2))**(7/2) * upsilon_new
    return Omegaswh2

def fturb(mbs, H, T, g):
    fturb = 2.7 * 10**(-5) * (g/100)**(1/6) * (T/100) * ((8 * np.pi)**(1/3)) / (mbs * H)
    return fturb

def h(T, g):
    h = 16.5 * 10**(-6) * (T/100) * (g/100)**(1/6)
    return h

def Omegaturbh2(H, mbs, alpha, g, T, kappa, f):
    Omegaturbh2 = 3.35 * 10**(-4) * (100/g)**(1/3) * (H * mbs / (8 * np.pi)**(1/3)) * ((0.1 * kappa * alpha) / (1 + alpha))**(3/2) * ((f/fturb(mbs, H, T, g))**3) / (((1 + (f/fturb(mbs, H, T, g)))**(11/3)) * (1 + (8 * math.pi * f/H)))
    #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((0.05*kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
    #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((epsilon_old * kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
    #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((epsilon_new * kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
    return Omegaturbh2

def Omegatotalh2(H, mbs, alpha, g, T, kappa, f):
    Omegatotalh2 = Omegaswh2(H, mbs, alpha, g, T, kappa, f) + Omegaturbh2(H, mbs, alpha, g, T, kappa, f)
    return Omegatotalh2

#print("Omegatotalh2(Hp, mbsTp, alphap, gp, Tp, kappap, f) =", Omegatotalh2(Hp, mbsTp, alphap, gp, Treh_p, kappap, 0.1))

#fx = np.linspace(10**(-8), 0.01, 10000)
fx=np.logspace(-8, 3, 1000)
snry1 = []
for f in fx:
    ss1 = Omegaswh2(Hp, mbsTp, alphap, gp, Treh_p, kappap, f)
    snry1.append(ss1)

#fx = np.linspace(10**(-8), 0.01, 10000)
fx=np.logspace(-8, 3, 1000)
snry2 = []
for f in fx:
    ss2 = Omegaturbh2(Hp, mbsTp, alphap, gp, Treh_p, kappap, f)
    snry2.append(ss2)

#fx = np.linspace(10**(-8), 0.01, 10000)
fx=np.logspace(-8, 3, 1000)
snry3 = []
for f in fx:
    ss3 = Omegatotalh2(Hp, mbsTp, alphap, gp, Treh_p, kappap, f)
    snry3.append(ss3)


fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=180, facecolor='white')
plt.plot(fx, snry1[:], color='blue')
plt.plot(fx, snry2[:], color='green')
plt.plot(fx, snry3[:], color='red')

plt.xscale('log')
plt.yscale('log')
#plt.xscale('linear')
#plt.yscale('linear')
plt.xlabel('f')
plt.ylabel('h2Omega')
#plt.xlim(10**(-8), 100)
plt.xlim(10**(-8), 1000)
plt.ylim(10**(-36), 10**(-2))
#plt.legend(['sw', 'turb', 'tot'], bbox_to_anchor=(1.05, 0.7), loc=3, borderaxespad=0)
plt.legend(['sw', 'turb', 'tot'])
plt.show()


