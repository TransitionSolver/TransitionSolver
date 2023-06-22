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



def writePhaseHistoryReport(fileName: str, paths: list[TransitionGraph.ProperPath], phaseStructure:
        PhaseStructure.PhaseStructure, analysisTime: float):
    report = {}

    if len(phaseStructure.transitions) > 0:
        report['transitions'] = [t.getReport(fileName) for t in phaseStructure.transitions]
    if len(paths) > 0:
        report['paths'] = [p.getReport() for p in paths]
    report['valid'] = any([p.bValid for p in paths])
    report['analysisTime'] = analysisTime

    print('Writing report...')

    try:
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
    except (json.decoder.JSONDecodeError, TypeError):
        print('We have a JSON serialisation error. The report is:')
        print(report)
        print('Failed to write report')


def main():
    # The folder where we will write the output.
    #outputFolder = 'output/example3'
    outputFolder = 'output/RSS/RSS_BP1'
    # Create the output folder if it doesn't exist already.
    pathlib.Path(str(pathlib.Path(outputFolder))).mkdir(parents=True, exist_ok=True)

    # Define a parameter point (i.e. values for each parameter in the model).
    #parameterPoint = [0.104005, 250, 3.5, 0.2]
    parameterPoint = np.loadtxt('input/RSS/RSS_BP1.txt')

    # Save the parameter point for future reference (e.g. so we know what parameter point resulted in the reported phase
    # history). Also, we can have PhaseTracer construct the potential by reading the saved parameter point.
    np.savetxt(outputFolder + '/parameter_point.txt', np.array([parameterPoint]))

    # Load the relative path to PhaseTracer from the config file.
    try:
        with open('config/config_user.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        traceback.print_exc()
        print('Unable to load configuration file.')
        sys.exit(1)

    try:
        PhaseTracer_directory = config['PhaseTracer_directory']
    except KeyError:
        print('Unable to load PhaseTracer directory from the configuration file.')
        sys.exit(1)

    try:
        windows = config['Windows']
    except KeyError:
        windows = False  # Just assume not Windows.

    if PhaseTracer_directory == '':
        sys.exit(1)

    # Call PhaseTracer to determine the phase structure of the potential. 'wsl' is the Windows Subsystem for Linux,
    # which is required because PhaseTracer does not run on Windows directly. The second element of the list is the
    # program name. The remaining elements are the input parameters for the specified program. The timeout (in seconds)
    # ensures that PhaseTracer cannot run indefinitely. shell=True is required so that WSL can be called from the shell.
    # stdout is routed to DEVNULL to suppress any print statements from PhaseTracer. stderr is routed to STDOUT so that
    # errors in PhaseTracer are printed here.


#subprocess.call([ PHASETRACER_DIR + 'bin/run_ToyModel', outputFolder + '/parameter_point.txt', outputFolder],timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
   # subprocess.call([ PHASETRACER_DIR + 'bin/run_SingletModel', outputFolder + '/parameter_point.txt', outputFolder],
   #     timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
   # subprocess.call([ PHASETRACER_DIR + 'bin/run_RSS', outputFolder + '/parameter_point.txt', outputFolder],
   #    timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


    command = (['wsl'] if windows else []) + [PhaseTracer_directory + 'bin/run_RSS', outputFolder +
        '/parameter_point.txt', outputFolder]
    subprocess.call(command, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


    # Load the phase structure saved by PhaseTracer.
    bFileExists, phaseStructure = PhaseStructure.load_data(outputFolder + '/phase_structure.dat', bExpectFile=True)

    # Validate the phase structure.
    if not bFileExists:
        print('Could not find phase structure file.')
        return
    if len(phaseStructure.transitionPaths) == 0:
        print('No valid transition path to the current phase of the Universe.')
        return

    # Load and configure a PhaseHistoryAnalyser object.
    analyser = PhaseHistoryAnalysis.PhaseHistoryAnalyser()
    analyser.bDebug = True
    analyser.bPlot = True
    analyser.bReportAnalysis = True
    analyser.bReportPaths = True
    analyser.timeout_phaseHistoryAnalysis = 100

    # Create the potential using the parameter point.
    #potential = ToyModel(*parameterPoint)
    #potential = SingletModel(*parameterPoint)
    potential = RealScalarSingletModel(*parameterPoint)

    # Analyse the phase history and track the execution time. The timing will be handled internally in a future version
    # of the code.
    startTime = time.perf_counter()
    paths, _ = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=0.95)
    endTime = time.perf_counter()

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolder + '/phase_history.json', paths, phaseStructure, endTime - startTime)


    



    #mbs = phaseStructure.transitions[0].meanBubbleSeparation
    #print("mean bubble seperation = ", mbs)
    #Tp = phaseStructure.transitions[0].Tp
    #print("Tp = ", Tp)
    #beta = (8 * np.phi)**(1/3) * vw / mean bubble separation
    #vw = phaseStructure.transitions[0].vw
    #beta = (8 * math.pi)**(1/3) * vw / mbs
    #print("beta = ",beta)
    
    
    Treh_p = phaseStructure.transitions[0].Treh_p
    
    mbsTp = phaseStructure.transitions[0].meanBubbleSeparation
    
    vwp = phaseStructure.transitions[0].vw
 
    betaTp = phaseStructure.transitions[0].analysis.betaTp

    #Tp = phaseStructure.transitions[0].Tp
    #print("Tp = ",Tp)
    
    gp = potential.ndof
    
    Hp = phaseStructure.transitions[0].analysis.Hp
    
    #thetaf = Hydrodynamics.HydroVars.pseudotraceFalse
    #thetat = Hydrodynamics.HydroVars.pseudotraceTrue
    #print("thetaf = ",thetaf)
    #print("thetat = ",thetat)
    alphap = phaseStructure.transitions[0].transitionStrength 
    
    kappap = alphap/(0.73 + 0.083 * math.sqrt(alphap) + alphap)
    
    K = kappap*alphap/(1+alphap)
    
    tau_sw = mbsTp * math.sqrt((4/3)/K) 
    
    upsilon_old = min(1,Hp*tau_sw)
    
    upsilon_new = 1 - 1 / np.sqrt(1 + 2*Hp*tau_sw)
    
    epsilon_old = (1-upsilon_old)**(2/3)
    
    epsilon_new = (1-upsilon_new)**(2/3)
    
    
    
    print("Treh_p = ", Treh_p)
    print("vwTp = ", vwp)
    print("mbsTp = ", mbsTp)
    print("K = ",K)
    print("tau_sw = ", tau_sw)
    print("Hp = ",Hp)
    print("Hp*tau_sw = ", Hp*tau_sw)
    
    
    print("betaTp = ",betaTp)
    print("gp = ",gp)
    print("Hp = ", Hp)
    print("alphap = ",alphap)
    print("kappap = ",kappap)
    
    print("upsilon_old = ",upsilon_old)
    print("upsilon_new = ",upsilon_new)
    print("epsilon_old = ",epsilon_old)
    print("epsilon_new = ",epsilon_new)
    
    
    
    
    
    
    def fsw(mbs,H,T,g):
        fsw = 8.9*10**(-6)*(g/100)**(1/6)*(T/100)*((8*np.pi)**(1/3))/(H*mbs) 
        return fsw
    
    def Omegaswh2(mbs,H,alpha,g,T,kappa,f):
        Omegaswh2 = 2.59*10**(-6)*(100/g)**(1/3)*(kappa*alpha/(1+alpha))**2*H*mbs/((8*np.pi)**(1/3))*(f/fsw(mbs,H,T,g))**3*(7/(4+3*(f/fsw(mbs,H,T,g))**2))**(7/2) * upsilon_new
        return Omegaswh2


    def fturb(mbs,H,T,g):
        fturb = 2.7*10**(-5)* (g/100)**(1/6)* (T/100) * ((8*np.pi)**(1/3))/(mbs*H) 
        return fturb

    def h(T,g): 
        h = 16.5*10**(-6)*(T/100)*(g/100)**(1/6)
        return h
    
    def Omegaturbh2(H,mbs,alpha,g,T,kappa,f):
        #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((0.1*kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/h(T,g))))
        #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H/beta)*((0.1*kappa*alpha)/(1+alpha))**(3/2)*vw*((f/fturb(vw,beta,H,T,g))**3)/(((1+(f/fturb(vw,beta,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
        
        
        Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((0.1*kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
        #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((0.05*kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
        #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((epsilon_old * kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
        #Omegaturbh2 = 3.35*10**(-4)*(100/g)**(1/3)*(H*mbs/(8*np.pi)**(1/3))*((epsilon_new * kappa*alpha)/(1+alpha))**(3/2)*((f/fturb(mbs,H,T,g))**3)/(((1+(f/fturb(mbs,H,T,g)))**(11/3)) * (1+(8*math.pi*f/H)))
        return Omegaturbh2

    def Omegatotalh2(H,mbs,alpha,g,T,kappa,f):
        Omegatotalh2 = Omegaswh2(H,mbs,alpha,g,T,kappa,f) + Omegaturbh2(H,mbs,alpha,g,T,kappa,f)
        return Omegatotalh2

    print("Omegatotalh2(Hp,mbsTp,alphap,gp,Tp,kappap,f) = ",Omegatotalh2(Hp,mbsTp,alphap,gp,Treh_p,kappap,1))

    fx=np.linspace(10**(-8), 0.01, 10000)
    #fx=np.linspace(10**(-8), 100, 10000000)
    snry1=[]
    for f in fx:
        ss1=Omegaswh2(Hp,mbsTp,alphap,gp,Treh_p,kappap,f)
        snry1.append(ss1)
        
    fx=np.linspace(10**(-8) , 0.01, 10000) 
    #fx=np.linspace(10**(-8) , 100, 10000000)  
    snry2=[]
    for f in fx:
        ss2=Omegaturbh2(Hp,mbsTp,alphap,gp,Treh_p,kappap,f)
        snry2.append(ss2)
        
    fx=np.linspace(10**(-8) , 0.01, 10000) 
    #fx=np.linspace(10**(-8) , 100, 10000000)   
    snry3=[]
    for f in fx:
        ss3=Omegatotalh2(Hp,mbsTp,alphap,gp,Treh_p,kappap,f)
        snry3.append(ss3)

    
    fig,ax=plt.subplots(1,1,figsize=(8,6),dpi=180,facecolor='white')
    plt.plot(fx,snry1[:],color='blue')
    plt.plot(fx,snry2[:],color='green')
    plt.plot(fx,snry3[:],color='red')
    
    plt.xscale('log') 
    plt.yscale('log') 
    #plt.xscale('linear')
    #plt.yscale('linear')
    plt.xlabel('f')
    plt.ylabel('h2Omega')
    #plt.xlim(10**(-8), 100)
    plt.xlim(10**(-8), 0.01)
    plt.ylim(10**(-36), 10**(-2))
    #plt.legend(['sw','turb','tot'],bbox_to_anchor=(1.05, 0.7), loc=3, borderaxespad=0) 
    plt.legend(['sw','turb','tot'])
    plt.show()
if __name__ == "__main__":
    main()