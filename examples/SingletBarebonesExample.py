from __future__ import annotations
from models.SingletModel import SingletModel
from analysis import PhaseStructure, PhaseHistoryAnalysis, TransitionGraph
import numpy as np
import subprocess
import json
import time
import pathlib



# The relative file path to PhaseTracer. This is user specific.
PHASETRACER_DIR = '/home/xuzhongxiu/PhaseTracer/'


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
    outputFolder = 'output/singletexample'
    # Create the output folder if it doesn't exist already.
    pathlib.Path(str(pathlib.Path(outputFolder))).mkdir(parents=True, exist_ok=True)

    # Define a parameter point (i.e. values for each parameter in the model).
    # parameterPoint =[mu2,mus2,lamdah,lamdas,lamdahs,v]
    parameterPoint = [-8720.10,-2430.215,0.129,0.025,0.15,246]
    #parameterPoint = [-8720.1, -4860.43, 0.129, 0.1, 0.3, 246]

    #输入参数参考文献：2208.01319
    # Save the parameter point for future reference (e.g. so we know what parameter point resulted in the reported phase
    # history). Also, we can have PhaseTracer construct the potential by reading the saved parameter point.
    np.savetxt(outputFolder + '/parameter_point.txt', np.array([parameterPoint]))

    # Call PhaseTracer to determine the phase structure of the potential. 'wsl' is the Windows Subsystem for Linux,
    # which is required because PhaseTracer does not run on Windows directly. The second element of the list is the
    # program name. The remaining elements are the input parameters for the specified program. The timeout (in seconds)
    # ensures that PhaseTracer cannot run indefinitely. shell=True is required so that WSL can be called from the shell.
    # stdout is routed to DEVNULL to suppress any print statements from PhaseTracer. stderr is routed to STDOUT so that
    # errors in PhaseTracer are printed here.
    subprocess.call([ PHASETRACER_DIR + 'bin/run_SingletModel', outputFolder + '/parameter_point.txt', outputFolder],
        timeout=60)


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
    potential = SingletModel(*parameterPoint)
    vtest =  246.0
    stest = 100.0
    print("potential = ", potential.Vtot([vtest, stest], 100) )
    print("V0 = ", potential.V0([vtest, stest]))
    #print("boson_massSq = ", potential.boson_massSq([vtest, stest],0))

    # 调用 boson_massSq 函数，并将结果存储在变量 results 中
    results = potential.boson_massSq([vtest, stest], 100)

    # 分别获取结果中的各个部分
    M = results[0]
    dof = results[1]
    c = results[2]

    # 输出 A+B
    print("A+B =", M[..., 0])

    # 输出 A-B
    print("A-B =", M[..., 1])

    # 输出 W
    print("W =", M[..., 2])

    # 输出 Z
    print("Z =", M[..., 3])

    
    # 调用 fermion_massSq 函数获取结果
    X = np.array([vtest, stest])
    massSq, dof = potential.fermion_massSq(X)

    # 输出结果到终端
    print("Top quark Mass Squared:", massSq)
    print("Degrees of Freedom:", dof)

    
    
    #print("rho_gs = ", potential.Vtot([vtest, 0], 0) )
    #Tm = 0.0
    #rhogstemp = potential.Vtot([vtest, 0], 0)
    #ta 
    #ta.calculateEnergyDensityAtT(Tm)
    #rhoftemp = ta.calculateEnergyDensityAtT(Tm)[0]
    #print("rhotot = ", rhoftemp - rhogstemp)
    #rhotottemp =  rhoftemp - rhogstemp
    #rhoRtemp = np.pi**2/30*potential.ndof*Ttest**4
    #rhoV =rhotottemp -  rhoRtemp
    #print("rho_V=", rhoV )
   
    # Analyse the phase history and track the execution time. The timing will be handled internally in a future version
    # of the code.
    startTime = time.perf_counter()
    paths, _ = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=1.)
    endTime = time.perf_counter()

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolder + '/phase_history.json', paths, phaseStructure, endTime - startTime)


if __name__ == "__main__":
    main()
