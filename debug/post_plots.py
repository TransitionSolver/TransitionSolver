import traceback
import typing

import numpy as np
import matplotlib.pyplot as plt
import json
from cosmoTransitions import pathDeformation

from analysis import phase_structure
from analysis.phase_structure import Transition, Phase


# Various functions that may be useful for debugging an effective potential or a first-order phase transition.


def plotHubbleParameter():
    AonV = 0.129
    v = 250.
    #folderName = f'Output/scans/Standard234/CheckRadiationDomination/{AonV:.6f}_{v:.6f}'
    #folderName = 'Output/scans/Standard234/noPercolationOpt1'
    folderName = 'Output/scans/xSM_MSbar_mixing/pipeline/noPercolationOpt6-Full'
    fileName = '0.json'
    transitionIndex = 2
    report = None

    with open(folderName + '/' + fileName) as reportFile:
        report = json.load(reportFile)

    if report is None:
        print('Failed to read report:', folderName + '/' + fileName)
        return

    transition = report['transitions'][transitionIndex]
    if 'actionCurveFile' in transition:
        data = np.loadtxt(transition['actionCurveFile'])

        T = data[..., 0]
        SonT = data[..., 1]
        rho_V = data[..., 2]
    else:
        T = transition['T']
        SonT = transition['SonT']
        rho_V = transition['deltaV']
        T.reverse()
        SonT.reverse()
        rho_V.reverse()

    dof = 100.
    xi2 = 30/(np.pi**2*dof)

    if T[0] > 0.5:
        T.insert(0, 0.01)
        rho_V.insert(0, rho_V[0])

    rho_R = [t**4/xi2 for t in T]

    H = [np.sqrt(rho_R[i] + rho_V[i]) for i in range(len(T))]
    dHdT = [(H[i] - H[i-1])/(T[i] - T[i-1]) for i in range(1, len(T))]

    plt.plot(T, np.sqrt(rho_R))
    plt.plot(T, np.sqrt(rho_V))
    plt.plot(T, H)
    plt.plot(T[1:], dHdT)
    plt.axvline(transition['Teq'], ls='--', c='g')
    plt.axvline(transition['TGammaLow'], ls='--', c='r')
    plt.axvline(transition['Tmin'], ls='--', c='r')
    #plt.axvline(transition['TGammaHigh'], ls='--')
    plt.legend(['$\\rho_R$', '$\\rho_V$', '$\\rho$'])
    plt.show()

    #T.reverse()
    #SonT.reverse()
    #rho_V.reverse()
    deltaV = rho_V

    integral = [0]*len(T)
    integralApprox = [0]*len(T)
    integralImproved = [0]*len(T)
    H = [0]*len(T)
    Hlin = [0]*len(T)
    Happrox = [0]*len(T)
    vw = transition['vw']
    ndof = 107.75
    radDensity = np.pi**2/30*ndof

    H[0] = np.sqrt(radDensity*T[0]**4 + deltaV[0])
    Hlin[0] = H[0]
    Happrox[0] = np.sqrt(deltaV[0])

    indexTeq = -1
    indexTlow = -1
    indexThigh = -1
    indexTGammaHigh = -1

    Teq = transition['Teq']
    TGammaLow = transition['TGammaLow']
    try:
        TGammaHigh = transition['TGammaHigh']
    except KeyError:
        TGammaHigh = T[-1]
        indexTGammaHigh = len(T)-1
    Tmin = transition['Tmin']

    b1 = 0.78
    Tlow = b1*Teq
    #rhoRatio = np.sqrt(rho_V_Zero) / (np.sqrt(radDensity)*Teq**2)
    #d = (rhoRatio - np.sqrt(2)) / (1 - b1)
    d = (1 - np.sqrt(2)) / (1 - b1)
    #Thigh = Teq*(-0.5*d + 0.5*np.sqrt(d*d + 4*(b1*d + rhoRatio)))
    b2 = -0.5*d + 0.5*np.sqrt(d*d + 4*(b1*d + 1))
    Thigh = b2*Teq

    if Thigh > T[-1]:
        indexThigh = len(T)-1

    print('Tlow:', Tlow)
    print('Thigh:', Thigh)

    for i in range(len(T)):
        if indexTlow < 0 and T[i] > Tlow:
            indexTlow = i

        if indexTeq < 0 and T[i] > Teq:
            indexTeq = i
            print('rho_V(Teq):', deltaV[i])
            print('rho_V(0):', deltaV[0])
            print('ratio:', deltaV[i]/deltaV[0])

        if indexThigh < 0 and T[i] > Thigh:
            indexThigh = i

        if indexTGammaHigh < 0 and T[i] > TGammaHigh:
            indexTGammaHigh = i

    for i in range(1, len(T)):
        H[i] = np.sqrt(radDensity*T[i]**4 + deltaV[i])
        if i < indexTlow:
            Hlin[i] = np.sqrt(deltaV[0])
        elif indexTlow <= i <= indexThigh:
            interp = (T[i] - Tlow)/(Thigh-Tlow)
            Hlin[i] = np.sqrt(deltaV[0]) + (np.sqrt(radDensity)*Thigh**2 - np.sqrt(deltaV[0]))*interp
        else:
            Hlin[i] = np.sqrt(radDensity)*T[i]**2

        if i < indexTeq:
            Happrox[i] = np.sqrt(deltaV[i])
        else:
            Happrox[i] = np.sqrt(radDensity)*T[i]**2

        for j in range(1, i+1):
            # Trapezoid rule.
            integral[i] += 0.5*(1/H[j] + 1/H[j-1])*(T[j] - T[j-1])

        for j in range(1, min(i+1, indexTeq)):
            integralApprox[i] += 0.5*(1/np.sqrt(deltaV[j]) + 1/np.sqrt(deltaV[j-1]))*(T[j] - T[j-1])

        for j in range(indexTeq, i+1):
            integralApprox[i] += 0.5*(1/np.sqrt(radDensity*T[j]**4) + 1/np.sqrt(radDensity*T[j-1]**4))*(T[j] - T[j-1])

        integralImproved[i] = T[min(i, indexTlow)]/H[0]

        for j in range(max(1, indexTlow), min(i, indexThigh)):
            integralImproved[i] += 0.5*(1/Hlin[j] + 1/Hlin[j-1])*(T[j] - T[j-1])

        for j in range(indexThigh, i):
            integralImproved[i] += 0.5*(1/np.sqrt(radDensity*T[j]**4) + 1/np.sqrt(radDensity*T[j-1]**4))*(T[j] - T[j-1])

        integral[i] *= vw/T[i]*H[i]
        integralApprox[i] *= vw/T[i]*Happrox[i]
        integralImproved[i] *= vw/T[i]*Hlin[i]

    modifier = (1/b2 + b1 + (b2 - b1)/(b2*b2 - 1)*np.log(b2*b2))

    print('Expected at TGammaHigh    :', vw*TGammaHigh/Teq*modifier - vw)
    print('Approximated at TGammaHigh:', integralImproved[indexTGammaHigh])
    print('Actual at TGammaHigh      :', integral[indexTGammaHigh])
    print('No perc:', Teq - TGammaHigh*modifier/(1+0.44/vw))

    plt.plot(T, integral, c='g', marker='.')
    plt.plot(T, integralApprox, c='r', marker='.')
    plt.plot(T, integralImproved, c='k', marker='.')
    plt.axhline(0.44, ls='--')
    #plt.axvline(indexTmin, ls='--')
    #plt.axvline(indexTGammaLow, ls=':')
    plt.axvline(TGammaHigh, ls=':', c='r')
    plt.axvline(Tmin, ls='--', c='g')
    plt.axvline(Tlow, ls=':', c='b')
    plt.axvline(Thigh, ls=':', c='b')
    plt.axvline(Teq, ls='--', c='b')
    plt.margins(0.)
    plt.show()

    plt.plot(T, H, c='g', marker='.')
    plt.plot(T, Hlin, c='k', marker='.')
    plt.plot(T, Happrox, c='r', marker='.')
    plt.axvline(Tlow, ls=':')
    plt.axvline(Teq, ls='--')
    plt.axvline(Thigh, ls=':')
    plt.ylim(0.)
    plt.margins(0.)
    plt.show()


def plotActionCurve(fileName, transitionID=-1):
    extension = fileName.split('.')[-1]
    if extension == 'txt':
        data = np.loadtxt(fileName)

        T = data[..., 0]
        SonT = data[..., 1]
    elif extension == 'json':
        if transitionID == -1:
            print('Transition ID must be specified when loading JSON data.')
            return
        with open(fileName) as f:
            data = json.load(f)

            transDict = None
            for tr in data['transitions']:
                if tr['id'] == transitionID:
                    transDict = tr
                    break

            if transDict is not None:
                T = transDict['T']
                SonT = transDict['SonT']
            else:
                print('Unable to find transition with id =', transitionID, 'in the JSON file.')
                return
    else:
        print('Unsupported file type:', extension)
        return

    plt.plot(T, SonT, marker='.')
    plt.xlabel('$T$')
    plt.ylabel('$S(T)$')
    plt.show()


def plotPotentialBetweenPhases(potentialClass, folderName, transitionID, deltaT, actionTemps, plotPadding,
        numPoints=100):
    try:
        scanPoint = np.loadtxt(folderName+'/validScanPoint.txt')
    except OSError:
        print('Scan point file does not exist:', folderName+'/validScanPoint.txt')
        return

    potential: potentialClass = potentialClass(*scanPoint)
    bFileExists, phaseHistory = phase_structure.load_data(folderName + '/0.dat', bExpectFile=False)

    if not bFileExists:
        print('Phase history data file does not exist:', folderName+'/0.dat')
        return

    try:
        with open(folderName+'/report.json') as reportFile:
            report = json.load(reportFile)
    except json.decoder.JSONDecodeError:
        print('Failed to decode JSON report:', folderName+'/report.json')
        traceback.print_exc()
        return

    transitionReport: typing.Optional[dict] = None

    for trReport in report['transitions']:
        if trReport['id'] == transitionID:
            transitionReport = trReport
            break

    if transitionReport is None:
        print(f'Failed to find transition ID {transitionID} in JSON report.')
        return

    transition: typing.Optional[Transition] = None

    for tr in phaseHistory.transitions:
        if tr.ID == transitionID:
            transition = tr
            break

    if transition is None:
        print(f'Failed to find transition ID {transitionID} in phase history data.')
        return

    fromPhase: Phase = phaseHistory.phases[transition.false_phase]
    toPhase: Phase = phaseHistory.phases[transition.true_phase]

    try:
        Tn = transitionReport['Tn']
        Tpw = transitionReport['Tpw']
        Tps = transitionReport['Tps']
        Tf = transitionReport['Tf']
    except KeyError:
        print('Failed to extract transition temperature from JSON report:')
        traceback.print_exc()
        return

    TnRange = np.linspace(Tn, Tn+deltaT*4, 4)
    TpwRange = np.linspace(Tpw, Tpw+deltaT*4, 4)
    TpsRange = np.linspace(Tps, Tps+deltaT*4, 4)
    TfRange = np.linspace(Tf, Tf+deltaT*4, 4)
    Tlabels = ['Tn', 'Tpw', 'Tps', 'Tf']
    deltaTlabels = ['', f'+{deltaT:.2f}', f'+{deltaT*2:.2f}', f'+{deltaT*3:.2f}']

    temperatureRanges = [TnRange, TpwRange, TpsRange, TfRange]
    interpolation = np.linspace(0.0, 1.0, numPoints)
    subplotIDs = [221, 222, 223, 224]

    # For each of the temperature ranges around Tn, Tpw, Tps and Tf.
    for i in range(len(temperatureRanges)):
        tempRange = temperatureRanges[i]
        fromLoc = [fromPhase.findPhaseAtT(T, potential) for T in tempRange]
        toLoc = [toPhase.findPhaseAtT(T, potential) for T in tempRange]
        samplePoints = [np.array([fromLoc[T] + interp*(toLoc[T] - fromLoc[T]) for interp in interpolation])
            for T in range(len(tempRange))]
        V = [np.array([potential.Vtot(point, tempRange[j]) for point in samplePoints[j]]) for j in range(len(tempRange))]

        plt.subplot(subplotIDs[i])
        for j in range(len(tempRange)):
            plt.plot(interpolation, V[j])
        plt.legend([Tlabels[i]+deltaTlabels[j] for j in range(len(tempRange))])
        plt.xlabel('Interpolation')
        plt.ylabel('V')
        plt.title(Tlabels[i])
        plt.margins(0.0, 0.01)

    plt.show()

    if len(actionTemps) == 0:
        return

    tunnelingSolutions = []

    def V(X): return potential.Vtot(X, T)
    def gradV(X): return potential.gradV(X, T)

    minH = np.infty
    minS = np.infty
    maxH = -np.infty
    maxS = -np.infty

    tunneling_findProfile_params = {'phitol': 1e-6, 'xtol': 1e-6}

    for T in actionTemps:
        fromLoc = fromPhase.findPhaseAtT(T, potential)
        toLoc = toPhase.findPhaseAtT(T, potential)
        tunnelingSolutions.append(pathDeformation.fullTunneling([toLoc, fromLoc], V, gradV,
            tunneling_findProfile_params=tunneling_findProfile_params))

        minH = min(minH, fromLoc[0], toLoc[0])
        minS = min(minS, fromLoc[1], toLoc[1])
        maxH = max(maxH, fromLoc[0], toLoc[0])
        maxS = max(maxS, fromLoc[1], toLoc[1])

    SonT = [tunnelingSolutions[i].action / actionTemps[i] for i in range(len(tunnelingSolutions))]

    print('T:', actionTemps)
    print('S/T: ', SonT)

    minH -= plotPadding[0]
    minS -= plotPadding[1]
    maxH += plotPadding[0]
    maxS += plotPadding[1]

    h = np.linspace(minH, maxH, numPoints)
    s = np.linspace(minS, maxS, numPoints)
    V = np.zeros(shape=(numPoints, numPoints))
    H, S = np.meshgrid(h, s)

    for i in range(len(h)):
        for j in range(len(s)):
            V[j, i] = potential.Vtot(np.array([h[i], s[j]]), actionTemps[0])

    plt.figure()
    plt.contour(H, S, V, levels=[1.2e9+(i*1e6) for i in range(80)], linewidths=0.5)

    for ts in tunnelingSolutions:
        plt.plot(ts.Phi[:,0], ts.Phi[:,1], lw=1.5)

    plt.xlabel('$h$')
    plt.ylabel('$s$')
    plt.legend([f'T:{actionTemps[i]:.2f}, S:{SonT[i]:.2f}' for i in range(len(actionTemps))])
    plt.show()
