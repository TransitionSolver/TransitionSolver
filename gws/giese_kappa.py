# Currently copied straight from Appendix B of https://arxiv.org/pdf/2010.09744.pdf and Appendix A of 2004.06995.

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import simps


def mu(a,b):
    return (a-b)/(1.-a*b)


def getwow(a,b):
    return a/(1.-a**2)/b*(1.-b**2)


def getvm(al,vw,cs2b):
    if vw**2<cs2b:
        return (vw,0)
    cc = 1.-3.*al+vw**2*(1./cs2b+3.*al)
    disc = -4.*vw**2/cs2b+cc**2
    if (disc<0.)|(cc<0.):
        return (np.sqrt(cs2b), 1)
    return ((cc+np.sqrt(disc))/2.*cs2b/vw, 2)


def dfdv(xiw, v, cs2):
    xi, w = xiw
    dxidv = (mu(xi,v)**2/cs2-1.)
    dxidv *= (1.-v*xi)*xi/2./v/(1.-v**2)
    dwdv = (1.+1./cs2)*mu(xi,v)*w/(1.-v**2)
    return [dxidv,dwdv]


def getKandWow(vw,v0,cs2):
    if v0==0:
        return 0,1
    n = 8*1024 # change accuracy here
    vs = np.linspace(v0, 0, n)
    sol = odeint(dfdv, [vw,1.], vs, args=(cs2,))
    xis, wows = (sol[:,0],sol[:,1])
    if mu(vw,v0)*vw<=cs2:
        ll=max(int(sum(np.heaviside(cs2-(mu(xis,vs)*xis),0.0))),1)
        vs = vs[:ll]
        xis = xis[:ll]
        wows = wows[:ll]/wows[ll-1]*getwow(xis[-1], mu(xis[-1],vs[-1]))
    Kint = simps(wows*(xis*vs)**2/(1.-vs**2), xis)
    return (Kint*4./vw**3, wows[0])


def alN(al,wow,cs2b,cs2s):
    da = (1./cs2b - 1./cs2s)/(1./cs2s + 1.)/3.
    return (al+da)*wow -da


def getalNwow(vp,vm,vw,cs2b,cs2s):
    Ksh,wow = getKandWow(vw,mu(vw,vp),cs2s)
    al = (vp/vm-1.)*(vp*vm/cs2b - 1.)/(1-vp**2)/3.
    return (alN(al,wow,cs2b,cs2s), wow)


def kappaNuMuModel(cs2b,cs2s,al,vw):
    vm, mode = getvm(al,vw,cs2b)
    if mode<2:
        almax,wow = getalNwow(0,vm,vw,cs2b,cs2s)
        if almax<al:
            print ("alpha too large for shock")
            raise Exception("alpha too large for shock")
            return 0
        vp = min(cs2s/vw,vw)
        almin,wow = getalNwow(vp,vm,vw,cs2b,cs2s)
        if almin>al:
            print ("alpha too small for shock")
            raise Exception("alpha too small for shock")
            return 0
        iv = [[vp,almin],[0,almax]]
        while (abs(iv[1][0]-iv[0][0])>1e-7):
            vpm = (iv[1][0]+iv[0][0])/2.
            alm = getalNwow(vpm,vm,vw,cs2b,cs2s)[0]
            if alm>al:
                iv = [iv[0],[vpm,alm]]
            else:
                iv = [[vpm,alm],iv[1]]
        vp = (iv[1][0]+iv[0][0])/2.
        Ksh,wow = getKandWow(vw,mu(vw,vp),cs2s)
    else:
        Ksh,wow,vp = (0,1,vw)
    if mode>0:
        Krf,wow3 = getKandWow(vw,mu(vw,vm),cs2b)
        Krf*= -wow*getwow(vp,vm)
    else:
        Krf = 0
    return (Ksh + Krf)/al


def kappaNuModel(cs2,al,vp,n=501):
    nu = 1./cs2+1.
    tmp = 1.-3.*al+vp**2*(1./cs2+3.*al)
    disc = 4*vp**2*(1.-nu)+tmp**2
    if disc<0:
        print("vp too small for detonation")
        print("vp = ", vp, "al = ", al, "cs2 = ", cs2, "disc =", disc)
        raise Exception("The bubble wall velocity is too small for detonation.   Currently the GWs module only supports detonations for the computation of sound wave sourced gravitational waves.  This requires going beyond the kappaNuModel from https://arxiv.org/abs/2004.069959 and needs the KappaNuMu model (https://arxiv.org/abs/2010.09744). We expect this to be addressed soon and the code will be updated.")
        return 0
    vm = (tmp+np.sqrt(disc))/2/(nu-1.)/vp
    wm = (-1.+3.*al+(vp/vm)*(-1.+nu+3.*al))
    wm /= (-1.+nu-vp/vm)

    def dfdv(xiw, v, nu):
        xi, w = xiw
        dxidv = (((xi-v)/(1.-xi*v))**2*(nu-1.)-1.)
        dxidv *= (1.-v*xi)*xi/2./v/(1.-v**2)
        dwdv = nu*(xi-v)/(1.-xi*v)*w/(1.-v**2)
        return [dxidv,dwdv]

    #n = 501 # change accuracy here
    #n = int(2e8+1)
    vs = np.linspace((vp-vm)/(1.-vp*vm), 0, n)
    sol = odeint(dfdv, [vp,1.], vs, args=(nu,))
    xis, ws = (sol[:,0],-sol[:,1]*wm/al*4./vp**3)

    return simps(ws*(xis*vs)**2/(1.-vs**2), xis)


if __name__ == "__main__":
    print(kappaNuModel(0.3333684071549121, 4942.716815118201, 0.9999819356748879))
