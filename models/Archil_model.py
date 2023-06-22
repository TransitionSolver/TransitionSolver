import matplotlib.pyplot as plt
import numpy as np
from cosmoTransitions import generic_potential

v2 = 246.**2


class model1(generic_potential.generic_potential):
    """
    A sample model which makes use of the *generic_potential* class.
    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    # kap = [-1.87,-1.88,-1.89,-1.9,-1.91,-1.92]*125.**2 /246.
    def init(self,kap=-121.95,yt=0.9946,g=0.6535,g1=0.35):
    
        # The init method is called by the generic_potential class, after it
        # already does some of its own initialization in the default __init__()
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses.
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 1

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v2

        # This next block sets all of the parameters that go into the potential
        # and the masses. This will obviously need to be changed for different
        # models.
       
        self.kap = kap
        self.mu2 = .5*(125.**2 +np.sqrt(v2)*self.kap)
        self.lam = .5/v2 *(125.**2 -np.sqrt(v2)*self.kap)
        self.yt = yt
        self.g = g
        self.g1 = g1        
    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -5. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
        return (np.array([X])[...,0] < -5.0).any()

    def V0(self, X):
        """
        This method defines the tree-level potential. It should generally be
        subclassed. (You could also subclass Vtot() directly, and put in all of
        quantum corrections yourself).
        """
        # X is the input field array. It is helpful to ensure that it is a
        # numpy array before splitting it into its components.
        X = np.asanyarray(X)
        # x and y are the two fields that make up the input. The array should
        # always be defined such that the very last axis contains the different
        # fields, hence the ellipses.
        # (For example, X can be an array of N two dimensional points and have
        # shape (N,2), but it should NOT be a series of two arrays of length N
        # and have shape (2,N).)
        rho = X[...,0]
        r = -.5*self.mu2*rho**2 + self.kappa*rho**3 /3 + .25*self.lam*rho**4
        return r

    def boson_massSq(self, X, T):
        X = np.array(X)
        rho = X[...,0]

        # We need to define the field-dependnet boson masses. This is obviously
        # model-dependent.
        # Note that these can also include temperature-dependent corrections.
        h2 = 3.*self.lam*rho**2 +2.*self.kap*rho -self.mu2
        W2 = (self.g/2)**2 *rho**2
        Z2 = (np.sqrt(self.g**2 +self.g1**2)/2)**2 *rho**2
        M = np.array([h2,W2,Z2])

        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.
        dof = np.array([1, 6, 3])

        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        c = np.array([1.5, 5/6, 5/6])

        return M, dof, c
        
    def fermion_massSq(self, X):
        X = np.array(X)
        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `X`. The
            shape should be such that  ``massSq.shape == (X[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.

        Notes
        -----
        Unlike :func:`boson_massSq`, no constant `c` is needed since it is
        assumed to be `c = 3/2` for all fermions. Also, no thermal mass
        corrections are needed.
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        Nfermions = 1
        rho = X[...,0]   # Comment out so that the placeholder doesn't
                         # raise an exception for Ndim < 2.
        m12 = (self.yt/np.sqrt(2))**2 *rho**2 # First fermion mass
        massSq = np.array([m12])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))
        dof = np.array([12])
        return massSq, dof


    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        return [np.array([0]), np.array([np.sqrt(v2)])]


#    def plotPhasesPhi(self, **plotArgs):
#        import matplotlib.pyplot as plt
#        if self.phases is None:
#            self.getPhases()
#        for key, p in self.phases.items():
#            phi_hmag = (p.X[...,0]**2)**.5
#            phi_smag = (p.X[...,1]**2)**.5
           #print("hphimag=",hphimag)
           
#            fig = plt.figure('Minima as a function of temperature')
            
#            ax = plt.subplot(121)
#            plt.plot( phi_hmag, p.T, **plotArgs)
#            plt.xlabel(R"$\phi_h(T)$")
#            plt.ylabel(R"$T$")
#            plt.axis([-10,300,0,400])
            #plt.title("Minima as a function of temperature")
            
#            ax = plt.subplot(122)
#            plt.plot( phi_smag, p.T, **plotArgs)
#            plt.xlabel(R"$\phi_s(T)$")
#            plt.ylabel(R"$T$")
#            plt.axis([-10,300,0,400])
            #plt.title("Minima as a function of temperature")
            
#            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=None)
       
            

             
            
            
        
def makePlots(m=None):
    import matplotlib.pyplot as plt
    if m is None:
        #m=model1(kap,yt,g,g1)
        m = model1(-121.95,0.9946,0.6535,0.35)
        #m = model1(-8720.10,-2430.215,0.129,0.025,0.15,246,0,0.9946,0.6535,0.35)
        m.findAllTransitions()
    # --
    plt.figure()
    m.plotPhasesPhi()
    plt.show()
    # --
    plt.figure(figsize=(8,3))
    ax = plt.subplot(131)
    T = 0
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_xlabel(R"$\phi_1$")
    ax.set_ylabel(R"$\phi_2$")
    
    ax = plt.subplot(132)
    T = m.TnTrans[1]['Tnuc']
    instanton = m.TnTrans[1]['instanton']
    phi = instanton.Phi
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.plot(phi[:,0], phi[:,1], 'k')
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_yticklabels([])
    ax.set_xlabel(R"$\phi_1$")
    
    ax = plt.subplot(133)
    T = m.TnTrans[0]['Tnuc']
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_yticklabels([])
    ax.set_xlabel(R"$\phi_1$")
    plt.show()
    
    
    # --
    plt.figure()
    plt.plot(instanton.profile1D.R, instanton.profile1D.Phi)
    plt.xlabel("radius")
    plt.ylabel(R"$\phi-\phi_{min}$ (along the path)")
    plt.title("Tunneling profile")
    plt.show()

if __name__ == "__main__":
  makePlots()  


