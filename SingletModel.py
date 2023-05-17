from __future__ import annotations
from AnalysablePotential import AnalysablePotential
from cosmoTransitions import generic_potential
import numpy as np

v2 = 246.**2


class SingletModel(AnalysablePotential):
    """
    A sample model which makes use of the *generic_potential* class.
    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    def init(self,mu2,mus2,lamda,lamda2,lamda3,v):
    
        # The init method is called by the generic_potential class, after it
        # already does some of its own initialization in the default __init__()
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses.
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 2

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v2

        # This next block sets all of the parameters that go into the potential
        # and the masses. This will obviously need to be changed for different
        # models.
       
        self.v = v
        self.mus2 = mus2
        self.mu2 = mu2
        self.lamda32 = lamda3**2
        self.lamda, self.lamda2, self.lamda3 = lamda, lamda2, lamda3
         
        
        
        # Used to determine the overall scale of the problem. This is used (for example) to estimate what is a
        # reasonable step size for a small offset in field space (e.g. for derivatives). Of course, in such an
        # application, a small fraction of this scale is used as the offset.
        self.fieldScale = self.v
        self.temperatureScale = self.v

        # Stop analysing the transition below this temperature.
        self.minimumTemperature = 0.1

        # The number of degrees of freedom in the model. In the Standard Model, this would be 106.75.
        self.ndof = 107.75
        # The number of degrees of freedom that are not included in the one-loop corrections. These need to be accounted
        # for in the free energy density to correctly determine quantities that depend on the energy density (e.g. the
        # Hubble rate). If all degrees of freedom are present in the one-loop corrections, this should be set to zero.
        self.raddof = 83.25

        # The zero point of the energy density. Typically, this is taken to be the free energy density in the global
        # minimum at zero temperature. If the current state of the Universe corresponds to the global minimum, then this
        # effectively sets the cosmological constant to zero.
        self.groundStateEnergy = self.Vtot([self.v,0], 0.)
        
               
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
        
        
        #V0=0.5*mu^2*phi^2 + 0.25*lamda*phih^4 + mus^2*S^2 + lamda2*S^4 + 0.5*lamda3*S^2*phih^2
        #\phi = (0, 1/\sqrt{2}h^0 )^T and S = s^0
        h,s = X[...,0], X[...,1]
        r=0.5*self.mu2*h**2 + 0.25*self.lamda*h**4 + self.mus2*s**2 + self.lamda2*s**4 + 0.5*self.lamda3*s**2*h**2
        return r

    def boson_massSq(self, X, T):
        X = np.array(X)
        h,s = X[...,0], X[...,1]

        # We need to define the field-dependnet boson masses. This is obviously
        # model-dependent.
        # Note that these can also include temperature-dependent corrections.
        
        
        # replace v with h^0 and v_s with s^0 
        a = (-self.lamda * (self.v)**2) + 3* self.lamda * h**2 + self.lamda3 * s**2
        b = 2*self.mu2 + 12* self.lamda2 * s**2 + self.lamda3 * h**2
        c0 = 4*self.lamda32 * h**2 * s**2
        A = 0.5*(a+b)
        B = 0.5*( np.sqrt((a-b)**2 + 4*c0))
        W2=(0.6535366*h/2)**2
        Z2=(h*np.sqrt(0.6535366**2+0.35**2)/2)**2
        M = np.array([A+B, A-B,W2,Z2])

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
        dof = np.array([1, 1, 3, 3])

        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        c = np.array([1.5, 1.5, 5/6, 5/6])

        return M, dof, c
        
    def fermion_massSq(self, X):
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
        h = X[...,0]
        s = X[...,1] # Comment out so that the placeholder doesn't
                         # raise an exception for Ndim < 2.
        m12 = (0.99472*h/np.sqrt(2))**2 # First fermion mass
        massSq = np.array([m12])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))
        dof = np.array([12])
        return massSq, dof

    def Vtot(self, X, T, include_radiation=True):
        """
        The total finite temperature effective potential.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        return y
        
        
    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        return [np.array([0,0]), np.array([self.v,0])]
        


# Returns a list of the parameters value that define this potential.
    def getParameterPoint(self):
        return [self.mu2,self.mus2,self.lamda,self.lamda2,self.lamda3,self.v]  



