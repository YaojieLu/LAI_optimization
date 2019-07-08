
"""
We define an Markov Decision Process.
We represent a policy as a dictionary of {state: action} pairs.
"""

import numpy as np

def Ef(dL, gs, L, slope, dt):
    """ Given leaf area and stomatal conductance, return whole-plant transpiration """
    return slope*(L+dL)*gs*dt

def gsmax_sf(dL, L, s, slope, dt):
    """ Given soil moisture and leaf area, return maximum stomatal conductance """
    return s/slope/(L+dL)/dt

def Anf(dL, gs, L, Amax, gs50, mL, cL, dt):
    """ Whole-plant photosynthesis rate """
    A = (L+dL)*Amax*gs/(gs+gs50)*dt
    """ Leaf maintenance cost and leaf construction cost """
    C = mL*(L+dL)*dt+cL*max(0, dL)
    return A - C

def rd_prob(rain_prob, mean_rd, rd, delta_rd):
    return rain_prob*(np.exp(-(1.0/mean_rd)*rd)-np.exp(-(1.0/mean_rd)*(rd+delta_rd)))

class MDP:
    
    """ An MDP is defined by a transition model and a reward function. """
    
    def __init__(self,
                 gamma = 0.99, dt = 1.0,
                 k = 0.1, mean_rd = 0.2,
                 slope = 0.05, Amax = 15, gs50 = 0.034, mL = 1, cL = 10,
                 dL_unit = 0.1,
                 gs_min = 0, gs_max = 1.0, gs_unit = 0.02,
                 L_min = 0, L_max = 10.0,
                 s_min = 0, s_max = 1.0):
        
        if not (0 < gamma <= 1.0):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        
        self.gamma = gamma
        self.dt = dt
        self.k = k
        self.mean_rd = mean_rd
        self.slope = slope
        self.Amax = Amax
        self.gs50 = gs50
        self.mL = mL
        self.cL = cL
        self.dL_unit = dL_unit
        self.gs_min = gs_min
        self.gs_max = gs_max
        self.gs_unit = gs_unit
        self.L_min = L_min
        self.L_max = L_max
        self.L_unit = dL_unit
        self.L_space = np.round(np.arange(L_min, L_max+self.L_unit, self.L_unit), 6)
        self.s_min = s_min
        self.s_max = s_max
        self.s_unit = np.round(Ef(self.L_unit, gs_unit, 0, slope, dt), 6)
        self.s_space = np.round(np.arange(s_min, s_max+self.s_unit, self.s_unit), 6)
    
    def R(self, dL, gs, L):
        """ Return the current reward for the state and action. """
        return Anf(dL, gs, L, self.Amax, self.gs50, self.mL, self.cL, self.dt)
        
    def T(self, dL, gs, L, s):
        """
        Transition model. From a state and an action,
        return a list of (probability, result-state) pairs.
        """
        E = Ef(dL, gs, L, self.slope, self.dt)
        sE = s - E
        rd_sE_len = int(round((self.s_max - sE) / self.s_unit + 1))
        rd_sE = np.linspace(0, 1-sE, rd_sE_len)
        if rd_sE_len >= 2:
            p_sE1 = 1.0 - self.k*self.dt
            p_sE2 = rd_prob(self.k*self.dt, self.mean_rd, 0, self.s_unit)
            p_sE = p_sE1 + p_sE2
            p = [(p_sE, np.round(sE, 6))]
            p_1 = self.k*self.dt*(np.exp(-(1.0/self.mean_rd)*(1.0-sE)))
            if rd_sE_len >= 3:
                for r in rd_sE[1:-1]:
                    p_si = rd_prob(self.k*self.dt, self.mean_rd, r, self.s_unit)
                    p.append((p_si, np.round(r+sE, 6)))
                p.append((p_1, 1.0))
            else:
                p.append((p_1, 1.0))
        else:
            p = [(1.0, 1.0)]
        return p
    
    def dL_space(self, L):
        """ Return a list of dL values that can be performed in this state. """
        return np.arange(self.L_min - L, self.L_max - L + self.L_unit, self.L_unit)
    
    def gs_space(self, dL, L, s):
        """ Return a list of gs values that can be performed in this state. """
        if L+dL == 0:
            return [0]
        else:
            gs_full_space_len = int((self.gs_max - self.gs_min) / self.gs_unit + 1)
            gs_full_space = np.linspace(self.gs_min, self.gs_max, gs_full_space_len)
            gsmax_s = gsmax_sf(dL, L, s, self.slope, self.dt)
            if gsmax_s > self.gs_max:
                return gs_full_space
            else:
                boundary = int(gsmax_s / self.gs_unit) + 1
                return gs_full_space[:boundary]
