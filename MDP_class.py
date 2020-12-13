
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

# Farquhar model
def Af(gs, ca,
       T=25, I=430,
       Kc=460, q=0.3, R=8.314, Jmax=48, Vcmax=31, z1=0.9, z2=0.9999, tau=43):
    Km = Kc+tau/0.105
    # Rubisco limitation
    Ac = 1/2*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tau)*gs+((ca+Km)*gs)**2)**(1/2))
    J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
    # RuBP limitation
    Aj = 1/2*(J+(2*tau+ca)*gs-(J**2+2*J*(2*tau-ca+2*tau)*gs+((ca+2*tau)*gs)**2)**(1/2))
    # Am = min(Ac, Aj)
    Am = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
    return Am

def Anf(dL, gs, ca, L, mL, cL, dt):
    """ Whole-plant photosynthesis rate """
    # self-shading
    L_effective = 10*(L+dL)/(L+dL+10)
    A = L_effective*Af(gs, ca)*dt
    """ Leaf maintenance cost and leaf construction cost """
    C = mL*(L+dL)*dt+cL*max(0, dL)
    return A-C

def rd_prob(rain_prob, mean_rd, rd, delta_rd):
    return rain_prob*(np.exp(-(1.0/mean_rd)*rd)-np.exp(-(1.0/mean_rd)*(rd+delta_rd)))

class MDP:
    
    """ An MDP is defined by a transition model and a reward function. """
    
    def __init__(self,
                 gamma=0.99, dt=1.0,
                 k=0.1, mean_rd=0.2,
                 slope=0.05, ca=400, mL=1, cL=10,
                 dL_unit=0.1,
                 gs_min=0, gs_max=1.0, gs_unit=0.02,
                 L_min=0, L_max=10.0,
                 s_min=0, s_max=1.0):
        
        if not (0 < gamma <= 1.0):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        
        self.gamma = gamma
        self.dt = dt
        self.k = k
        self.mean_rd = mean_rd
        self.slope = slope
        self.ca = ca
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
        return Anf(dL, gs, self.ca, L, self.mL, self.cL, self.dt)
        
    def T(self, dL, gs, L, s):
        """
        Transition model. From a state and an action,
        return a list of (probability, result-state) pairs.
        """
        E = Ef(dL, gs, L, self.slope, self.dt)
        sE = s-E
        rd_sE_len = int(round((self.s_max-sE) / self.s_unit+1))
        rd_sE = np.linspace(0, 1-sE, rd_sE_len)
        if rd_sE_len >= 2:
            p_sE1 = 1.0-self.k*self.dt
            p_sE2 = rd_prob(self.k*self.dt, self.mean_rd, 0, self.s_unit)
            p_sE = p_sE1+p_sE2
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
        return np.arange(self.L_min-L, self.L_max-L+self.L_unit, self.L_unit)
    
    def gs_space(self, dL, L, s):
        """ Return a list of gs values that can be performed in this state. """
        if L+dL == 0:
            return [0]
        else:
            gs_full_space_len = int((self.gs_max-self.gs_min) / self.gs_unit+1)
            gs_full_space = np.linspace(self.gs_min, self.gs_max, gs_full_space_len)
            gsmax_s = gsmax_sf(dL, L, s, self.slope, self.dt)
            if gsmax_s > self.gs_max:
                return gs_full_space
            else:
                boundary = int(gsmax_s / self.gs_unit)+1
                return gs_full_space[:boundary]
