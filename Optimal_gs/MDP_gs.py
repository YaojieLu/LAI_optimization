"""
Markov Decision Process

First we define an MDP.
We also represent a policy as a dictionary of {state: action} pairs.
We then define the value_iteration and policy_iteration algorithms.
"""

from utils import argmax
import random
import numpy as np

def Ef(gs, slope, dt):
    """ Given stomatal conductance, return transpiration rate """
    E = slope*gs*dt
    return E

def gsmax_sf(s, slope, dt):
    """ Given soil moisture, return maximum stomatal conductance """
    gsmax_s = s/slope/dt
    return gsmax_s

def Af(gs, Amax, gs50):
    """ Photosynthesis rate """
    A = Amax*gs/(gs + gs50)
    return A

def rd_prob(rain_prob, mean_rd, rd, delta_rd):
    prob = rain_prob*(np.exp(-(1.0/mean_rd)*rd)-np.exp(-(1.0/mean_rd)*(rd+delta_rd)))
    return prob

class MDP:
    
    """An Markov Decision Process is defined by an initial state, transition model,
    and reward function. The transition model T(s, a) return a
    list of (p, s') pairs."""
    
    def __init__(self, gamma = 0.99, dt = 1.0, # dt: step size
                 k = 0.1, mean_rd = 0.2,
                 slope = 0.05, Amax = 15, gs50 = 0.034,
                 action_min = 0, action_max = 1.0, action_unit = 0.01,
                 state_min = 0, state_max = 1.0):
        if not (0 < gamma <= 1.0):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        self.gamma = gamma
        self.dt = dt
        self.k = k
        self.mean_rd = mean_rd
        self.slope = slope
        self.Amax = Amax
        self.gs50 = gs50
        self.action_min = action_min
        self.action_max = action_max
        self.action_unit = action_unit
        self.state_min = state_min
        self.state_max = state_max
        self.state_unit = np.round(Ef(gs = action_unit, slope = slope, dt = dt), 6)
        self.states = np.round(np.arange(state_min, state_max+self.state_unit, self.state_unit), 6)
    
    def R(self, action):
        """Return a numeric reward for an action."""
        reward = Af(action, self.Amax, self.gs50)
        return reward
        
    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""
        s, gs = state, action
        E = Ef(gs = gs, slope = self.slope, dt = self.dt)
        sE = s - E
        l = int(np.round((self.state_max - sE) / self.state_unit + 1))
        rd_sE = np.linspace(0, 1-sE, l)
        if l >= 2:
            p_sE1 = 1.0 - self.k*self.dt
            p_sE2 = rd_prob(self.k*self.dt, self.mean_rd, 0, self.state_unit)
            p_sE = p_sE1 + p_sE2
            p = [(p_sE, np.round(sE, 6))]
            p_1 = self.k*self.dt*(np.exp(-(1.0/self.mean_rd)*(1.0-sE)))
            if l >= 3:
                for r in rd_sE[1:-1]:
                    p_si = rd_prob(self.k*self.dt, self.mean_rd, r, self.state_unit)
                    p.append((p_si, np.round(r+sE, 6)))
                p.append((p_1, 1.0))
            else:
                p.append((p_1, 1.0))
        else:
            p = [(1.0, 1.0)]
        return p
    
    def actions(self, state):
        """Return a list of actions that can be performed in this state."""
        s = state
        gsmax_s = gsmax_sf(s = s, slope = self.slope, dt = self.dt)
        l = int((self.action_max - self.action_min) / self.action_unit + 1)
        action_full = np.linspace(self.action_min, self.action_max, l)
        
        if gsmax_s > self.action_max:
            return action_full
        else:
            boundary = int(gsmax_s/self.action_unit) + 1
            return action_full[:boundary]

# _______________________________________________________________________________

def fast_value_iteration(mdp, gap, utility = None, epsilon = 1):
    """Solving an MDP by value iteration"""
    
    states_reduced = mdp.states[0::gap]
    U = utility or {s: 0.0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    i = 0    
    while True:
        U_reduced = []
        for s in states_reduced:
            U_s = max((R(a) + gamma * sum(p*U[s1] for (p, s1) in T(s, a)))
                                                     for a in mdp.actions(s))
            U_reduced.append(U_s)
        
        U1_values = np.interp(mdp.states, states_reduced, U_reduced)
        U1 = dict(zip(mdp.states, U1_values))
        U_values = np.fromiter(U.values(), dtype = float)
        delta = max(abs(U_values - U1_values))
        print(delta)
        i += 1
        U = U1
        if delta <= epsilon*(1.0 - gamma)/gamma:
            return U

def value_iteration(mdp, utility = None, epsilon = 1):
    """Solving an MDP by value iteration"""
    
    U1 = utility or {s: 0.0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = max((R(a) + gamma * sum(p*U[s1] for (p, s1) in T(s, a)))
                                                     for a in mdp.actions(s))
            delta = max(delta, abs(U1[s] - U[s]))
        print(delta)
        if delta <= epsilon*(1.0 - gamma)/gamma:
            return U

# _______________________________________________________________________________

def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""
    
    return mdp.R(a) + mdp.gamma * sum(p*U[s1] for (p, s1) in mdp.T(s, a))

def policy_evaluation(pi, U, mdp, k = 20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(pi[s]) + gamma * sum(p*U[s1] for (p, s1) in T(s, pi[s]))
    return U

def policy_iteration(mdp):
    """Solve an MDP by policy iteration"""
    
    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    f = lambda a: expected_utility(a, s, U, mdp)
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), key = f)
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action."""
    
    pi = {}
    f = lambda a : expected_utility(a, s, U, mdp)
    for count, s in enumerate(mdp.states):
        if count % 1 == 0:
    #for s in mdp.states:
            pi[s] = argmax(mdp.actions(s), key = f)
    return pi
