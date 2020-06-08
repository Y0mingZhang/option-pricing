import numpy as np
import math

class EuropeanCall:
    """ European Call Option on a one-step binomial model. """
    def __init__(self, S_0, S_up, S_down, A_0, A_1, K):
        """
        args:
        S_0, S_up, S_down are stock prices
        A_0, A_1 are bond prices
        K : strike price
        """
        self.S_0, self.S_up, self.S_down, self.A_0, self.A_1, self.K \
        =  S_0, S_up, S_down, A_0, A_1, K

    def get_replicating_portfolio(self):
        """
        returns: [# of stock, # of bond]
        """
        
        """ Construct linear system and solve using linear algebra"""
        a = np.array([[self.S_down, self.A_1],
                           [self.S_up, self.A_1]])
        b = np.array([max(self.S_down - self.K, 0), max(self.S_up - self.K, 0)])
        
        return np.linalg.solve(a, b)
    
    def get_option_price(self):
        """
        returns: price of option
        """
        x, y = self.get_replicating_portfolio()
        return x * self.S_0 + y * self.A_0

class EuropeanPut:
    """ European Put Option on a one-step binomial model. """
    def __init__(self, S_0, S_up, S_down, A_0, A_1, K):
        """
        args:
        S_0, S_up, S_down are stock prices
        A_0, A_1 are bond prices
        K : strike price
        """
        self.S_0, self.S_up, self.S_down, self.A_0, self.A_1, self.K \
        =  S_0, S_up, S_down, A_0, A_1, K

    def get_replicating_portfolio(self):
        """
        returns: [# of stock, # of bond]
        """
        
        """ Construct linear system and solve using linear algebra"""
        a = np.array([[self.S_down, self.A_1],
                           [self.S_up, self.A_1]])
        b = np.array([max(self.K - self.S_down, 0), max(self.K - self.S_up, 0)])
        
        return np.linalg.solve(a, b)
    
    def get_option_price(self):
        """
        returns: price of option
        """
        x, y = self.get_replicating_portfolio()
        return x * self.S_0 + y * self.A_0


def return_and_risk(V_0, V_1):
    """
    args:
    V_0: time of portfolio at time 0
    V_1: A map from all possible values of V_1 to probabilty associated with that value
    (e.g {110 : 0.7, 90 : 0.3})
    """
    
    # Verify that the total probability sums to 1
    assert(sum([V_1[key] for key in V_1]) == 1)
    
    # Generate all possibe returns
    all_returns = {(outcome - V_0) / V_0 : prob for outcome, prob in V_1.items()}
    
    # Compute expecteed return
    expected_return = sum([possible_return * prob for 
                           possible_return, prob in all_returns.items()])
    
    # Compute risk
    variance = sum([(possible_return-expected_return) ** 2 * prob for 
                   possible_return, prob in all_returns.items()])
    return expected_return, variance ** 0.5


class InterestModel:
    """ An interest model """
    def __init__(self, annual_interest, compound_period):
        """
        args:
        annual_interest: nominal annual interest rate
        compound_period: The period of compounding (e.g. 0.5 for 6 months,
         and 0 for coutinuous compounding)
        """
        self.annual_interest, self.compound_period = \
            annual_interest, compound_period
    
    def get_return_rate(self, t):
        if self.compound_period == 0:
            # Continuous
            return math.exp(self.annual_interest * t)
        else:
            # Discrete?
            times_compounded = t // self.compound_period
            return math.pow(1 + self.annual_interest * self.compound_period, times_compounded)
    
    def get_new_value(self, V, t):
        """
        args:
        V: savings at t = 0
        t: new time t
        returns: new value of savings
        """
        return V * self.get_return_rate(t)

from math import exp
EuropeanPut(50, 40, 60, exp(-1.1), 1, 50)