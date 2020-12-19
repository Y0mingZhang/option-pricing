import math
import numpy as np

class Node:
    def __init__(self, ex_div, cum_div, T, up=None, down=None):
        self.ex_div = ex_div
        self.cum_div = cum_div
        self.T = T
        self.up = up
        self.down = down

def get_stock_price_tree(u, d, S0, T, div, div_time):

    def complete_tree(t, cum_div):
        if t > T:
            return None
        ex_div = cum_div
        if t in div_time:
            ex_div -= div
        return Node(ex_div, cum_div, t, 
            up=complete_tree(t+1, ex_div * (1+u)),
            down=complete_tree(t+1, ex_div * (1+d)))
    
    return complete_tree(0, S0)

def HedgeEuropeanOptionBinomial(S, X, r, BF, T, mode):
    """
    S: stock price tree
    r: risk free rate
    X: strike price
    T: number of steps
    BF: Bond Face Value at maturity
    mode: either "call" or "put"
    """
    assert(mode == "call" or mode == "put")
    def payoff(node):
        if mode == "call":
            return max(0, node.ex_div - X)
        else:
            return max(0, X - node.ex_div)

    def bond_price(t):
        return math.pow(1+r, t - T) * BF
    def hedge_node(node):
        if node.up and node.down:
            # Replicate use cum_div prices

            curr_bond_value = bond_price(node.T)
            next_bond_value = bond_price(node.T+1)
            a = np.array([[node.up.cum_div, next_bond_value],[node.down.cum_div, next_bond_value]])
            b = np.array([hedge_node(node.up), hedge_node(node.down)])
            node.position_in_shares, node.position_in_bonds = np.linalg.solve(a, b)
            node.price = node.position_in_shares * node.ex_div + node.position_in_bonds * curr_bond_value

        else:
            node.price = payoff(node)
        return node.price
    
    hedge_node(S)
    return S

def HedgeAmericanOptionBinomial(S, X, r, BF, T, mode):
    """
    S: stock price tree
    r: risk free rate
    X: strike price
    T: number of steps
    BF: Bond Face Value at maturity
    mode: either "call" or "put"
    """
    assert(mode == "call" or mode == "put")
    def payoff(node):
        if mode == "call":
            return max(0, node.cum_div - X)
        else:
            return max(0, X - node.cum_div)

    def bond_price(t):
        return math.pow(1+r, t - T) * BF
    def hedge_node(node):
        if node.up and node.down:
            # Replicate use cum_div prices

            curr_bond_value = bond_price(node.T)
            next_bond_value = bond_price(node.T+1)
            a = np.array([[node.up.cum_div, next_bond_value],[node.down.cum_div, next_bond_value]])
            b = np.array([hedge_node(node.up), hedge_node(node.down)])
            node.position_in_shares, node.position_in_bonds = np.linalg.solve(a, b)
            node.required_capital = node.position_in_shares * node.ex_div + node.position_in_bonds * curr_bond_value
            node.payoff = payoff(node)
            node.early_exercise = bool(node.payoff >= node.required_capital)
            node.price = max(node.required_capital, node.payoff)
        else:
            node.price = payoff(node)
            node.early_exercise = False
        return node.price
    
    hedge_node(S)
    return S

def print_stock_prices(root):
    q = [root]
    node_format = "({}, {}) | {:.2f} | {:.2f}"
    print("Node | Ex div | Cum div")
    while q:
        skip_node = set()
        next_q = []
        for node in q:
            node_repr = (round(node.ex_div, 5), round(node.cum_div, 5))
            if node_repr not in skip_node:
                print(node_format.format(node.T, len(skip_node), node.ex_div, node.cum_div))
                skip_node.add(node_repr)
                if node.up and node.down:
                    next_q.append(node.up)
                    next_q.append(node.down)
        q = next_q

def print_hedge_positions(root):
    q = [root]
    node_format = "({}, {}) | {:.2f} | {:.2f} | {:.2f}"
    leaf_format = "({}, {}) | - | - | {:.2f}"
    print("Node | Position in shares | Position in bonds | Price of the option")
    while q:
        skip_node = set()
        next_q = []
        for node in q:
            node_repr = (round(node.ex_div, 5), round(node.cum_div, 5))
            if node_repr not in skip_node:
                if node.up and node.down:
                    next_q.append(node.up)
                    next_q.append(node.down)
                    print(node_format.format(node.T, len(skip_node), 
                        node.position_in_shares, node.position_in_bonds, node.price))
                
                else:
                    print(leaf_format.format(node.T, len(skip_node), 
                        node.price))
                skip_node.add(node_repr)
        q = next_q

def print_european_option(root):
    q = [root]
    node_format = "({}, {}) | {:.2f}"
    print("Node | Price of the option")
    while q:
        skip_node = set()
        next_q = []
        for node in q:
            node_repr = (round(node.ex_div, 5), round(node.cum_div, 5))
            if node_repr not in skip_node:
                if node.up and node.down:
                    next_q.append(node.up)
                    next_q.append(node.down)
                    print(node_format.format(node.T, len(skip_node), 
                        node.price))
                
                else:
                    print(node_format.format(node.T, len(skip_node), 
                        node.price))
                skip_node.add(node_repr)
        q = next_q

def print_american_option(root):
    q = [root]
    node_format = "({}, {}) | {:.2f} | {}"
    print("Node | Price of the option | Early Exercise")
    while q:
        skip_node = set()
        next_q = []
        for node in q:
            node_repr = (round(node.ex_div, 5), round(node.cum_div, 5))
            if node_repr not in skip_node:
                if node.up and node.down:
                    next_q.append(node.up)
                    next_q.append(node.down)
                    print(node_format.format(node.T, len(skip_node), 
                        node.price, node.early_exercise))
                
                else:
                    print(node_format.format(node.T, len(skip_node), 
                        node.price, node.early_exercise))
                skip_node.add(node_repr)
        q = next_q

def normal_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def d_plus(S, X, r, sigma, T, t):
    return (math.log(S / X) + (r + sigma ** 2 / 2) * (T - t)) / (sigma * math.sqrt(T - t))

def d_minus(S, X, r, sigma, T, t):
    return (math.log(S / X) + (r - sigma ** 2 / 2) * (T - t)) / (sigma * math.sqrt(T - t))

def EuropeanCallBlackScholes(S, X, r, sigma, T, t):
    return S * normal_cdf(d_plus(S, X, r, sigma, T, t)) - X * math.exp(-r * (T-t)) * normal_cdf(d_minus(S, X, r, sigma, T, t))

def EuropeanPutBlackScholes(S, X, r, sigma, T, t):
    return -S * normal_cdf(-d_plus(S, X, r, sigma, T, t)) + X * math.exp(-r * (T-t)) * normal_cdf(-d_minus(S, X, r, sigma, T, t))

def Delta(S, X, r, sigma, T, t, type):
    D_call = normal_cdf(d_plus(S, X, r, sigma, T, t))
    if type == 'call':
        return D_call
    assert(type == 'put')
    return D_call - 1

def Gamma(S, X, r, sigma, T, t):
    res = 1/(S * sigma * math.sqrt(2 * math.pi * T))
    res *= math.exp(-d_plus(S, X, r, sigma, T, t)**2 / 2)
    return res

def Vega(S, X, r, sigma, T, t):
    res = S * math.sqrt(T) / math.sqrt(2 * math.pi)
    res *= math.exp(-0.5 * d_plus(S, X, r, sigma, T, t)**2)
    return res

def Theta(S, X, r, sigma, T, t, type):
    assert(type == 'put' or type == 'call')
    d_p = d_plus(S, X, r, sigma, T, t)
    d_m = d_minus(S, X, r, sigma, T, t)
    call = -S * sigma / (2 * math.sqrt(2 * math.pi * T))
    call *= math.exp(-0.5 * d_p ** 2)
    call -= r * X * math.exp(-r * T) * normal_cdf(d_m)

    if type == 'call':
        return call
    return call - r * X * math.exp(-r * T)

def Rho(S, X, r, sigma, T, t, type):
    assert(type == 'put' or type == 'call')
    d_m = d_minus(S, X, r, sigma, T, t)
    call = T * X * math.exp(-r * T) * normal_cdf(d_m)

    if type == 'call':
        return call
    return call - T * X * math.exp(-r * T)

