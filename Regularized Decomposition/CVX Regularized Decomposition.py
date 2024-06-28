import cvxpy as cp
import numpy as np
import re

# Generate the scenario_T data
number_of_scenario = 3
scenario_T = np.column_stack((
    np.linspace(2, 3, number_of_scenario), 
    np.linspace(2.4, 3.6, number_of_scenario), 
    np.linspace(16, 24, number_of_scenario)
)).tolist() 

# Create Random Variable's dictionary
xi = {}
for idx, s in enumerate(scenario_T):
    xi[f"XI {idx+1}"] = tuple(s)

A = np.array([[1, 1, 1]])
b = np.array([500])
c = np.array([150, 230, 260])

n = c.shape[0] # number of first stage variables
m = 6 # number of second stage variables
f = 4 # total number of constraints

"""
y1k = y1, w1k = y2, y2k = y3, w2k = y4, w3k = y5, w4k = y6
"""

p = np.array([1/len(scenario_T) for i in range(len(scenario_T))]) # scenario probabilities

W = np.array([[-1, 1, 0, 0, 0, 0],
              [0, 0, -1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1, 0]])  


def uncertain_params(k):
    a1, a2, a3 = xi[f"XI {k}"]
    h = np.array([-200, -240, 0, 6000])
    q = np.array([238, -170, 210, -150, -36, -10])
    T = np.array([[-1*a1, 0, 0],
                  [0, -1*a2, 0],
                  [0, 0, -1*a3],
                  [0, 0, 0]])
    return h, q, T

    
def optimality_cuts(x_value, k, p=p, W=W):
    h, q, T = uncertain_params(k)
    Tx = np.matmul(T, x_value)

    y = cp.Variable(m, nonneg=True)
    objective = cp.Minimize(q.T @ y)
    constraints = [W @ y <= h - Tx]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status == cp.INFEASIBLE:
        raise ValueError("Subproblem is infeasible, so a feasible cut must be added.") 
    else:
        pi = constraints[0].dual_value
        E = -np.matmul(np.reshape(p[k-1] * np.array(pi), newshape=(1, len(T))), T)
        e = -np.matmul(np.reshape(p[k-1] * np.array(pi), newshape=(1, len(T))), h)
        return E, e, y.value, pi


def feasibility_cuts(x_value, k, W=W):
    I = np.identity(f)
    ones = np.ones((f,))
    h, q, T = uncertain_params(k)
    Tx = np.matmul(T, x_value)

    y = cp.Variable(m, nonneg=True)
    v_pos = cp.Variable(f, nonneg=True)
    v_neg = cp.Variable(f, nonneg=True)

    objective = cp.Minimize(cp.sum(v_pos) + cp.sum(v_neg))
    constraints = [W @ y + I @ v_pos - I @ v_neg == h - Tx]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    D = -np.matmul(np.reshape(np.array(constraints[0].dual_value), newshape=(1, len(T))), T)
    d = -np.matmul(np.reshape(np.array(constraints[0].dual_value), newshape=(1, len(T))), h)
    return D, d, problem.value


def first_stage_problem(c=c, A=A, b=b, xi=xi):
    init_a = np.random.random(size=(n,))
    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(c.T @ x + 0.5*cp.norm(x - init_a, 2)**2)
    constraints = [A @ x <= b]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status == cp.INFEASIBLE:
        raise ValueError("First stage problem is infeasible.") 
    else:
        return x.value


def run_first_iterarion(c=c, A=A, b=b, xi=xi):
    x_opt = first_stage_problem(c=c, A=A, b=b, xi=xi)

    cuts = {}
    initial_theta = -np.Inf
    for k in range(1, len(xi)+1):
        try: 
            E, e = optimality_cuts(x_value=x_opt, k=k)[:2]
            w = e[0] - np.sum(E[0] * np.array(x_opt))
            if w > initial_theta:
                cuts[f'Cut 0-{k}'] = {"E": E[0], "e": e[0]}
        except ValueError:
            D, d = feasibility_cuts(x_value=x_opt, k=k)[:2]
            cuts[f'Cut feasible{k}'] = {"D": D[0], "d": d[0]}  
            break
    return cuts, x_opt
    


v = 0
y_opt = {}
epsilon = 0.00001
max_iterations = 25
consecutive_below_threshold = 0
cuts, a = run_first_iterarion()
while v < max_iterations:
    ones = np.ones((1,))
    x = cp.Variable(n, nonneg=True)
    theta = cp.Variable(len(xi))
    objective = cp.Minimize(c.T @ x + cp.sum(theta) + 0.5*cp.norm(x - a, 2)**2)
    constraints = [A @ x <= b]

    for cut_name, cut in cuts.items():
        if 'feasible' not in cut_name:
            E, e = cut["E"], cut["e"]
            k = int(re.findall(r'\d+', cut_name)[-1])
            constraints.append(E @ x + theta[k-1] >= e)
        else:
            D, d = cut["D"], cut["d"] 
            constraints.append(D @ x >= d)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    x_opt = x.value
    opt_theta = theta.value

    Pi = {}
    lamda_x = 0
    need_feasible_cut = False
    stop_rule = {1: False, 2: False}
    for k in range(1, len(xi)+1):
        try: 
            E, e, y, pi = optimality_cuts(x_value=x_opt, k=k)
            y_opt[f"XI {k}"] = y
            Pi[k] = pi
            w = e[0] - np.sum(E[0] * np.array(x_opt))
            if w > opt_theta[k-1]:
                cuts[f'Cut {v+1}-{k}'] = {"E": E[0], "e": e[0]}
            else:
                stop_rule[k] = True    
        except ValueError:
            need_feasible_cut = True
            break
        h, q, T = uncertain_params(k)
        lamda_x += np.matmul(np.reshape(np.array(Pi[k]), newshape=(1, len(T))), h - np.matmul(T, x_opt))

    if need_feasible_cut:
        D, d = feasibility_cuts(x_value=x_opt, k=k)[:2]
        cuts[f"Cut feasible {v}"] = {"D": D[0], "d": d[0]}   
        continue

    if sum(stop_rule.values()) == len(xi): # w1 < theta1 & w2 < theta2 & ...
        a = x_opt

    lamda_a = 0
    for k in range(1, len(xi)+1):
        h, q, T = uncertain_params(k)
        lamda_a += np.matmul(np.reshape(np.array(Pi[k]), newshape=(1, len(T))), h - np.matmul(T, a))
    
    if abs(np.matmul(c.T, x_opt) + lamda_x - np.matmul(c.T, a) - lamda_a) > epsilon:
        a = x_opt
        consecutive_below_threshold = 0
    else:
        consecutive_below_threshold += 1
        if consecutive_below_threshold >= 3:
            print("Measure has been below the threshold for 3 consecutive iterations. Stopping.")
            break
    v += 1


for j in range(len(xi)):
    y_opt[f'XI {j+1}'] = [round(num, 2) for num in y_opt[f'XI {j+1}']]


print(f'\nOptimal value of Objective function: {sum(opt_theta):.1f}\n')
for j in range(n):
    print(f'Optimal value of x{j+1}: {a[j]:.1f}\n')
for j in range(len(xi)):
    print(f'Optimal value of y when xi {j+1}: {y_opt[f'XI {j+1}']}\n')
