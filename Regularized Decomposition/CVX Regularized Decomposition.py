import cvxpy as cp
import numpy as np
import re


xi = {"XI 1": (500, 100, -24, -28),
      "XI 2": (300, 300, -28, -32)} # random variable

A = np.array([[1, 1],
              [-1, 0],
              [0, -1]])
b = np.array([120, -40, -20])
c = np.array([100, 150])

n = c.shape[0] # number of first stage variables
m = 2 # number of second stage variables
f = 4 # total number of constraints

p = np.array([0.4, 0.6]) # scenario probabilities
W = np.array([[6, 10],
              [8, 5],
              [1, 0],
              [0, 1]])              
T = np.array([[-60, 0],
              [0, -80],
              [0, 0],
              [0, 0]])


def uncertain_params(k):
    d1, d2, q1, q2 = xi[f"XI {k}"]
    h = np.array([0, 0, d1, d2])
    q = np.array([q1, q2])
    return h, q

    
def optimality_cuts(x_value, k, p=p, W=W):
    h, q = uncertain_params(k)
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


def feasibility_cuts(x_value, k, W=W, T=T):
    I = np.identity(f)
    ones = np.ones((f,))
    h, q = uncertain_params(k)
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
epsilon = 1
main_stop_rule = False
cuts, a = run_first_iterarion()
while not main_stop_rule:
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
        h = uncertain_params(k)[0]
        lamda_x += np.matmul(np.reshape(np.array(Pi[k]), newshape=(1, len(T))), h - np.matmul(T, x_opt))

    if need_feasible_cut:
        D, d = feasibility_cuts(x_value=x_opt, k=k)[:2]
        cuts[f"Cut feasible {v}"] = {"D": D[0], "d": d[0]}   
        continue

    if sum(stop_rule.values()) == len(xi): # w1 < theta1 & w2 < theta2 & ...
        a = x_opt

    lamda_a = 0
    for k in range(1, len(xi)+1):
        h = uncertain_params(k)[0]
        lamda_a += np.matmul(np.reshape(np.array(Pi[k]), newshape=(1, len(T))), h - np.matmul(T, a))
    
    if abs(np.matmul(c.T, x_opt) + lamda_x - np.matmul(c.T, a) - lamda_a) > epsilon:
        a = x_opt
    else:
        main_stop_rule = True
    v += 1


print(f'\nOptimal value of Objective function: {sum(opt_theta):.3f}\n')
for j in range(n):
    print(f'Optimal value of a{j+1}: {a[j]:.3f}\n')
for j in range(len(xi)):
    print(f'Optimal value of y when xi {j+1}: {y_opt[f'XI {j+1}']}\n')
