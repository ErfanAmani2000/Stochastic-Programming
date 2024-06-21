import cvxpy as cp
import numpy as np


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
        E = -np.matmul(np.reshape(p[k-1] * np.array(constraints[0].dual_value), newshape=(1, len(T))), T)
        e = -np.matmul(np.reshape(p[k-1] * np.array(constraints[0].dual_value), newshape=(1, len(T))), h)
        return E, e, y.value


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
    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(c.T @ x)
    constraints = [A @ x <= b]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status == cp.INFEASIBLE:
        raise ValueError("First stage problem is infeasible.") 
    else:
        return x.value


def run_first_iterarion(c=c, A=A, b=b, xi=xi):
    x_opt = first_stage_problem(c=c, A=A, b=b, xi=xi)

    E = np.zeros((len(xi),))
    e = 0
    for k in range(1, len(xi)+1):
        try: 
            add_E, add_e = optimality_cuts(x_value=x_opt, k=k)[:2]
            E += add_E[0]
            e += add_e[0]
        except ValueError:
            D, d = feasibility_cuts(x_value=x_opt, k=k)[:2]
            cuts =  {f"Cut feasible{k}": 
                {"D": D[0], "d": d[0]}
                    }   
            break
    cuts =  {"Cut 0": 
        {"E": E, "e": e}
            }  
    return cuts



v = 1
w = 0
y_opt = {}
opt_theta = -np.Inf
cuts = run_first_iterarion()
while w > opt_theta: # w = e - E*x > theta
    ones = np.ones((1,))
    x = cp.Variable(n, nonneg=True)
    theta = cp.Variable()
    objective = cp.Minimize(c.T @ x + theta)
    constraints = [A @ x <= b]

    for cut_name, cut in cuts.items():
        if 'feasible' not in cut_name:
            E, e = cut["E"], cut["e"] 
            constraints.append(E @ x + theta >= e)
        else:
            D, d = cut["D"], cut["d"] 
            constraints.append(D @ x >= d)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    x_opt = x.value
    opt_theta = theta.value

    E = np.zeros((len(xi),))
    e = 0
    need_feasible_cut = False
    for k in range(1, len(xi)+1):
        try: 
            add_E, add_e, y = optimality_cuts(x_value=x_opt, k=k)
            y_opt[f"XI {k}"] = y
            E += add_E[0]
            e += add_e[0]
        except ValueError:
            need_feasible_cut = True
            break

    if need_feasible_cut:
        D, d = feasibility_cuts(x_value=x_opt, k=k)[:2]
        cuts[f"Cut feasible {v}"] = {"D": D[0], "d": d[0]}   
        continue
    else:
        cuts[f"Cut {v}"] = {"E": E, "e": e}

    w = e - np.sum(E * np.array(x_opt))
    v += 1


print(f'\nOptimal value of theta: {opt_theta:.3f}\n')
for j in range(n):
    print(f'Optimal value of x{j+1}: {x_opt[j]:.3f}\n')
