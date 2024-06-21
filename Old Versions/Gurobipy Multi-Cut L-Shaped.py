from gurobipy import *
import numpy as np


xi = {"XI 1": (500, 100, -24, -28),
      "XI 2": (300, 300, -28, -32)}

A = np.array([[1, 1],
              [-1, 0],
              [0, -1]])
b = np.array([120, -40, -20])
c = np.array([100, 150])

n = c.shape[0] # number of first stage variables
m = 2 # number of second stage variables

p = np.array([0.4, 0.6])
W = np.array([[6, 10],
              [8, 5],
              [1, 0],
              [0, 1]])              
T = np.array([[-60, 0],
              [0, -80],
              [0, 0],
              [0, 0]])


model = Model("LP")
model.setParam('OutputFlag', 0)
x = model.addMVar(n, vtype=GRB.CONTINUOUS, name="x")
model.setObjective(c @ x, GRB.MINIMIZE)
model.addConstr(A @ x <= b, name="constraints")
model.optimize()

x_opt = np.array([x.x[i] for i in range(n)])


def optimality_cuts(x_value, k, p=p, W=W):
    d1, d2, q1, q2 = xi[f"XI {k}"]
    h = np.array([0, 0, d1, d2])

    q = np.array([q1, q2])
    Tx = np.matmul(T, x_value)

    model = Model("Optimality Model")
    model.setParam('OutputFlag', 0)
    y = model.addMVar(m, vtype=GRB.CONTINUOUS, name="y")
    model.setObjective(q @ y, GRB.MINIMIZE)
    constraints = model.addConstr(W @ y <= h - Tx, name="constraints")
    model.optimize()

    E = np.matmul(np.reshape(p[k-1] * np.array(constraints.pi), newshape=(1, len(T))), T)
    e = np.matmul(np.reshape(p[k-1] * np.array(constraints.pi), newshape=(1, len(T))), h)
    opt_y = [v.x for v in model.getVars()]
    return E, e, opt_y


E1, e1 = optimality_cuts(x_value=x_opt, k=1)[:2] 
E2, e2 = optimality_cuts(x_value=x_opt, k=2)[:2] 

cuts =  {"Cut 0": [
    {"E1": E1, "e1": e1},
    {"E2": E2, "e2": e2}
                  ]
        }   



v = 0
y_opt = {}
while v < 20:
    master_problem = Model("Master Problem")
    master_problem.setParam('OutputFlag', 0)
    x = master_problem.addMVar(n, vtype=GRB.CONTINUOUS, name="x")
    theta = master_problem.addMVar(len(xi), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
    ones = np.ones((len(xi),))
    master_problem.setObjective(c @ x + ones @ theta, GRB.MINIMIZE)
    master_problem.addConstr(A @ x <= b, "main constraints")

    for cut in cuts.keys():
        for k in range(1, len(xi)+1):
            master_problem.addConstr(cuts[cut][k-1][f"E{k}"] @ x + theta[k-1] >= cuts[cut][k-1][f"e{k}"])

    master_problem.optimize()

    x_opt = [v.x for v in master_problem.getVars()[:n]]
    opt_theta = [v.x for v in master_problem.getVars()[n:]]

    E1, e1, opt_y1 = optimality_cuts(x_value=x_opt, k=1) 
    E2, e2, opt_y2 = optimality_cuts(x_value=x_opt, k=2) 
    y_opt["XI 1"] = opt_y1
    y_opt["XI 2"] = opt_y2

    if (opt_theta[0] >= e1[0] - np.sum(E1[0] * np.array(x_opt))) or \
       (opt_theta[1] >= e2[0] - np.sum(E2[0] * np.array(x_opt))):
        break
    else:
        cuts[f"Cut {v+1}"] = [{"E1": E1, "e1": e1}, {"E2": E2, "e2": e2}]
    v += 1


optimal_obj_value =  np.sum(p * np.array([np.sum(np.array(xi[f"XI {k}"][2:]) * np.array(y_opt[f"XI {k}"])) for k in range(1, len(xi)+1)]))

print(f'\nOptimal Objective Value: {optimal_obj_value:.3f}\n')
for j in range(n):
    print(f'Optimal value of x{j+1}: {x_opt[j]:.3f}\n')
for j in range(m):
    print(f'Optimal value of theta{j+1}: {opt_theta[j]:.3f}\n')
for i in y_opt.keys():
    print(f'Optimal values for y in {i}: {y_opt[i]}\n')
