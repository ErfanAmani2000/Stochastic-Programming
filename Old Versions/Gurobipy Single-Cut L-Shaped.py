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


E = np.zeros((len(xi),))
e = 0
for k in range(1, len(xi)+1):
    E += optimality_cuts(x_value=x_opt, k=k)[0][0]
    e += optimality_cuts(x_value=x_opt, k=k)[1][0]

cuts =  {"Cut 0": 
    {"E": E, "e": e}
        }   


v = 0
w = 0
y_opt = {}
opt_theta = -np.Inf
while w > opt_theta:
    master_problem = Model("Master Problem")
    master_problem.setParam('OutputFlag', 0)
    x = master_problem.addMVar(n, vtype=GRB.CONTINUOUS, name="x")
    theta = master_problem.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
    
    master_problem.setObjective(c @ x + theta, GRB.MINIMIZE)
    master_problem.addConstr(A @ x <= b, "main constraints")

    for cut in cuts.keys():
        master_problem.addConstr(cuts[cut]["E"] @ x + theta >= cuts[cut]["e"])
    
    master_problem.optimize()

    x_opt = [v.x for v in master_problem.getVars()[:n]]
    opt_theta = [v.x for v in master_problem.getVars()[n:]][0]

    E = np.zeros((len(xi),))
    e = 0
    for k in range(1, len(xi)+1):
        E += optimality_cuts(x_value=x_opt, k=k)[0][0]
        e += optimality_cuts(x_value=x_opt, k=k)[1][0]
        y_opt[f"XI {k}"] = optimality_cuts(x_value=x_opt, k=k)[2]
        
    cuts[f"Cut {v+1}"] = {"E": E, "e": e}
    w = e - np.sum(E * np.array(x_opt))
    v += 1


print(f'\nOptimal value of theta: {opt_theta:.3f}\n')
for j in range(n):
    print(f'Optimal value of x{j+1}: {x_opt[j]:.3f}\n')
for i in y_opt.keys():
    print(f'Optimal values for y in {i}: {y_opt[i]}\n')
