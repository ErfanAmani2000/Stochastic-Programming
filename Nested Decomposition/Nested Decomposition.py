#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pulp
from pulp import *
import numpy as np
import math


# In[92]:


# Indices 
I = range(3)
Iy = range(2)
Iw = range(4)


# In[93]:


# Input Data
Cost = [150, 230, 260]
Price = [238, 210]
Revenue = [170, 150, 36, 10]
Required = [200, 240]

kisi = [[2, 2.4, 16],
        [2.5, 3, 20],
        [3, 3.6, 24]]

scenario_n = len(kisi)

S1 = range(scenario_n)
S2 = range(scenario_n**2)

p = [[1],
    [1/3 for s in S1],
    [1/9 for s in S2]]


# In[94]:


global NLDS_tk
NLDS_tk = [[0 for k in range(int(scenario_n**(t-1)))] 
             for t in range(1,4)]


# In[95]:


global DIR


# In[96]:


global pi_tk
pi_tk = [[0 for k in range(int(scenario_n**(t-1)))] 
         for t in range(1,4)]

global sigma_tk
sigma_tk = [[0 for k in range(int(scenario_n**(t-1)))] 
            for t in range(1,4)]


# In[97]:


global h_tk
h_tk = [[0 for k in range(int(scenario_n**(t-1)))] 
        for t in range(1,4)]

global T_tk
T_tk = [[0 for k in range(int(scenario_n**t))] 
        for t in range(1,3)]


# In[98]:


global e_tk
e_tk = [[[0] for k in range(int(scenario_n**(t-1)))] 
        for t in range(1,4)]

global s_tk
s_tk = [[0 for k in range(int(scenario_n**(t-1)))] 
            for t in range(1,4)]


# In[99]:


global counter
counter = 0

global opt_counter
opt_counter = 0

global iteration
iteration = 0

global optimality
optimality = False


# In[100]:


# Define & Solve All NLDS Problems

#Variables
## 1st stage
Theta_t1 = LpVariable("Theta1")
X_t1 = LpVariable.matrix("X1", I, lowBound=0) # Yield
## 2nd stage
Theta_t2 = LpVariable.matrix("Theta2", S1, lowBound=0) 
X_t2 = LpVariable.matrix("X2", (I,S1), lowBound=0)  # Yield
Y_t1 = LpVariable.matrix("Y1", (Iy,S1), lowBound=0) # Purchase
W_t1 = LpVariable.matrix("W1", (Iw,S1), lowBound=0) # Sale
## 3rd stage
Theta_t3 = LpVariable.matrix("Theta3", S2, lowBound=0) 
Y_t2 = LpVariable.matrix("Y2", (Iy,S2), lowBound=0) # Purchase
W_t2 = LpVariable.matrix("W2", (Iw,S2), lowBound=0) # Sale

for t in range(1,4):
     
    k_t = 3**(t-1)
    for k in range(1,k_t + 1):
        
        NLDS_Problem = LpProblem(f"NLDS(t:{t}_k:{k})", LpMinimize)
            
        if t == 1:
            
            #Objective Function    
            NLDS_Problem += lpSum(Cost[i]*X_t1[i] for i in I) + Theta_t1 

            #Constraints
            NLDS_Problem += lpSum(X_t1[i] for i in I) <= 500 # Total Area

            NLDS_Problem += Theta_t1 == 0, "theta=0"
            
            NLDS_Problem.solve()


        if t == 2:

            #Objective Function
            NLDS_Problem += (lpSum(Cost[i]*X_t2[i][k-1] for i in I)                              + (lpSum(Price[i]*Y_t1[i][k-1] for i in Iy)                                       - lpSum(Revenue[i]*W_t1[i][k-1] for i in Iw))                              + Theta_t2[k-1]) 

            #Constraints
            NLDS_Problem += lpSum(X_t2[i][k-1] for i in I) <= 500 # Total Area

            for i in Iy:
                NLDS_Problem += kisi[k-1][i]*(X_t1[i].varValue) + Y_t1[i][k-1] - W_t1[i][k-1] >= Required[i] # Minimum Needed

            NLDS_Problem += kisi[k-1][i]*(X_t1[3-1].varValue) - W_t1[3-1][k-1] - W_t1[4-1][k-1] >= 0 # Sugar Beets Logic
            NLDS_Problem += W_t1[3-1][k-1] <= 6000 # Sugar Beet's Saling Price Logic

            NLDS_Problem += (X_t1[3-1].varValue) + X_t2[3-1][k-1] <= 500 # Crop Rotation

            NLDS_Problem += Theta_t2[k-1] == 0, "theta=0"
            
            NLDS_Problem.solve()  


        if t == 3:        
            a_k = (k-1)//3 # Ancestor Scenario of k
            k3 = (k-1)%3 + 1 # kisi = 1, 2, or 3?

            #Objective Function
            NLDS_Problem += (lpSum(Price[i]*Y_t2[i][k-1] for i in Iy)                                    - lpSum(Revenue[i]*W_t2[i][k-1] for i in Iw)) + Theta_t3[k-1]

            #Constraints
            for i in Iy:
                NLDS_Problem += kisi[k3-1][i]*(X_t2[i][a_k].varValue) + Y_t2[i][k-1] - W_t2[i][k-1] >= Required[i] # Minimum Needed

            NLDS_Problem += kisi[k3-1][i]*(X_t2[3-1][a_k].varValue) - W_t2[3-1][k-1] - W_t2[4-1][k-1] >= 0 # Sugar Beets Logic
            NLDS_Problem += W_t2[3-1][k-1] <= 6000 # Sugar Beet's Saling Price Logic

            NLDS_Problem += Theta_t3[k-1] == 0, "theta=0"
                      
        NLDS_tk[t-1][k-1] = NLDS_Problem


# In[101]:


def Calculate_Matrices(t,k): 

    global h_tk, T_tk
    
    if t == 1:
        h = [500]
        
  
    if t == 2:
        h = [-500, Required[0], Required[1], 0, -6000, -500]
        
        T = [[0 for v in range(3)] for m in range(6)]
        
        
        for m in range(1,4):
            v = m-1
            T[m][v] = kisi[k-1][v]
            
        T[-1][-1] = -1

    
    if t == 3:
        h = [Required[0], Required[1], 0, -6000]
        
        T = [[0 for v in range(3)] for m in range(4)]
        
        k3 = (k-1)%3 + 1 # kisi = 1, 2, or 3?
        for m in range(3):
            v = m
            T[m][v] = kisi[k3-1][v]

    
    h_tk[t-1][k-1] = h
    if t != 1:
        T_tk[t-2][k-1] = T


# In[102]:


for t in range(1,4):
    k_t = 3**(t-1)
    for k in range(1,k_t + 1):
        Calculate_Matrices(t,k)


# In[103]:


def Check_Feasibility(NLDS):
    
    status = LpStatus[NLDS.status]
    if status == -1 or status == -2:
        if t == 1:
            print("The problem is Infeasible!")
        else:
            print("Feasibility Cut is needed!")
            DIR = "BACK"
            t -= 1
            k = (k-1)//3
            Step_1(t,k)


# In[104]:


def Update_Next_Stage_NLDS(t,k):
    
    global NLDS_tk
    
    u  = t + 1
    k_u = 3**(u-1)
    for k in range(1,k_u + 1):
        
        if u == 2:
            for c in range(2, 4):
                i = c-2
                NLDS_tk[u-1][k-1].constraints[f"_C{c}"] = (kisi[k-1][i]*(X_t1[i].varValue)                                                              + Y_t1[i][k-1] - W_t1[i][k-1] >= Required[i])
            
            
            NLDS_tk[u-1][k-1].constraints["_C4"] = (kisi[k-1][i]*(X_t1[3-1].varValue)                                                       - W_t1[3-1][k-1] - W_t1[4-1][k-1] >= 0)
            
            NLDS_tk[u-1][k-1].constraints["_C6"] = (X_t1[3-1].varValue) + X_t2[3-1][k-1] <= 500
            
            
        if u == 3:
            k3 = (k-1)%3 + 1
            for c in range(1,3):
                i = c-1
                NLDS_tk[u-1][k-1].constraints[f"_C{c}"] = (kisi[k3-1][i]*(X_t2[i][a_k].varValue)                                                              + Y_t2[i][k-1] - W_t2[i][k-1] >= Required[i])
                
            NLDS_tk[u-1][k-1].constraints["_C3"] = (kisi[k3-1][i]*(X_t2[3-1][a_k].varValue)                                                       - W_t2[3-1][k-1] - W_t2[4-1][k-1] >= 0)


# In[105]:


def Dual_Values(NLDS,t,k):
    
    global pi_tk, sigma_tk
    
    pi = list()
    sigma = list()
    for name, c in list(NLDS.constraints.items()):
        if name != "theta=0" and name != "feasibility_cut" and name[:-1] != "optimality_cut":
            pi.append(c.pi)
            
        if name[:-1] == "optimality_cut":
            sigma.append(c.pi)
    
    pi_tk[t-1][k-1] = pi
    sigma_tk[t-1][k-1] = sigma


# In[106]:


def Step_1(t,k):
    
    if optimality == True:
        return
    
    global NLDS_tk
    
    print("<<step1>>")
    print(f"t = {t}, k = {k}")
    
    global DIR
    
    NLDS = NLDS_tk[t-1][k-1]
    NLDS.solve()

    print(f"\n NLDS(t:{t}-k:{k})")
    print("Objective Value:", NLDS.objective.value())
    for v in NLDS.variables():
        print(v.name, "=", v.varValue)
        
    Check_Feasibility(NLDS)
    
    if t != 3:
        Update_Next_Stage_NLDS(t,k)
      
    Dual_Values(NLDS,t,k)
    
    k_t = 3**(t-1)
    if k < k_t:
        k += 1
        print("\n")
        Step_1(t,k)
        
    else:
        if t == 1:
            DIR = "FORE"
            
        if t < 3 and DIR == "FORE":
            t += 1
            k = 1
            print("\n")
            Step_1(t,k)
            
        elif t == 3:
            DIR = "BACK"
    
    
        if DIR == "BACK":
            print("\n")
            Step_2(t,k)


# In[107]:


def Calculate_E(u,j):

    E = list()
    
    C_k = range(3*(j-1)+1,3*j+1)  
       
    pi_T_uk = [np.dot(np.array(pi_tk[u][k-1]),np.array(T_tk[u-1][k-1])).tolist() for k in C_k]
    
    for v in range(len(pi_T_uk[0])):
        E_v = sum((p[u][k]/p[u-1][j-1])*pi_T_uk[k][v] for k in range(len(pi_T_uk)))
        E.append(E_v)
    
    return E


# In[108]:


def Calculate_e(u,j):
    
    global e_tk
    global s_tk
    
    C_k = range(3*(j-1),3*j)
    
    pi_h_uk = [np.dot(np.array(pi_tk[u][k-1]),np.array(h_tk[u][k-1])) for k in C_k]
    sigma_e_tk = [sum(sigma_tk[u][k][m]*e_tk[u][k][m+1] for m in range(s_tk[u][k])) for k in C_k]
    
    e = sum((p[u][k-1]/p[u-1][j-1])*(pi_h_uk[k] + sigma_e_tk[k]) for k in range(len(pi_h_uk)))
    
    e_tk[u-1][j-1].append(e)
    
    return e


# In[109]:


def Add_Optimality_Cut(t,k,E,e):
    
    global NLDS_tk
    
    if t == 1:        
        NLDS_tk[t-1][k-1] += lpSum(E[i]*X_t1[i] for i in range(len(E))) + Theta_t1 >= e, f"optimality_cut{counter}"
        
    if t == 2:
        NLDS_tk[t-1][k-1] += lpSum(E[i]*X_t2[i][k-1] for i in range(len(E))) + Theta_t2[k-1] >= e, f"optimality_cut{counter}"
        
    print("Optimality Cut:", NLDS_tk[t-1][k-1].constraints[f"optimality_cut{counter}"], "\n")


# In[110]:


def Step_2(t,k):
    
    global optimality
    
    if optimality == True:
        return

    
    global NLDS_tk
    
    print("<<step2>>")
    print(f"t = {t}")
    
    
    global counter
    global opt_counter
    global iteration
    
    opt_counter = 0

    if t == 1:
        t += 1
        
        iteration += 1
        print("\niteration", iteration)
        print("**************************************")

        Step_1(t,k)
        
    else:
        
        u = t - 1
        k_u = 3**(u-1)
        
        for j in range(1, k_u + 1):
            print(f"u = {u}, j = {j}")

            E = Calculate_E(u,j)
            e = Calculate_e(u,j) 

            
            if u == 1:
                theta_bar = e - sum(E[v]*(X_t1[v].varValue) for v in range(len(E)))
            if u == 2:
                theta_bar = e - sum(E[v]*(X_t2[v][j-1].varValue) for v in range(len(E)))
                
            theta_bar = round(theta_bar, 2)

            flag = 0
            for name, c in list(NLDS_tk[u-1][j-1].constraints.items()):
                if name == "theta=0":
                    flag = 1
                    del NLDS_tk[u-1][j-1].constraints["theta=0"]
                    Add_Optimality_Cut(u,j,E,e)
                    if u == 1:
                        opt_counter += 1
                    s_tk[u-1][j-1] = 1
                    counter += 1

            if flag == 0: # constraint "theta=0" does'nt exist
                if ((u == 1 and theta_bar > round(Theta_t1.varValue, 2)) 
                    or (u == 2 and (theta_bar > round(Theta_t2[j-1].varValue, 2)))):
                    Add_Optimality_Cut(u,j,E,e)
                    if u == 1:
                        opt_counter += 1
                    s_tk[u-1][j-1] += 1
                    counter += 1
        
        
    if t == 2 and opt_counter == 0:
        print("Achieved Optimal Solution")
        optimality = True
        return
                
    else:
        t -= 1
        k = 1
                
        if t == 1:
            DIR = "FORE"

       
    iteration += 1
    print("\niteration", iteration)
    print("**************************************")

    Step_1(t,k)


# In[111]:


t = 1
k = 1
iteration += 1
    
print("\niteration", iteration)
print("**************************************")

Step_1(t,k)


# In[112]:


print(f"\n NLDS(t:{t}-k:{k})")
for v in NLDS_tk[0][0].variables():
    print(v.name, "=", v.varValue)

    
for k in S1:
    print(f"\n NLDS(t:{t}-k:{k})")
    for v in NLDS_tk[1][k].variables():
        print(v.name, "=", v.varValue)


# In[113]:


Objective_Value = 0 

for t in range(1,4):
    k_t = 3**(t-1)
    for k in range(1,k_t + 1):
        
        Objective_Value += p[t-1][k-1]*NLDS_tk[t-1][k-1].objective.value()
        
print(Objective_Value)

