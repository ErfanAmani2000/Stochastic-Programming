from pulp import *
import numpy as np


LpSolverDefault.msg = 0  


class Nested_Decomposition_Optimizer:
    def __init__(self, scenario, H, T, h, c, q, probability, node_gen, x1, x2, y2, w2, y3, w3, theta1, theta2):
        self.scenario = scenario
        self.H = H
        self.T = T
        self.h = h
        self.c = c
        self.q = q
        self.probability = probability
        self.node_gen = node_gen
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2
        self.w2 = w2
        self.y3 = y3
        self.w3 = w3
        self.theta1 = theta1
        self.theta2 = theta2
        self.total_scenario = self.scenario[1]*self.scenario[2]


    def initialize_and_solve_NLDS(self, scenario, T, h, c, q1, q2):
        # Initialize NLDS dictionary
        NLDS = {1: [], 2: [], 3: []}

        # Stage 1
        NLDS[1] = LpProblem(name="NLDS(1-1)", sense=LpMinimize)  # Create first-stage problem
        NLDS[1] += pulp.lpSum([c[j] * self.x1[j] for j in range(3)]) + self.theta1  # Objective function
        NLDS[1] += pulp.lpSum([self.x1[j] for j in range(3)]) <= 500  # Constraint on x1 variables
        NLDS[1] += (self.theta1 == 0, "first_iteration")  # First iteration constraint
        NLDS[1].solve()  # Solve the first-stage problem

        # Stage 2
        for k in range(scenario[1]):
            NLDS[2].append(LpProblem(f"NLDS(2-{k+1})", sense=LpMinimize))  # Create second-stage subproblems
            NLDS[2][k] += (pulp.lpSum([c[j] * self.x2[j, k] for j in range(3)]) + 
                           pulp.lpSum([q1[j] * self.y2[j, k] for j in range(2)]) +
                           pulp.lpSum([q2[j] * self.w2[j, k] for j in range(4)]) + 
                           self.theta2[k])  # Objective function
            NLDS[2][k] += pulp.lpSum([self.x2[j, k] for j in range(3)]) <= 500  # Constraint on x2 variables
            NLDS[2][k] += self.x2[2, k] <= 500 - self.x1[2].value()  # Linking constraint with first-stage decision
            NLDS[2][k] += self.y2[0, k] - self.w2[0, k] >= 200 - T[1][k][2][0] * self.x1[0].value()  # Scenario-specific constraint
            NLDS[2][k] += self.y2[1, k] - self.w2[1, k] >= 240 - T[1][k][3][1] * self.x1[1].value()  # Scenario-specific constraint
            NLDS[2][k] += self.w2[2, k] + self.w2[3, k] <= -T[1][k][4][2] * self.x1[2].value()  # Scenario-specific constraint
            NLDS[2][k] += self.w2[2, k] <= 6000  # Upper bound constraint on w2
            NLDS[2][k] += (self.theta2[k] == 0, "first_iteration")  # First iteration constraint
            NLDS[2][k].solve()  # Solve the second-stage subproblem

        # Stage 3
        counter = 0
        for scenario1 in range(scenario[1]):
            for scenario2 in range(scenario[2]):
                NLDS[3].append(LpProblem(f"NLDS(3-{counter+1})", sense=LpMinimize))  # Create third-stage subproblems
                NLDS[3][counter] += (pulp.lpSum([q1[j] * self.y3[j, scenario1, scenario2] for j in range(2)]) +
                                     pulp.lpSum([q2[j] * self.w3[j, scenario1, scenario2] for j in range(4)]))  # Objective function
                NLDS[3][counter] += self.y3[0, scenario1, scenario2] - self.w3[0, scenario1, scenario2] >= 200 - T[2][scenario2][0][0] * self.x2[0, scenario1].value()  # Scenario-specific constraint
                NLDS[3][counter] += self.y3[1, scenario1, scenario2] - self.w3[1, scenario1, scenario2] >= 240 - T[2][scenario2][1][1] * self.x2[1, scenario1].value()  # Scenario-specific constraint
                NLDS[3][counter] += self.w3[2, scenario1, scenario2] + self.w3[3, scenario1, scenario2] <= -T[2][scenario2][2][2] * self.x2[2, scenario1].value()  # Scenario-specific constraint
                NLDS[3][counter] += self.w3[2, scenario1, scenario2] <= 6000  # Upper bound constraint on w3
                NLDS[3][counter].solve()  # Solve the third-stage subproblem
                counter += 1  # Increment the counter for the next scenario
        return NLDS  # Return the solved NLDS problems


    def optimality_cut(self, NLDS, t, x_variable, E1, e1, theta1, supple_e1_int, cut_number):
        # Add optimality cut to the problem
        NLDS[t-1] += lpDot(x_variable, E1) + theta1 >= supple_e1_int
        x_variable = []  # Reset the list of variables
        e1.append(supple_e1_int)  # Append the new cut value
        cut_number += 1  # Increment the cut counter
        return NLDS, x_variable, e1, E1, cut_number  # Return updated values


    def step1(self, NLDS, t, direction, pi, sigma):
        x = []  # Initialize a list to store dual variables
        if t == 1:
            NLDS[t].solve()  # Solve the first-stage problem
            for k in range(self.scenario[1]):
                # Update constraints for the second-stage subproblems
                NLDS[t+1][k].constraints["_C2"] = self.x2[2, k] <= 500 - self.x1[2].value()
                NLDS[t+1][k].constraints["_C3"] = self.y2[0, k] - self.w2[0, k] >= 200 - self.T[t][k][2][0]* self.x1[0].value()
                NLDS[t+1][k].constraints["_C4"] = self.y2[1, k] - self.w2[1, k] >= 240 - self.T[t][k][3][1]* self.x1[1].value()
                NLDS[t+1][k].constraints["_C5"] = self.w2[2, k] + self.w2[3, k] <= - self.T[t][k][4][2]* self.x1[2].value()
            direction = "FORE"  # Set direction to forward
            if (t < self.H and direction == "FORE"):
                t += 1  # Increment the time period
                return self.step1(NLDS, t, direction, pi, sigma)  # Recursive call for next time period
            else:
                return NLDS, t, direction, pi, sigma  # Return if end of the horizon or direction changes

        if t == 2:
            for k in range(self.scenario[1]):
                NLDS[t][k].solve()  # Solve the second-stage subproblems
                for i in range(len(list(NLDS[t][k].constraints.items()))):
                    x.append(list(NLDS[t][k].constraints.items())[i][1].pi)  # Collect dual variables
                pi[t][k] = x[:6]  # Store dual variables for constraints
                sigma[t][k] = x[6:]  # Store remaining dual variables
                x = []  # Reset the list for the next subproblem

            counter = 0
            for scenario1 in range(self.scenario[1]):
                for scenario2 in range(self.scenario[2]):
                    # Update constraints for third-stage subproblems
                    NLDS[t+1][counter].constraints["_C1"] = self.y3[0, scenario1, scenario2] - self.w3[0, scenario1, scenario2] >= 200 - self.T[t][scenario2][0][0]* self.x2[0, scenario1].value()
                    NLDS[t+1][counter].constraints["_C2"] = self.y3[1, scenario1, scenario2] - self.w3[1, scenario1, scenario2] >= 240 - self.T[t][scenario2][1][1]* self.x2[1, scenario1].value()
                    NLDS[t+1][counter].constraints["_C3"] = self.w3[2, scenario1, scenario2] + self.w3[3, scenario1, scenario2] <= - self.T[t][scenario2][2][2]* self.x2[2, scenario1].value()
                    counter += 1
            if (t < self.H and direction == "FORE"):
                t += 1  # Increment the time period
                return self.step1(NLDS, t, direction, pi, sigma)  # Recursive call for next time period
            else:
                return NLDS, t, direction, pi, sigma  # Return if end of the horizon or direction changes

        if t == 3:
            for k in range(self.total_scenario):
                NLDS[t][k].solve()  # Solve the third-stage subproblems
                for i in range(len(list(NLDS[t][k].constraints.items()))):
                    x.append(list(NLDS[t][k].constraints.items())[i][1].pi)  # Collect dual variables
                pi[t][k] = x[:]  # Store all dual variables
                x = []  # Reset the list for the next subproblem
            direction = "BACK"  # Set direction to backward
            return NLDS, t, direction, pi, sigma  # Return and switch to step2 for backward phase


    def step2(self, NLDS, e1, e2, t, direction, pi, sigma):
        cut_number = 0  # Initialize cut counter
        if t == 1:
            t += 1  # Increment the time period
            self.step1(NLDS, t, direction, pi, sigma)  # Call step1 for the next time period
        x_variable = []  # Initialize list for variables
        x_value = []  # Initialize list for variable values

        if t == 3:
            counter = 0
            for k in range(self.node_gen[t]):
                supple_E2 = np.zeros((self.scenario[2], 3))  # Initialize supplemental E2 array
                supple_e2 = np.zeros(self.scenario[2])  # Initialize supplemental e2 array
                if (len(e2[k]) == 0):
                    x_variable.extend((self.x2[0, k], self.x2[1, k], self.x2[2 ,k]))  # Add decision variables
                    del NLDS[t-1][k].constraints["first_iteration"]  # Remove initial iteration constraint

                    for m in range(self.scenario[2]):
                        supple_pi = np.array(pi[t][counter])  # Convert dual variables to array
                        supple_E2[m] = (((self.probability[t][m])/(self.probability[t-1][k]))*(supple_pi@self.T[t-1][m]))
                        supple_e2[m] = (((self.probability[t][m])/(self.probability[t-1][k]))*(supple_pi@self.h[t]))
                        counter += 1

                    E2 = np.sum(supple_E2, axis = 0)  # Sum E2 over scenarios
                    supple_e2_int = np.sum(supple_e2, axis = 0)  # Sum e2 over scenarios

                    NLDS[t-1][k] += lpDot(x_variable, E2) + self.theta2[k] >= supple_e2_int  # Add optimality cut
                    x_variable = []  # Reset variable list
                    e2[k].append(supple_e2_int)  # Append new cut value
                else:
                    x_variable.extend((self.x2[0, k], self.x2[1, k], self.x2[2, k]))  # Add decision variables
                    x_value.extend((self.x2[0, k].value(), self.x2[1, k].value(), self.x2[2, k].value()))  # Add variable values

                    for m in range(self.scenario[2]):
                        supple_pi = np.array(pi[t][counter])  # Convert dual variables to array
                        supple_E2[m] = (((self.probability[t][m])/(self.probability[t-1][k]))*(supple_pi@self.T[t-1][m]))
                        supple_e2[m] = (((self.probability[t][m])/(self.probability[t-1][k]))*(supple_pi@self.h[t]))
                        counter += 1

                    E2 = np.sum(supple_E2, axis = 0)  # Sum E2 over scenarios
                    supple_e2_int = np.sum(supple_e2, axis = 0)  # Sum e2 over scenarios
                    theta_hat = round((supple_e2_int - E2@x_value), 2)  # Compute theta hat
                    x_value = []  # Reset value list

                    if theta_hat > self.theta2[k].value():
                        NLDS[t-1][k] += lpDot(x_variable, E2) + self.theta2[k] >= supple_e2_int  # Add optimality cut
                        x_variable = []  # Reset variable list
                        e2[k].append(supple_e2_int)  # Append new cut value
            t = t-1  # Decrement the time period
            return NLDS, True, e1, e2, t, direction  # Return and indicate continuation

        if t == 2:
            supple_E1 = np.zeros((self.scenario[1], 3))  # Initialize supplemental E1 array
            supple_e1 = np.zeros(self.scenario[1])  # Initialize supplemental e1 array
            if (len(e1) == 0):
                x_variable.extend((self.x1[0], self.x1[1], self.x1[2]))  # Add decision variables
                del NLDS[t-1].constraints["first_iteration"]  # Remove initial iteration constraint

                for m in range(self.scenario[1]):
                    supple_pi = np.array(pi[t][m])  # Convert dual variables to array
                    supple_sigma = np.array(sigma[t][m])  # Convert sigma to array
                    e2_array = np.array(e2[m])  # Convert e2 to array
                    supple_E1[m] = (((self.probability[t][m])/(self.probability[t-1]))*(supple_pi@self.T[t-1][m])) 
                    supple_e1[m] = (((self.probability[t][m])/(self.probability[t-1]))*(supple_pi@self.h[t] + supple_sigma@e2_array))

                E1 = np.sum(supple_E1, axis = 0)  # Sum E1 over scenarios
                supple_e1_int = np.sum(supple_e1, axis = 0)  # Sum e1 over scenarios
                NLDS[t-1] += lpDot(x_variable, E1) + self.theta1 >= supple_e1_int  # Add optimality cut
                x_variable = []  # Reset variable list
                e1.append(supple_e1_int)  # Append new cut value
                cut_number += 1  # Increment cut counter
            else:
                x_variable.extend((self.x1[0], self.x1[1], self.x1[2]))  # Add decision variables
                x_value.extend((self.x1[0].value(), self.x1[1].value(), self.x1[2].value()))  # Add variable values

                for m in range(self.scenario[1]):
                    supple_pi = np.array(pi[t][m])  # Convert dual variables to array
                    supple_sigma = np.array(sigma[t][m])  # Convert sigma to array
                    e2_array = np.array(e2[m])  # Convert e2 to array
                    supple_E1[m] = (((self.probability[t][m])/(self.probability[t-1]))*(supple_pi@self.T[t-1][m])) 
                    supple_e1[m] = (((self.probability[t][m])/(self.probability[t-1]))*(supple_pi@self.h[t] + supple_sigma@e2_array))

                E1 = np.sum(supple_E1, axis = 0)  # Sum E1 over scenarios
                supple_e1_int = np.sum(supple_e1)  # Sum e1 over scenarios
                theta_hat = round((supple_e1_int - E1@x_value), 2)  # Compute theta hat
                x_value = []  # Reset value list

                if theta_hat > self.theta1.value():
                    NLDS, x_variable, e1, E1, cut_number = self.optimality_cut(NLDS, t, x_variable, E1, e1, self.theta1, supple_e1_int, cut_number)  # Add optimality cut

            if cut_number == 0:
                return NLDS, False, e1, e2, t, direction  # Return and indicate no cuts added
            else:
                t = t-1  # Decrement the time period
                direction = "FORE"  # Set direction to forward
                return NLDS, True, e1, e2, t, direction  # Return and indicate continuation
    

    def run_algorithm(self):
        # Unpack cost coefficients for y and w variables
        q1, q2 = self.q
        
        # Initialize and solve NLDS for stages 1, 2, and 3
        NLDS = self.initialize_and_solve_NLDS(self.scenario, self.T, self.h, self.c, q1, q2)
        
        # Initialize iteration count, stage, and other parameters
        iter, t, k = 0, 1, 0
        direction, preceeding_condition = "FORE", True  # Set initial direction and condition
        e1, e2 = [], [[] for i in range(11)]  # Initialize e1 and e2 lists for cuts

        # Initialize dictionaries for dual variables (pi and sigma)
        pi = {2: [0]*self.scenario[1], 3: [0]*self.total_scenario}
        sigma = {2: [0]*self.scenario[1], 3: [0]}

        # Main loop of the nested decomposition algorithm
        while preceeding_condition:
            if t == 1:
                iter += 1  # Increment iteration count for stage 1
            # Execute step1 of the algorithm
            NLDS, t, direction, pi, sigma = self.step1(NLDS, t, direction, pi, sigma)
            # Execute step2 of the algorithm
            NLDS, preceeding_condition, e1, e2, t, direction = self.step2(NLDS, e1, e2, t, direction, pi, sigma)
        
        # Print the number of iterations taken to reach the optimal solution
        print(f"Optimal solution is reached after {iter} iterations.")
        
        # Print the objective function value and optimal values of stage 1 variables
        print("The objective function and its corresponding variables' optimal value are: ", NLDS[1].objective.value())
        return NLDS



# Define the scenario dictionary
scenario = {1: 10, 2: 10}

# Generate the scenario_T data
scenario_T = np.column_stack((
    np.linspace(2, 3, 10), 
    np.linspace(2.4, 3.6, 10), 
    np.linspace(-16, -24, 10)
)).tolist() 

# Initialize and populate T1
T1 = np.array([
    [
        [0, 0, 0],
        [0, 0, 1],
        [scenario_T[index][0], 0, 0],
        [0, scenario_T[index][1], 0],
        [0, 0, scenario_T[index][2]],
        [0, 0, 0]
    ] for index in range(scenario[1])
])

# Initialize and populate T2
T2 = np.array([
    [
        [scenario_T[index][0], 0, 0],
        [0, scenario_T[index][1], 0],
        [0, 0, scenario_T[index][2]],
        [0, 0, 0]
    ] for index in range(scenario[2])
])

# Create the T dictionary
T = {1: T1, 2: T2}

# Define probabilities for each stage
prob = {1: 1, 2: 0.1*np.ones(10), 3: 0.01*np.ones(100)}  # Probabilities for stages 1, 2, and 3

# Calculate the total number of scenarios for stage 3
total_scenario = scenario[1]*scenario[2]  # Total number of scenarios

# Define cost coefficients and constraints for stages 2 and 3
h = {2: np.array([500, 500, 200, 240, 0, 6000]), 3: np.array([200, 240, 0, 6000])}  # Constraints for stages 2 and 3
c = np.array([150, 230, 260])  # Cost coefficients for x variables
q = (np.array([238, 210]), np.array([-170, -150, -36, -10]))  # Cost coefficients for y and w variables

# Define the number of nodes generated for each stage
node_gen = {2: 1, 3: 10}  # Number of nodes for stages 2 and 3

# Define decision variables
theta2 = LpVariable.dicts('theta2', [j for j in range(10)], cat="Continuous")  # Theta variables for stage 2
theta1 = LpVariable('theta1', cat="Continuous")  # Theta variable for stage 1
x1 = LpVariable.dicts('x1', [j for j in range(3)], 0, cat="Continuous")  # x variables for stage 1
x2 = LpVariable.dicts('x2', [(j, scenario1) for j in range(3) for scenario1 in range(scenario[1])], 0, cat="Continuous")  # x variables for stage 2
y2 = LpVariable.dicts('y2', [(j, scenario1) for j in range(2) for scenario1 in range(scenario[1])], 0, cat="Continuous")  # y variables for stage 2
w2 = LpVariable.dicts('w2', [(j, scenario1) for j in range(4) for scenario1 in range(scenario[1])], 0, cat="Continuous")  # w variables for stage 2
y3 = LpVariable.dicts('y3', [(j, scenario1, scenario2) for j in range(2) for scenario1 in range(scenario[1]) for scenario2 in range(scenario[2])], 0, cat="Continuous")  # y variables for stage 3
w3 = LpVariable.dicts('w3', [(j, scenario1, scenario2) for j in range(4) for scenario1 in range(scenario[1]) for scenario2 in range(scenario[2])], 0, cat="Continuous")  # w variables for stage 3

# Initialize the optimizer with all necessary parameters
optimizer = Nested_Decomposition_Optimizer(
    scenario=scenario, H=3, T=T, h=h, c=c, q=q, probability=prob, node_gen=node_gen,
    x1=x1, x2=x2, y2=y2, w2=w2, y3=y3, w3=w3, theta1=theta1, theta2=theta2
)

# Run the nested decomposition algorithm
NLDS = optimizer.run_algorithm()

# Print the optimal values of variables for all scenarios in stage 1
for variable in NLDS[1].variables():
    print(f"{variable.name}: {variable.value()}")

# Print the optimal values of variables for all scenarios in stage 2
for i in range(10):
    for variable in NLDS[2][i].variables():
        print(f"{variable.name}: {variable.value()}")

# Print the optimal values of variables for all scenarios in stage 3
for i in range(100):
    for variable in NLDS[3][i].variables():
        print(f"{variable.name}: {variable.value()}")
