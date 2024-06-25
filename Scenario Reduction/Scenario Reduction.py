import pandas as pd
import numpy as np  
import cvxpy as cp


class ScenarioProduction:
    def __init__(self, n_scenario, n_product, corn, wheat, sugarbeat):
        """
        Parameters:
        n_scenario (int): Number of scenarios.
        n_product (int): Number of products.
        corn (list): Range of corn yields.
        wheat (list): Range of wheat yields.
        sugarbeat (list): Range of sugarbeet yields.
        """
        self.n_scenario = n_scenario  # Set number of scenarios
        self.n_product = n_product  # Set number of products
        self.corn = corn  # Set corn yield range
        self.wheat = wheat  # Set wheat yield range
        self.sugarbeat = sugarbeat  # Set sugarbeet yield range


    def scenario_generator(self):
        """
        Generate scenarios and their probabilities.

        Returns:
        scenarios (numpy.ndarray): Generated scenarios.
        probabilities (numpy.ndarray): Corresponding probabilities.
        """
        scenarios = np.zeros((self.n_scenario, self.n_product))  # Initialize scenarios array with zeros
        probabilities = np.zeros(self.n_scenario)  # Initialize probabilities array with zeros
        random_numbers = list(np.random.random(size=(self.n_scenario - 1,)))  # Generate random numbers
        random_numbers.sort()  # Sort random numbers

        for ind, prob in enumerate(random_numbers):  # Iterate through sorted random numbers
            if ind == 0:  # First random number case
                probabilities[ind] = prob  # Set first probability
                multiplier = prob / 2  # Calculate multiplier
            elif ind == self.n_scenario - 2:  # Last random number before final case
                probabilities[ind] = (prob - random_numbers[ind - 1])  # Set probability difference
                probabilities[ind + 1] = (1 - prob)  # Set final probability
                multiplier = (1 + prob) / 2  # Calculate multiplier
                # Calculate final scenario values for corn, wheat, and sugarbeet
                scenarios[ind + 1][0] = self.corn[0] + (multiplier * (self.corn[1] - self.corn[0]))
                scenarios[ind + 1][1] = self.wheat[0] + (multiplier * (self.wheat[1] - self.wheat[0]))
                scenarios[ind + 1][2] = self.sugarbeat[0] + (multiplier * (self.sugarbeat[1] - self.sugarbeat[0]))
                multiplier = (prob + random_numbers[ind - 1]) / 2  # Recalculate multiplier
            else:  # Middle random numbers case
                probabilities[ind] = (prob - random_numbers[ind - 1])  # Set probability difference
                multiplier = (prob + random_numbers[ind - 1]) / 2  # Calculate multiplier

            # Calculate scenario values for corn, wheat, and sugarbeet
            scenarios[ind][0] = self.corn[0] + (multiplier * (self.corn[1] - self.corn[0]))
            scenarios[ind][1] = self.wheat[0] + (multiplier * (self.wheat[1] - self.wheat[0]))
            scenarios[ind][2] = self.sugarbeat[0] + (multiplier * (self.sugarbeat[1] - self.sugarbeat[0]))

        return scenarios, probabilities  # Return generated scenarios and probabilities


    @staticmethod
    def calculate_min_distance(remaining_scenarios, candidate_scenario, candidate_scenario_ind):
        """
        Calculate the minimum distance between a candidate scenario and remaining scenarios.

        Parameters:
        remaining_scenarios (list): Remaining scenarios.
        candidate_scenario (list): Candidate scenario.
        candidate_scenario_ind (int): Index of the candidate scenario.

        Returns:
        min_i_distance (float): Minimum distance to the closest scenario.
        """
        min_i_distance = np.inf  # Initialize minimum distance to infinity
        for ind, remained_scenario in enumerate(remaining_scenarios):  # Iterate through remaining scenarios
            if ind == candidate_scenario_ind:  # Skip the candidate scenario itself
                continue  # Continue to the next iteration
            else:
                linear_distance = np.array(remained_scenario) - np.array(candidate_scenario)  # Calculate linear distance
                distance_i_j = np.sum(linear_distance ** 2)  # Calculate squared distance
                if distance_i_j <= min_i_distance:  # Update minimum distance if smaller
                    min_i_distance = distance_i_j  # Set new minimum distance
        return min_i_distance  # Return minimum distance


    def backward_elimination_step_0(self, scenarios, probabilities):
        """
        Perform the initial step of backward elimination.

        Parameters:
        scenarios (numpy.ndarray): Scenarios to be considered.
        probabilities (numpy.ndarray): Corresponding probabilities.

        Returns:
        candidate (list): Selected candidate scenario.
        candidate_ind (int): Index of the selected candidate.
        candidate_prob (float): Probability of the selected candidate.
        """
        Z_min = np.inf  # Initialize minimum Z value to infinity
        candidate = None  # Initialize candidate scenario
        candidate_ind = None  # Initialize candidate index
        candidate_prob = None  # Initialize candidate probability

        for ind1, candidate_scenario in enumerate(scenarios):  # Iterate through scenarios
            min_i_distance = self.calculate_min_distance(scenarios, candidate_scenario, ind1)  # Calculate minimum distance
            Z_i = probabilities[ind1] * min_i_distance  # Calculate Z value
            if Z_i <= Z_min:  # Update minimum Z value and candidate if smaller
                Z_min = Z_i  # Set new minimum Z value
                candidate = candidate_scenario  # Set new candidate scenario
                candidate_ind = ind1  # Set new candidate index
                candidate_prob = probabilities[ind1]  # Set new candidate probability

        return candidate, candidate_ind, candidate_prob  # Return candidate scenario, index, and probability


    def backward_elimination(self, scenarios, probabilities):
        """
        Perform backward elimination to reduce the number of scenarios.

        Parameters:
        scenarios (numpy.ndarray): Scenarios to be reduced.
        probabilities (numpy.ndarray): Corresponding probabilities.

        Returns:
        remaining_scenarios (list): Remaining scenarios after elimination.
        remaining_probabilities (list): Probabilities of remaining scenarios.
        deleted_scenarios (list): Deleted scenarios.
        deleted_probabilities (list): Probabilities of deleted scenarios.
        """
        deleted_scenarios = []  # Initialize list of deleted scenarios
        deleted_probabilities = []  # Initialize list of deleted probabilities
        remaining_scenarios = scenarios.tolist()  # Convert scenarios to list
        remaining_probabilities = probabilities.tolist()  # Convert probabilities to list
        Z_min = np.inf  # Initialize minimum Z value to infinity
        candidate = None  # Initialize candidate scenario
        epsilon = 0.01  # Set epsilon threshold

        # Perform initial step of backward elimination
        first_deleted, first_deleted_ind, first_deleted_prob = self.backward_elimination_step_0(scenarios, probabilities)
        deleted_scenarios.append(first_deleted)  # Add first deleted scenario to list
        deleted_probabilities.append(first_deleted_prob)  # Add first deleted probability to list
        del remaining_scenarios[first_deleted_ind]  # Remove first deleted scenario from remaining
        del remaining_probabilities[first_deleted_ind]  # Remove first deleted probability from remaining

        while True:  # Loop until convergence
            for ind, candidate_scenario in enumerate(remaining_scenarios):  # Iterate through remaining scenarios
                min_i_distance = self.calculate_min_distance(remaining_scenarios, candidate_scenario, ind)  # Calculate minimum distance
                sum_probs = sum(deleted_probabilities) + remaining_probabilities[ind]  # Calculate sum of probabilities
                Z_i = min_i_distance * sum_probs  # Calculate Z value
                if Z_i <= Z_min:  # Update minimum Z value and candidate if smaller
                    Z_min = Z_i  # Set new minimum Z value
                    candidate = candidate_scenario  # Set new candidate scenario
                    candidate_ind = ind  # Set new candidate index
                    candidate_prob = remaining_probabilities[ind]  # Set new candidate probability

            if Z_min <= epsilon:  # Check if minimum Z value is within threshold
                deleted_scenarios.append(candidate)  # Add candidate to deleted scenarios
                deleted_probabilities.append(candidate_prob)  # Add candidate probability to deleted
                del remaining_scenarios[candidate_ind]  # Remove candidate scenario from remaining
                del remaining_probabilities[candidate_ind]  # Remove candidate probability from remaining
                Z_min = np.inf  # Reset minimum Z value
            else:
                break  # Exit loop if Z value exceeds threshold

        return remaining_scenarios, remaining_probabilities, deleted_scenarios, deleted_probabilities  # Return remaining and deleted scenarios and probabilities


    @staticmethod
    def calculate_nearest_node(remaining_scenarios, deleted_scenario):
        """
        Calculate the nearest node for a given deleted scenario.

        Parameters:
        remaining_scenarios (list): Remaining scenarios.
        deleted_scenario (list): Deleted scenario.

        Returns:
        selected_node (int): Index of the nearest node.
        """
        min_i_distance = np.inf  # Initialize minimum distance to infinity
        selected_node = None  # Initialize selected node
        for ind, remained_scenario in enumerate(remaining_scenarios):  # Iterate through remaining scenarios
            linear_distance = np.array(remained_scenario) - np.array(deleted_scenario)  # Calculate linear distance
            distance_i_j = np.sum(linear_distance ** 2)  # Calculate squared distance
            if distance_i_j <= min_i_distance:  # Update minimum distance if smaller
                min_i_distance = distance_i_j  # Set new minimum distance
                selected_node = ind  # Set new selected node
        return selected_node  # Return index of nearest node

    @staticmethod
    def clustering_scenarios(remaining_scenarios, remaining_probabilities, deleted_scenarios, deleted_probabilities):
        """
        Recalculate the probabilities of remaining scenarios by clustering deleted scenarios.

        Parameters:
        remaining_scenarios (list): Remaining scenarios.
        remaining_probabilities (list): Probabilities of remaining scenarios.
        deleted_scenarios (list): Deleted scenarios.
        deleted_probabilities (list): Probabilities of deleted scenarios.

        Returns:
        remaining_probabilities (list): Updated probabilities of remaining scenarios.
        """
        for ind, deleted_scenario in enumerate(deleted_scenarios):  # Iterate through deleted scenarios
            nearest_node = ScenarioProduction.calculate_nearest_node(remaining_scenarios, deleted_scenario)  # Find nearest remaining scenario
            remaining_probabilities[nearest_node] += deleted_probabilities[ind]  # Add deleted scenario probability to nearest remaining scenario
        return remaining_probabilities  # Return updated probabilities



corn = [2.4, 3.6]  # Define range for corn yield
wheat = [2, 3]  # Define range for wheat yield
sugarbeat = [16, 24]  # Define range for sugarbeet yield
n_scenario = 200  # Set number of scenarios
n_product = 3  # Set number of products
n_period = 2  # Number of periods
n_product_to_buy = 2  # Number of products to buy
n_product_to_sell = 4  # Number of products to sell


scenario_production = ScenarioProduction(n_scenario, n_product, corn, wheat, sugarbeat)  # Create ScenarioProduction instance

# Generate and reduce first year scenarios
first_year_scenarios, first_year_probs = scenario_production.scenario_generator()
first_remaining_scenarios, first_remaining_probs, first_deleted_scenarios, first_deleted_probs = scenario_production.backward_elimination(first_year_scenarios, first_year_probs)

# Generate and reduce second year scenarios
second_year_scenarios, second_year_probs = scenario_production.scenario_generator()
second_remaining_scenarios, second_remaining_probs, second_deleted_scenarios, second_deleted_probs = scenario_production.backward_elimination(second_year_scenarios, second_year_probs)

# Update probabilities by clustering deleted scenarios
first_remaining_probs = ScenarioProduction.clustering_scenarios(first_remaining_scenarios, first_remaining_probs, first_deleted_scenarios, first_deleted_probs)
second_remaining_probs = ScenarioProduction.clustering_scenarios(second_remaining_scenarios, second_remaining_probs, second_deleted_scenarios, second_deleted_probs)

# Define parameters
n_first_scenarios = len(first_remaining_probs)  # Number of first year remaining scenarios
n_second_scenarios = len(second_remaining_probs)  # Number of second year remaining scenarios


print(f'Total number of generated scenarios: {len(first_year_scenarios) * len(second_year_scenarios)}')
print(f'Number of remaining scenarios after backward elimination: {n_first_scenarios * n_second_scenarios}')


# Define cost parameters
C = [150, 230, 260]  # Planting cost
B = [238, 210]  # Buying cost
d = [200, 240]  # Required production
S = [170, 150, 36, 10]  # Selling price

# Scenarios and probabilities from the ScenarioProduction class
scenarios = [first_remaining_scenarios, second_remaining_scenarios]
probs = [first_remaining_probs, second_remaining_probs]

# Decision Variables
x = [[cp.Variable(nonneg=True) for i in range(n_product)] for t in range(n_period)]  # Planting decision variables
y = [[[cp.Variable(nonneg=True) for i in range(n_product_to_buy)] for s in range(len(scenarios[t]))] for t in range(n_period)]  # Buying decision variables
w = [[[cp.Variable(nonneg=True) for i in range(n_product_to_sell)] for s in range(len(scenarios[t]))] for t in range(n_period)]  # Selling decision variables

constraints = []  # Initialize list of constraints

# Add constraints
for t in range(n_period):
    constraints.append(cp.sum([x[t][i] for i in range(n_product)]) <= 500)  # Constraint equation (2)
    for s in range(len(scenarios[t])):
        constraints.append(scenarios[t][s][2] * x[t][2] >= w[t][s][2] + w[t][s][3])  # Constraint equation (4)
        constraints.append(w[t][s][2] <= 6000)  # Constraint equation (5)
        for i in range(n_product_to_buy):
            constraints.append(scenarios[t][s][i] * x[t][i] + y[t][s][i] - w[t][s][i] >= d[i])  # Constraint equation (3)

constraints.append(x[0][2] + x[1][2] <= 500)  # Constraint equation (6)

# Define the objective function
objective = cp.Minimize(
    cp.sum([cp.sum([C[i] * x[t][i] for i in range(n_product)]) for t in range(n_period)]) +
    cp.sum([cp.sum([cp.sum([B[i] * y[t][s][i] for i in range(n_product_to_buy)]) for s in range(len(scenarios[t]))]) for t in range(n_period)]) -
    cp.sum([cp.sum([cp.sum([S[i] * w[t][s][i] for i in range(n_product_to_sell)]) for s in range(len(scenarios[t]))]) for t in range(n_period)])
)

# Create and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Print the optimal value of the objective function
print(f"Optimal value: {problem.value:.3f}")

# Extract and organize the values
for t in range(n_period):
  print(f'Planting Area for wheat, corn and sugarbeat in the begining of period {t+1}: ({x[t][0].value:.3f}, {x[t][1].value:.3f}, {x[t][2].value:.3f})')

products = ['Wheat', 'Corn', 'Sugarbeat (Normal Price)', 'Sugarbeat (Low Price)']
data = []

for t in range(n_period):
    for s in range(len(scenarios[t])):
        planting_arcs = [round(x[t][i].value, 3) for i in range(n_product)]
        need_to_buy = [round(y[t][s][i].value if i < len(y[t][s]) else 0.0, 3) for i in range(n_product_to_buy)]
        sell_value = [round(w[t][s][i].value if i < len(w[t][s]) else 0.0, 3) for i in range(n_product_to_sell)]
        
        # Prepare data for the DataFrame
        data.append([
            ['Planting Arcs'] + planting_arcs,
            ['Need to Buy'] + need_to_buy + [0.0] * (len(products) - len(need_to_buy)),  # Adjust the length
            ['Sell Value'] + sell_value + [0.0] * (len(products) - len(sell_value))  # Adjust the length
        ])

number_of_scenario_to_show = 4
# Print data in the specified format
for i, scenario_data in enumerate(data[:number_of_scenario_to_show]): # Show only the output of first m scenarios
    df = pd.DataFrame(scenario_data, columns=['Product'] + products)
    print(f'Scenario {i+1}')
    print(df.to_string(index=False))
    print('-----------------------------------------------------------------------------')
