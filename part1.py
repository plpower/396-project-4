import csv
import numpy as np
from scipy.stats import geom
import math
import matplotlib.pyplot as plt


def exponential_weights(payoff_matrix, epsilon, low, h, n, m, step):
    total_revenue = 0
    actions_chosen = []

    for r in range(n):
        # (1) GENERATE M BIDS
        bids = generate_bids(low, h, m)
        sorted_bids = np.sort(bids)
        v_1 = sorted_bids[-1]
        v_2 = sorted_bids[-2]
        print('bids', bids)

        # (2) UPDATE REVENUE FOR EACH RESERVE PRICE IN PAYOFF MATRIX
        for reserve_price, val in payoff_matrix.items():
            if reserve_price > v_1:
                val.append(0)
            else:
                val.append(max(reserve_price, v_2))

        # (3) CALCULATE THE PROBABILITY
        # calculate the probability of chosing every action in round r
        probabilities, round_action = get_probabilities(r, epsilon, h, payoff_matrix, low, step)
        # chose action for round r with probabilities
        if r == 0:
            action_chosen = np.random.choice(round_action)
        else:
            action_chosen = np.random.choice(round_action, p=probabilities)
        actions_chosen.append(action_chosen)
        round_revenue = payoff_matrix[action_chosen][r]
        total_revenue += round_revenue

    average_revenue = total_revenue/n
    return actions_chosen, payoff_matrix, average_revenue


def get_probabilities(r, e, h, test_data, low, step):
    hindsight_payoffs = []
    total_payoff = 0
    probabilities = []
    curr_action = []

    # print(len(np.arange(0, my_value + 0.01, 0.01)))
    if r == 0:
        for action in np.arange(low, h+step, step):
            # action_payoff = test_data[action][0]
            curr_action.append(action)
            hindsight_payoff = 0
            hindsight_payoffs.append(hindsight_payoff)
            probs = [1 for _ in range(len(test_data))]
        return probs, curr_action
    else:
        for action in np.arange(low, h+step, step):
            # action_payoff = test_data[action][r]
            curr_action.append(action)
            hindsight_payoff = sum(test_data[action][:r])
            hindsight_payoffs.append((1+e) ** (hindsight_payoff/h))
        total_payoff = sum(hindsight_payoffs)

        for action in range(len(test_data)):
            # print(action)
            probabilities.append(hindsight_payoffs[action]/total_payoff)

        return probabilities, curr_action


# def calculate_regret(payoff_matrix, payoff, h):
#     # calculate OPT
#     action_bihs = []
#     for action in payoff_matrix:
#         action_bihs.append(sum(payoff_matrix[action]))
#     best_bih = max(action_bihs)

#     regret = (best_bih - payoff) / len(payoff_matrix[0])

#     return regret


def theo_opt_epsilon(k, n):
    epsilon = math.sqrt(np.log(k)/n)
    return epsilon

def generate_bids(low, high, m):
    ### generate m bids drawn from uniform distribution
    return np.random.uniform(low, high, m)

def generate_action_space(low, high, step):
    # start with the example of our bid of 45 with a given value of 50
    # discretize the bid space to k bids between 0 and v
    # action j corresponds to bidding bj

    action_space = {}

    for j in np.arange(low, high+step, step):
        action_space[j] = []

    print('action space', action_space)
    return action_space


if __name__ == "__main__":
    print('i like the view')
    print('you do?')
    print('youre my best view')
    print('meh!')

    ## experiment 1
    # bidders drawn from uniforma distribution of U[0,1]
    # action space: discretize reserve price from 0,1 with step size 100
    low = 0
    high = 1
    step = 0.01
    n = 1000
    m = 2

    payoff_matrix = generate_action_space(low, high, step)

    epsilon = theo_opt_epsilon(len(payoff_matrix), n)

    # print(epsilon)
  
    avg_regret = []
    # for i in range(100):
    actions_chosen, payoff_matrix, average_revenue = exponential_weights(payoff_matrix, epsilon, low, high, n, m, step)
    # regret = calculate_regret(test_data, total_payoff)
    # avg_regret.append(regret)
    print('actions chosen', actions_chosen)
    print('expected revenue', average_revenue)
    # print('payoff matrix', payoff_matrix)

    # print('theo payoff', np.average(total_payoff))
    # print("THEO", np.average(avg_regret))