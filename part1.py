import csv
import numpy as np
from scipy.stats import geom
import math
import matplotlib.pyplot as plt


def exponential_weights(payoff_matrix, epsilon, low, h, n, m, step):
    total_revenue = 0
    actions_chosen = []
    all_revenues = []

    for r in range(n):
        # (1) GENERATE M BIDS
        bids = generate_bids(low, h, m)
        sorted_bids = np.sort(bids)
        v_1 = sorted_bids[-1]
        v_2 = sorted_bids[-2]

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
        all_revenues.append(round_revenue)

    average_revenue = total_revenue/n
    return actions_chosen, payoff_matrix, average_revenue, all_revenues

def exponential_weights_quadratic(payoff_matrix, epsilon, low, h, n, m, step):
    total_revenue = 0
    actions_chosen = []
    all_revenues = []

    for r in range(n):
        # (1) GENERATE M BIDS
        bids = generate_bids_quadratic()
        sorted_bids = np.sort(bids)
        v_1 = sorted_bids[-1]
        v_2 = sorted_bids[-2]

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
        all_revenues.append(round_revenue)

    average_revenue = total_revenue/n
    return actions_chosen, payoff_matrix, average_revenue, all_revenues

def get_probabilities(r, e, h, test_data, low, step):
    hindsight_payoffs = []
    total_payoff = 0
    probabilities = []
    curr_action = []

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



def theo_opt_epsilon(k, n):
    epsilon = math.sqrt(np.log(k)/n)
    return epsilon

def generate_bids(low, high, m):
    ### generate m bids drawn from uniform distribution
    return np.random.uniform(low, high, m)

def generate_bids_quadratic():
    # draw from quadratic cumulative distribution function F(z) = z^2
    qs = np.random.uniform(0, 1, 2)
    bids = []
    for q in range(len(qs)):
        bids.append(math.sqrt(qs[q]))
    print('bids', bids)
    return bids

def generate_action_space(low, high, step):
    # start with the example of our bid of 45 with a given value of 50
    # discretize the bid space to k bids between 0 and v
    # action j corresponds to bidding bj

    action_space = {}

    for j in np.arange(low, high+step, step):
        action_space[j] = []

    # print('action space', action_space)
    return action_space

def expected_optimal_revenue(h, optimal_reserve_price, m):
    ## ONLY WORKS FOR UNIFORM DISTRIBUTION U[0,h]
    prob = []
    expected_revenue = []
    # consider m+1 cases
    for case in range(m+1):
        if case == 0:
            # reserve price > all bids --> dont sell
            prob.append(1/(2**m))
            expected_revenue.append(0)
        else:
            # case = number of values above reserve price
            prob.append(1/(2**case))
            # expected revenue = E[v2 | v1...vcase > 1/2]
            values = np.linspace(optimal_reserve_price, h, case+2)
            # print(values)
            expected_revenue.append(values[-3])
            # print(values[-3])
    
    prob = np.asarray(prob)
    expected_revenue = np.asarray(expected_revenue)
    total_expected_revenue = np.multiply(prob, expected_revenue)
    total_expected_revenue = np.sum(total_expected_revenue)
    return total_expected_revenue

def calculate_regret(h, optimal_reserve_price, m, alg_avg_rev):
    # ONLY FOR UNIFORM DISTRIBUTIONS
    opt_reserve_price = h/2
    expected_opt_rev = expected_optimal_revenue(h, opt_reserve_price, m)
    regret = expected_opt_rev - alg_avg_rev
    return regret

if __name__ == "__main__":
    # print('i like the view')
    # print('you do?')
    # print('youre my best view')
    # print('meh!')

    ## experiment 1
    # bidders drawn from uniforma distribution of U[0,1]
    # action space: discretize reserve price from 0,1 with step size 100
    # low = 0
    # high = 1
    # step = 0.1
    # n = 100
    # m = 10
    # epoch = 100
    
    # payoff_matrix = generate_action_space(low, high, step)
    # epsilon = theo_opt_epsilon(len(payoff_matrix), n)

    # GENERAL CASE - U[0,1] 2 bidders 
    # m = 2
    # low = 0
    # high = 1
    # step = 0.01
    # n = 100
    # payoff_matrix = generate_action_space(low, high, step)
    # epsilon = theo_opt_epsilon(len(payoff_matrix), n)
    # actions_chosen, payoff_matrix, average_revenue, all_revenues = exponential_weights(payoff_matrix, epsilon, low, high, n, m, step)    

    # plot_reg = []
    # plot_rev = []

    # for i, rev in enumerate(all_revenues):
    #     plot_reg.append((0.41 - rev)/(i+1))
    #     plot_rev.append(sum(all_revenues[:i])/(i+1))
    
    # plt.figure(1)
    # plt.plot(np.arange(n), actions_chosen, label="EW")
    # plt.plot(np.arange(n), l, label="Optimal")
    # plt.title("Reserve Price Over Time for 2 Bidders U[0,1]")
    # plt.legend(loc="upper right")
    # plt.ylabel("reserve price")
    # plt.xlabel("round")
    # plt.savefig('Part1_reserve_vs_round_VAR1.png')

    # plt.figure(1)
    # plt.plot(np.arange(n), plot_reg, label="EW")
    # plt.title("Regret Over Time for 2 Bidders U[0,1]")
    # plt.ylabel("regret")
    # plt.xlabel("round")
    # plt.savefig('Part1_REGRET.png')

    # plt.figure(2)
    # plt.plot(np.arange(n), plot_rev, label="EW")
    # plt.title("Revenue Over Time for 2 Bidders U[0,1]")
    # plt.ylabel("revenue")
    # plt.xlabel("round")
    # plt.savefig('Part1_REVENUE.png')

    # VARIATION ON NUMBER OF BIDDERS
    # num_bidders = np.arange(2, 100)
    # revenue_per_m = []
    # regret_per_m = []
    # for m in range(2, 100):
    #     payoff_matrix = generate_action_space(low, high, step)
    #     epsilon = theo_opt_epsilon(len(payoff_matrix), n)
    #     actions_chosen, payoff_matrix, average_revenue, all_revenues = exponential_weights(payoff_matrix, epsilon, low, high, n, m, step)
    #     revenue_per_m.append(average_revenue)
    #     regret_per_m.append(calculate_regret(high, high/2, m, average_revenue))
    #     print('done with round m', m)

    # # VARIATION ON NUMBER OF BIDDERS
    # plt.figure(3)
    # plt.plot(num_bidders, revenue_per_m)
    # plt.title("Number of Bidders vs. Average Revenue")
    # plt.xlabel("Number of Bidders")
    # plt.ylabel("Average Revenue")
    # plt.savefig('Part1_num_bidders_vs_revenue.png')

    # ## VARIATION ON DISTRIBUTION
    # m = 2
    # low = 0
    # step = 0.01
    # n = 100
    # h_values = np.arange(1, 50)
    # revenue_per_h = []
    # regret_per_h = []
    # for h in range(1, 50):
    #     payoff_matrix = generate_action_space(low, h, step)
    #     epsilon = theo_opt_epsilon(len(payoff_matrix), n)
    #     actions_chosen, payoff_matrix, average_revenue, all_revenues = exponential_weights(payoff_matrix, epsilon, low, h, n, m, step)
    #     revenue_per_h.append(average_revenue)
    #     regret_per_h.append(calculate_regret(h, h/2, m, average_revenue)/h)
    #     print('done with round h', h)
    
    # plt.figure(1)
    # plt.plot(h_values, regret_per_h)
    # plt.title("Regret for Different Distributions")
    # plt.xlabel("h value in U[0,h] distribution")
    # plt.ylabel("Regret")
    # plt.savefig('Part1_dist_vs_regret.png')

    # VARIATION IN DIST: QUADRATIC DISTRIBUTION
    # payoff_matrix = generate_action_space(0, 1, 0.01)
    # epsilon = theo_opt_epsilon(len(payoff_matrix), 5000)
    # actions_chosen, payoff_matrix, average_revenue, all_revenues = exponential_weights_quadratic(payoff_matrix, epsilon, 0, 1, 5000, 2, 0.01)    
    # print(actions_chosen)

    # l = [math.sqrt(3)/3] * 5000
    # plt.figure(1)
    # plt.plot(np.arange(n), actions_chosen, label="EW")
    # plt.plot(np.arange(n), l, label="Optimal")
    # plt.legend(loc="upper right")
    # plt.title("Reserve Price Chosen Overtime for Quadratic cdf")
    # plt.ylabel("reserve price")
    # plt.xlabel("round")
    # plt.savefig('Part1_actions_of_quadratic.png')

