import csv
import numpy as np
from scipy.stats import geom
import math
import matplotlib.pyplot as plt

def exponential_weights(payoff_matrix_p1, epsilon, h, r):
    mask = payoff_matrix_p1[0]
    no_mask = payoff_matrix_p1[1]
    stay_home = payoff_matrix_p1[2]

    # calculate the probability of chosing every action in round r
    probabilities, round_actions = get_probabilities(
        r, epsilon, h, payoff_matrix_p1)

    # chose action for round r with probabilities
    action = np.random.choice(round_actions, p=probabilities)

    # regret is 2 * h sqrt(ln(k) / h)
    # learning rate is sqrt(ln(k) / h)

    return action


def get_probabilities(r, e, h, payoff_matrix):
    hindsight_payoffs = []
    total_payoff = 0
    probabilities = []
    curr_action = []

    if r == 0:
        for action in range(len(payoff_matrix)):
            # print('action index', action)
            # action_payoff = payoff_matrix[action][0]
            curr_action.append(action)
            hindsight_payoff = 0
            hindsight_payoffs.append(hindsight_payoff)
        return [(1/3), (1/3), (1/3)], curr_action
    else:
        for action in range(len(payoff_matrix)):
            # action_payoff = payoff_matrix[action][r]
            curr_action.append(action)
            hindsight_payoff = sum(payoff_matrix[action][:r])
            hindsight_payoffs.append((1+e) ** (hindsight_payoff/h))
        total_payoff = sum(hindsight_payoffs)

        for action in range(len(payoff_matrix)):
            probabilities.append(hindsight_payoffs[action]/total_payoff)

        return probabilities, curr_action


def follow_perturbed_leader(payoff_matrix_p2, epsilon, r, hallucinations):
    # get values for each action at every round
    mask = payoff_matrix_p2[0]
    no_mask = payoff_matrix_p2[1]
    stay_home = payoff_matrix_p2[2]

    # Choose the payoff of the BIH action - take into account the round 0 hallucinations

    bih1, bih2, bih3 = best_in_hindsight(
        mask, no_mask, stay_home, r, hallucinations)
    max_action = np.argmax([bih1, bih2, bih3])

    return max_action


def best_in_hindsight(mask, no_mask, stay_home, curr_round, hallucinations):
    # best in hindsight DOES NOT include the current round
    # to find BIH of entire action, all it on curr_round = length of list

    bih1 = sum(mask) + hallucinations[0]
    bih2 = sum(no_mask) + hallucinations[1]
    bih3 = sum(stay_home) + hallucinations[2]

    return bih1, bih2, bih3


def calculate_regret(payoff_matrix, payoff, h):
    # calculate OPT
    action_bihs = []
    for action in payoff_matrix:
        action_bihs.append(sum(payoff_matrix[action]))
    best_bih = max(action_bihs)

    regret = (best_bih - payoff) / len(payoff_matrix[0])

    return regret


def theo_opt_epsilon(k, n):
    epsilon = math.sqrt(np.log(k)/n)

    return epsilon


if __name__ == "__main__":
    print('hi')