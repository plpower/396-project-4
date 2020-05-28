import csv
import numpy as np
from scipy.stats import geom
from scipy.spatial import distance
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    # print('point1', point1, 'point2', point2)
    # print(distance.euclidean(point1, point2))
    return distance.euclidean(point1, point2)
    # return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def nearest_neighbor(test_bids, train_bids, labels, m):
    # take in a new point
    # find the point that its closest to
    # return that label as a prediction
    revenue = 0
    predictions = []
    # loop though test bids
    for i in range(m):
        closest_distance = 100
        closest_label = None
        test_bid = [test_bids[0][i], test_bids[1][i]]
        # print('test bid', test_bid)
        # loop through train bids
        for j in range(m):
            train_bid = [train_bids[0][j], train_bids[1][j]]
            dist = abs(test_bid[0] - train_bid[0]) + abs(test_bid[1] - train_bid[1])
            if dist < closest_distance:
                closest_distance = dist 
                closest_label = labels[j]
                closest_bid = train_bid
        predictions.append(closest_label)
        if closest_label == 1:
            revenue += test_bids[0][i] + test_bids[1][i] - 1
        # print(revenue)
    return predictions, revenue

def generate_bids(high, m):
    ### generate m bids drawn from uniform distribution
    x = np.random.uniform(0, high, m)
    y = np.random.uniform(0, high, m)

    bids = [x, y]
    return bids

def run_it():
    high = 1
    m = 100

    train_labels = []
    test_labels = []

    train_bids = generate_bids(high, m)
    total_rev = 0

    for i in range(m):
        if train_bids[0][i] + train_bids[1][i] >= 1:
            total_rev += train_bids[0][i] + train_bids[1][i] - 1
            train_labels.append(1)
        else:
            train_labels.append(0)

    expected_rev = total_rev / m
    print("EXPECTED REV", expected_rev)
    test_bids = generate_bids(high, m)

    for i in range(m):
        if test_bids[0][i] + test_bids[1][i] >= 1:
            test_labels.append(1)
        else:
            test_labels.append(0)

    
    predictions, test_revenue = nearest_neighbor(test_bids, train_bids, train_labels, m)
    test_rev = test_revenue / m
    
    accuracy = 0
    correct_labels_x = []
    correct_labels_y = []
    incorrect_labels_x = []
    incorrect_labels_y = []
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            accuracy += 1
            correct_labels_x.append(test_bids[0][i])
            correct_labels_y.append(test_bids[1][i])
        else:
            incorrect_labels_x.append(test_bids[0][i])
            incorrect_labels_y.append(test_bids[1][i])

    accuracy = accuracy/len(predictions)
    print("ACURACY", accuracy)

    return accuracy, expected_rev, test_rev

if __name__ == "__main__":
    a = []
    er = []
    tr = []
    for _ in range(1000):
        accuracy, expected_rev, test_rev = run_it()
        a.append(accuracy)
        er.append(expected_rev)
        tr.append(test_rev)
    
    plt.figure(1)
    plt.scatter(np.arange(1000), er, label="expected", color='red')
    plt.scatter(np.arange(1000), tr, label="test", color='blue')
    plt.legend(loc="lower left")
    plt.title("Expected and Actual Revenue Over 1000 Simulations")
    plt.ylabel("revenue")
    plt.xlabel("simulation number")
    plt.savefig('Part3_REVS.png')

    high = 1
    m = 100

    train_labels = []
    test_labels = []

    train_bids = generate_bids(high, m)
    total_rev = 0

    for i in range(m):
        if train_bids[0][i] + train_bids[1][i] >= 1:
            total_rev += train_bids[0][i] + train_bids[1][i] - 1
            train_labels.append(1)
        else:
            train_labels.append(0)

    expected_rev = total_rev / m
    test_bids = generate_bids(high, m)

    for i in range(m):
        if test_bids[0][i] + test_bids[1][i] >= 1:
            test_labels.append(1)
        else:
            test_labels.append(0)

    
    predictions, test_revenue = nearest_neighbor(test_bids, train_bids, train_labels, m)
    test_rev = test_revenue / m

    accuracy = 0
    correct_labels_x = []
    correct_labels_y = []
    incorrect_labels_x = []
    incorrect_labels_y = []
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            accuracy += 1
            correct_labels_x.append(test_bids[0][i])
            correct_labels_y.append(test_bids[1][i])
        else:
            incorrect_labels_x.append(test_bids[0][i])
            incorrect_labels_y.append(test_bids[1][i])

    accuracy = accuracy/len(predictions)
    print("ACURACY", accuracy)

    # add graph of value space 
    intro_idx = np.where(np.asarray(train_labels) == 1)[0]
    intro_x = train_bids[0][intro_idx]
    intro_y = train_bids[1][intro_idx]
    no_intro_idx = np.where(np.asarray(train_labels) == 0)[0]
    no_intro_x = train_bids[0][no_intro_idx]
    no_intro_y = train_bids[1][no_intro_idx]

    plt.figure(4)
    plt.scatter(intro_x, intro_y, label="introduce", color='red')
    plt.scatter(no_intro_x, no_intro_y, label="do not introduce", color='blue')
    plt.legend(loc="lower left")
    plt.title("Training Dataset Value Space for Selling Introductions")
    plt.ylabel("Employer")
    plt.xlabel("Employee")
    plt.savefig('Part3_Training_Set_Value_Space.png')

    plt.figure(2)
    plt.scatter(correct_labels_x, correct_labels_y, label="correct prediction", color='blue')
    plt.scatter(incorrect_labels_x,incorrect_labels_y, label="incorrect prediction", color='green')
    plt.legend(loc="lower left")
    plt.title("Testing Set Value Space for Selling Introductions")
    plt.ylabel("Employer")
    plt.xlabel("Employee")
    plt.savefig('Part3_Testing_Set_Value_Space.png')

    plt.figure(3)
    plt.scatter(intro_x, intro_y, label="correct: intro", color='red')
    plt.scatter(no_intro_x, no_intro_y, label="correct: no intro", color='blue')
    plt.scatter(incorrect_labels_x,incorrect_labels_y, label="incorrect prediction", color='green')
    plt.legend(loc="lower left")
    plt.title("Incorrect and Correctly Predicted Introductions")
    plt.ylabel("Employer")
    plt.xlabel("Employee")
    plt.savefig('Part3_Incorrect_Labels_Value_Space.png')
