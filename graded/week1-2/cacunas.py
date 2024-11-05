import matplotlib.pyplot as plt
import numpy as np
import public_tests as tst
import utils as utl

# UNQ_C1
# GRADED FUNCTION: estimate_gaussian


def estimate_gaussian(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates mean and variance of all features
    in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    ### START CODE HERE ###
    m: int
    m = X.shape[0]

    mu: np.ndarray
    var: np.ndarray

    mu = np.array(sum([x for x in X]) / m)
    var = np.array(sum([(x - mu) ** 2 for x in X])) / m
    ### END CODE HERE ###

    return mu, var


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon: float = 0.0
    best_F1 = 0.0
    F1: float = 0.0

    m: int = y_val.shape[0]

    step_size = (max(p_val) - min(p_val)) / 1_000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        ### START CODE HERE ###
        y_pred: np.ndarray = np.array([1 if p < epsilon else 0 for p in p_val])
        tp: int = 0
        fp: int = 0
        tn: int = 0
        fn: int = 0

        for i in range(m):
            tp += 1 if y_pred[i] == y_val[i] and y_pred[i] == 1 else 0
            fp += 1 if y_pred[i] != y_val[i] and y_pred[i] == 1 else 0
            tn += 1 if y_pred[i] == y_val[i] and y_pred[i] == 0 else 0
            fn += 1 if y_pred[i] != y_val[i] and y_pred[i] == 0 else 0

        prec: float = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec: float = tp / (tp + fn) if tp + fn > 0 else 0.0
        F1 = 2 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0
        ### END CODE HERE ###

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1


def main():
    print("=" * 80)
    print("Course 3 - Week 1 - Assignment 2")
    print("-" * 80)
    print("\tPart 0: Intro")
    print(np.version)
    print(plt.__name__)

    print("-" * 80)
    print("\tPart 1: Graded 1: Gaussian Estimate")
    X_train: np.ndarray
    X_val: np.ndarray
    X_train, X_val, y_val = utl.load_data()

    # #### View the variables
    # Let's get more familiar with your dataset.
    # - A good place to start is to just print out each variable and see what it contains.
    #
    # The code below prints the first five elements of each of the variables

    # Display the first five elements of X_train
    print("The first 5 elements of X_train are:\n", X_train[:5])

    # Display the first five elements of X_val
    print("The first 5 elements of X_val are\n", X_val[:5])

    # Display the first five elements of y_val
    print("The first 5 elements of y_val are\n", y_val[:5])

    # #### Check the dimensions of your variables
    #
    # Another useful way to get familiar with your data is to view its dimensions.
    #
    # The code below prints the shape of `X_train`, `X_val` and `y_val`.

    print("The shape of X_train is:", X_train.shape)
    print("The shape of X_val is:", X_val.shape)
    print("The shape of y_val is: ", y_val.shape)

    # #### Visualize your data
    #
    # Before starting on any task, it is often useful to understand the data by visualizing it.
    # - For this dataset, you can use a scatter plot to visualize the data (`X_train`), since it has only two properties to plot (throughput and latency)
    #
    # - Your plot should look similar to the one below
    # Create a scatter plot of the data. To change the markers to blue "x",
    # we used the 'marker' and 'c' parameters
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="x", c="b")

    # Set the title
    plt.title("The first dataset")
    # Set the y-axis label
    plt.ylabel("Throughput (mb/s)")
    # Set the x-axis label
    plt.xlabel("Latency (ms)")
    # Set axis range
    plt.axis((0, 30, 0, 30))
    plt.show()

    # Estimate mean and variance of each feature
    mu, var = estimate_gaussian(X_train)

    print("Mean of each feature:", mu)
    print("Variance of each feature:", var)

    tst.estimate_gaussian_test(estimate_gaussian)

    # Plotting code
    utl.visualize_fit(X_train, mu, var)

    print("-" * 80)
    print("\tPart 2: Graded 2: Selecting the threshold")

    p_val: np.ndarray = utl.multivariate_gaussian(X_val, mu, var)
    epsilon: float
    F1: float
    epsilon, F1 = select_threshold(y_val, p_val)

    print("Best epsilon found using cross-validation: %e" % epsilon)
    print("Best F1 on Cross Validation Set: %f" % F1)

    # UNIT TEST
    tst.select_threshold_test(select_threshold)

    print("=" * 80)


if __name__ == "__main__":
    main()
