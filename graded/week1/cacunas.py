import numpy as np
import public_tests as tst
import utils as utl


def find_closest_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids
    """
    # You need to return the following variables correctly
    idx: np.ndarray = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    for i, x in enumerate(X):
        idx[i] = np.argmin([np.dot(x - mu, x - mu) for mu in centroids])
    ### END CODE HERE ###

    return idx


# UNQ_C2
# GRADED FUNCTION: compute_centroids


def compute_centroids(X: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    # Useful variables
    n: int
    n = X.shape[-1]

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    ### START CODE HERE ###
    for k in range(K):
        points = [x for i, x in enumerate(X) if idx[i] == k]
        centroids[k] = sum(points) / len(points)
    ### END CODE HERE ##

    return centroids


def main():
    print(np.version)

    X = utl.load_data()

    # The code below prints the first five elements in the variable `X` and the dimensions of the variable
    print("First five elements of X are:\n", X[:5])
    print("The shape of X is:", X.shape)

    # Select an initial set of centroids (3 Centroids)
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Find closest centroids using initial_centroids
    idx = find_closest_centroids(X, initial_centroids)

    print("=" * 80)
    print("\tGraded Part 1")

    # Print closest centroids for the first three elements
    print("First three elements in idx are:", idx[:3])

    # UNIT TEST
    tst.find_closest_centroids_test(find_closest_centroids)

    print("=" * 80)
    print("\tGraded Part 2")

    K: int = 3
    centroids = compute_centroids(X, idx, K)

    print("The centroids are:", centroids)

    # UNIT TEST
    tst.compute_centroids_test(compute_centroids)


if __name__ == "__main__":
    main()
