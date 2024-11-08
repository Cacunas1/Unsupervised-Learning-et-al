import numpy as np
import tensorflow as tf
from tensorflow import keras

import recsys_utils as utl

# GRADED FUNCTION: cofi_cost_func
# UNQ_C1


def cofi_cost_func(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    lambda_: float,
) -> float:
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies, num_features)): matrix of item features
      W (ndarray (num_users, num_features)) : matrix of user parameters
      b (ndarray (1, num_users)             : vector of user parameters
      Y (ndarray (num_movies, num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies, num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    # nm: int
    # nu: int
    # nm, nu = Y.shape
    J: float = 0.0
    ### START CODE HERE ###
    Y_predicted: np.ndarray = R * (X @ W.T + b)
    Error: np.ndarray = Y_predicted - Y
    Regularization: float = np.trace(W.T @ W) + np.trace(X.T @ X)
    J_aux = 0.5 * np.trace(Error.T @ Error) + 0.5 * lambda_ * Regularization
    J = J_aux
    ### END CODE HERE ###

    return J


def main():
    print("=" * 80)
    print("Programming Assingment: Week 2")
    print("Collaborative Filtering for Recommender Systems")

    print("-" * 80)
    print("\tPart 0: Prelude")
    print(f"Numpy version: {np.version.version}")
    print(f"TensorFlow version: {tf.version.VERSION}")
    print(f"Keras version: {keras.version()}")

    X: np.ndarray
    W: np.ndarray
    b: np.ndarray
    Y: np.ndarray
    R: np.ndarray
    num_movies: int
    num_features: int
    num_users: int
    X, W, b, num_movies, num_features, num_users = utl.load_precalc_params_small()
    Y, R = utl.load_ratings_small()

    print("Y", Y.shape, "R", R.shape)
    print("X", X.shape)
    print("W", W.shape)
    print("b", b.shape)
    print(f"No. of features: {num_features:_}")
    print(f"No. of movies: {num_movies:_}")
    print(f"No. of users: {num_users:_}")

    #  From the matrix, we can compute statistics like average rating.
    tsmean = np.mean(Y[0, R[0, :].astype(bool)])
    print(f"Average rating for movie 1 : {tsmean:0.3f} / 5")

    print("-" * 80)
    print("\t Part 1: Collaborative Filtering Cost Function")

    num_users_r: int = 4
    num_movies_r: int = 5
    num_features_r: int = 3

    X_r: np.ndarray = X[:num_movies_r, :num_features_r]
    W_r: np.ndarray = W[:num_users_r, :num_features_r]
    b_r: np.ndarray = b[0, :num_users_r].reshape(1, -1)
    Y_r: np.ndarray = Y[:num_movies_r, :num_users_r]
    R_r: np.ndarray = R[:num_movies_r, :num_users_r]

    # Evaluate cost function
    J: float = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0)
    print(f"Cost: {J:0.2f}")

    # + [markdown] id="xGznmQ91odYL"
    # **Expected Output (lambda = 0)**:
    # $13.67$.

    # + deletable=false editable=false
    # Evaluate cost function with regularization
    J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5)
    print(f"Cost (with regularization): {J:0.2f}")


    print("=" * 80)


if __name__ == "__main__":
    main()
