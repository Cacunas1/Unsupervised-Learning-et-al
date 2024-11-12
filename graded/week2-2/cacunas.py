from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras

import recsysNN_utils as utl


def main():
    print("=" * 80)
    print("Programming Assingment: Week 2 - 2")
    print("Content-based Filtering for Recommender Systems")

    print("-" * 80)
    print("\tPart 0: Prelude")
    print(f"Numpy version: {np.version.version}")
    print(f"TensorFlow version: {tf.version.VERSION}")
    print(f"Keras version: {keras.version()}")
    pd.set_option("display.precision", 1)


    top10_df: pd.DataFrame = pd.read_csv("./data/content_top10_df.csv")
    bygenre_df: pd.DataFrame = pd.read_csv("./data/content_bygenre_df.csv")
    print("\tTop 10:")
    print(top10_df)
    print("\tBy genre:")
    print(bygenre_df)

    item_train: np.ndarray
    user_train: np.ndarray
    y_train: np.ndarray
    item_features: list
    user_features: list
    item_vecs: np.ndarray
    movie_dict: defaultdict
    user_to_genre: defaultdict

    (
        item_train,
        user_train,
        y_train,
        item_features,
        user_features,
        item_vecs,
        movie_dict,
        user_to_genre,
    ) = utl.load_data()


    print(f"item_train: {type(item_train)}")
    print(f"user_train: {type(user_train)}")
    print(f"y_train: {type(y_train)}")
    print(f"item_features: {type(item_features)}")
    print(f"user_features: {type(user_features)}")
    print(f"item_vecs: {type(item_vecs)}")
    print(f"movie_dict: {type(movie_dict)}")
    print(f"user_to_genre: {type(user_to_genre)}")

    num_user_features = (
        user_train.shape[1] - 3
    )  # remove userid, rating count and ave rating during training
    print(f"num_user_features: {num_user_features}")
    num_item_features = item_train.shape[1] - 1  # remove movie id at train time
    print(f"num_item_features: {num_item_features}")
    uvs = 3  # user genre vector start
    ivs = 3  # item genre vector start
    print(f"ivs: {ivs}")
    u_s = 3  # start of columns to use in training, user
    i_s = 1  # start of columns to use in training, items
    print(f"i_s: {i_s}")
    print(f"Number of training vectors: {len(item_train):_}")

    # Let's look at the first few entries in the user training array.
    utl.pprint_train(user_train, user_features, uvs, u_s, maxcount=5)

    # scale training data
    item_train_unscaled = item_train
    user_train_unscaled = user_train
    y_train_unscaled: np.ndarray = y_train
    print(f"y_train_unscaled: {y_train_unscaled.shape}")

    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    item_train = scalerItem.transform(item_train)

    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    user_train = scalerUser.transform(user_train)

    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(y_train.reshape(-1, 1))
    y_train = scalerTarget.transform(y_train.reshape(-1, 1))
    # ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

    print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
    print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))

    # To allow us to evaluate the results, we will split the data into training
    # and test sets as was discussed in Course 2, Week 3. Here we will use
    # [sklean train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    # to split and shuffle the data. Note that setting the initial random
    # state to the same value ensures item, user, and y are shuffled
    # identically.

    item_train, item_test = train_test_split(
        item_train, train_size=0.80, shuffle=True, random_state=1
    )
    user_train, user_test = train_test_split(
        user_train, train_size=0.80, shuffle=True, random_state=1
    )
    y_train, y_test = train_test_split(
        y_train, train_size=0.80, shuffle=True, random_state=1
    )
    print(f"movie/item training data shape: ({item_train.shape[0]:_}, {item_train.shape[1]:_})")
    print(f"movie/item test data shape: ({item_test.shape[0]:_}, {item_test.shape[1]:_})")

    # The scaled, shuffled data now has a mean of zero.
    utl.pprint_train(user_train, user_features, uvs, u_s, maxcount=5)


if __name__ == "__main__":
    main()
