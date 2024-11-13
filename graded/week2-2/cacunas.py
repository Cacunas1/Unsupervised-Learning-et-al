import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import public_tests as tst
import recsysNN_utils as utl
import tabulate
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras


@keras.utils.register_keras_serializable("CacunasSolutions")
class L2NormLayer(tf.keras.Layer):
    def call(self, x):
        return tf.linalg.l2_normalize(x, axis=1)


def get_model(
    num_user_features: int, num_item_features: int, debug: bool = False
) -> tuple[keras.Model, keras.Model, keras.Model]:
    unq_c1_usr = os.path.join(os.getcwd(), "models", "unq_c1_usr.keras")
    unq_c1_itm = os.path.join(os.getcwd(), "models", "unq_c1_itm.keras")
    unq_c1_mod = os.path.join(os.getcwd(), "models", "unq_c1_mod.keras")
    user_NN: Optional[keras.Model]
    item_NN: Optional[keras.Model]
    model: Optional[keras.Model]

    if (
        os.path.isfile(unq_c1_usr)
        and os.path.isfile(unq_c1_itm)
        and os.path.isfile(unq_c1_mod)
    ):
        user_NN = keras.models.load_model(unq_c1_usr)
        item_NN = keras.models.load_model(unq_c1_itm)
        model = keras.models.load_model(unq_c1_mod)
    else:
        num_outputs: int = 32
        tf.random.set_seed(1)

        user_NN = tf.keras.models.Sequential(
            [
                ### START CODE HERE ###
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(num_outputs, activation="linear"),
                ### END CODE HERE ###
            ]
        )

        item_NN = tf.keras.models.Sequential(
            [
                ### START CODE HERE ###
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(num_outputs, activation="linear"),
                ### END CODE HERE ###
            ]
        )

        # create the user input and point to the base network
        input_user: tf.keras.Layer = tf.keras.Input(shape=(num_user_features,))
        vu: tf.keras.Layer = user_NN(input_user)

        # vu = tf.linalg.l2_normalize(vu, axis=1)
        vu = L2NormLayer()(vu)

        # create the item input and point to the base network
        input_item = keras.layers.Input(shape=(num_item_features,))
        vm: Layer = item_NN(input_item)
        # vm = tf.linalg.l2_normalize(vm, axis=1)
        vm = L2NormLayer()(vm)

        # compute the dot product of the two vectors vu and vm
        output: Layer = keras.layers.Dot(axes=1)([vu, vm])

        # specify the inputs and output of the model
        model = keras.Model([input_user, input_item], output)

        if debug:
            model.summary()

        user_NN.save(unq_c1_usr)
        item_NN.save(unq_c1_itm)
        model.save(unq_c1_mod)

    if model is None:
        raise ValueError("Model not properly initialized")
    if user_NN is None:
        raise ValueError("User NN not properly initialized")
    if item_NN is None:
        raise ValueError("Item NN not properly initialized")

    return model, user_NN, item_NN


# GRADED_FUNCTION: sq_dist
# UNQ_C2
def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    ### START CODE HERE ###
    d = np.dot(a - b, a - b)
    ### END CODE HERE ###
    return d


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

    print("-" * 80)
    print("\tPart 1: Neural Network for content-based filtering")

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
    print(
        f"movie/item training data shape: ({item_train.shape[0]:_}, {item_train.shape[1]:_})"
    )
    print(
        f"movie/item test data shape: ({item_test.shape[0]:_}, {item_test.shape[1]:_})"
    )

    # The scaled, shuffled data now has a mean of zero.
    utl.pprint_train(user_train, user_features, uvs, u_s, maxcount=5)

    model: keras.Model
    user_NN: keras.Model
    item_NN: keras.Model
    model, user_NN, item_NN = get_model(num_item_features, num_user_features, True)

    tst.test_tower(user_NN)
    tst.test_tower(item_NN)

    tf.random.set_seed(1)

    cost_fn = tf.keras.losses.MeanSquaredError()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss=cost_fn)

    model_fit_fp = os.path.join(os.getcwd(), "models", "model_fit.keras")

    if os.path.isfile(model_fit_fp):
        model = tf.keras.load_model(model_fit_fp)
    else:
        model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
        model.save(model_fit_fp)

    model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

    print("-" * 80)
    print("\tPart 2: Predictions -Finding Similar Items")

    new_user_id: int = 5000
    new_rating_ave: int = 0.0
    new_action: float = 0.0
    new_adventure: float = 5.0
    new_animation: float = 0.0
    new_childrens: float = 0.0
    new_comedy: float = 0.0
    new_crime: float = 0.0
    new_documentary: float = 0.0
    new_drama: float = 0.0
    new_fantasy: float = 5.0
    new_horror: float = 0.0
    new_mystery: float = 0.0
    new_romance: float = 0.0
    new_scifi: float = 0.0
    new_thriller: float = 0.0
    new_rating_count: int = 3

    user_vec: np.ndarray = np.array(
        [
            [
                new_user_id,
                new_rating_count,
                new_rating_ave,
                new_action,
                new_adventure,
                new_animation,
                new_childrens,
                new_comedy,
                new_crime,
                new_documentary,
                new_drama,
                new_fantasy,
                new_horror,
                new_mystery,
                new_romance,
                new_scifi,
                new_thriller,
            ]
        ]
    )

    # generate and replicate the user vector to match the number movies in the data set.
    user_vecs = utl.gen_user_vecs(user_vec, len(item_vecs))

    # scale our user and item vectors
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)

    # make a prediction
    y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

    # unscale y prediction
    y_pu = scalerTarget.inverse_transform(y_p)

    # sort the results, highest prediction first
    sorted_index = (
        np.argsort(-y_pu, axis=0).reshape(-1).tolist()
    )  # negate to get largest rating first
    sorted_ypu = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]  # using unscaled vectors for display

    utl.print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount=10)

    uid: int = 2
    # form a set of user vectors. This is the same vector, transformed and repeated.
    user_vecs, y_vecs = utl.get_user_vecs(
        uid, user_train_unscaled, item_vecs, user_to_genre
    )

    # scale our user and item vectors
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)

    # make a prediction
    y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

    # unscale y prediction
    y_pu = scalerTarget.inverse_transform(y_p)

    # sort the results, highest prediction first
    sorted_index = (
        np.argsort(-y_pu, axis=0).reshape(-1).tolist()
    )  # negate to get largest rating first
    sorted_ypu = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]  # using unscaled vectors for display
    sorted_user = user_vecs[sorted_index]
    sorted_y = y_vecs[sorted_index]

    # print sorted predictions for movies rated by the user
    utl.print_existing_user(
        sorted_ypu,
        sorted_y.reshape(-1, 1),
        sorted_user,
        sorted_items,
        ivs,
        uvs,
        movie_dict,
        maxcount=50,
    )

    a1 = np.array([1.0, 2.0, 3.0])
    b1 = np.array([1.0, 2.0, 3.0])
    a2 = np.array([1.1, 2.1, 3.1])
    b2 = np.array([1.0, 2.0, 3.0])
    a3 = np.array([0, 1, 0])
    b3 = np.array([1, 0, 0])
    print(f"squared distance between a1 and b1: {sq_dist(a1, b1):0.3f}")
    print(f"squared distance between a2 and b2: {sq_dist(a2, b2):0.3f}")
    print(f"squared distance between a3 and b3: {sq_dist(a3, b3):0.3f}")

    # Public tests
    tst.test_sq_dist(sq_dist)

    input_item_m = tf.keras.layers.Input(shape=(num_item_features))  # input layer
    vm_m = item_NN(input_item_m)  # use the trained item_NN
    vm_m = tf.linalg.l2_normalize(
        vm_m, axis=1
    )  # incorporate normalization as was done in the original model
    model_m = tf.keras.Model(input_item_m, vm_m)
    model_m.summary()

    scaled_item_vecs = scalerItem.transform(item_vecs)
    vms = model_m.predict(scaled_item_vecs[:, i_s:])
    print(f"size of all predicted movie feature vectors: {vms.shape}")

    count = 50  # number of movies to display
    dim = len(vms)
    dist = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            dist[i, j] = sq_dist(vms[i, :], vms[j, :])

    m_dist = np.ma.masked_array(
        dist, mask=np.identity(dist.shape[0])
    )  # mask the diagonal

    disp = [["movie1", "genres", "movie2", "genres"]]
    for i in range(count):
        min_idx = np.argmin(m_dist[i])
        movie1_id = int(item_vecs[i, 0])
        movie2_id = int(item_vecs[min_idx, 0])
        disp.append(
            [
                movie_dict[movie1_id]["title"],
                movie_dict[movie1_id]["genres"],
                movie_dict[movie2_id]["title"],
                movie_dict[movie1_id]["genres"],
            ]
        )
    table = tabulate.tabulate(disp, tablefmt="html", headers="firstrow")
    print(table)

    print("=" * 80)


if __name__ == "__main__":
    main()
