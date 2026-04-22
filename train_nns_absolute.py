
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("Using GPU", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print("Using CPU")

def train_or_load_model(
    data_filename,
    test_filename,
    model_dir="models/nns_absolute",
    prediction_dir="predictions/nns_absolute",
    epochs=1000,
    force_retrain=False,
    scaler="scaler_absolute.pkl",
    mah_noise=0.0
):
    """
    Trains or loads an NN model on the data in `data_filename`,
    then evaluates it on the data in `test_filename`. The model filename
    is derived from `data_filename` so each (n, seed) combination is unique.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    # Build a model filename from the training CSV
    # e.g. if data_filename is "rss_n=50_noise=0.0_seed=2.csv",
    # this becomes "full_model_rss_n=50_noise=0.0_seed=2.h5"
    base_model_name = os.path.basename(data_filename).replace('.csv', '')
    model_filename = os.path.join(model_dir, f"full_model_{base_model_name}.h5")

    # 1) Load training data
    data = pd.read_csv(data_filename)
    output_columns = ["X", "Y", "Z"]
    XYZ_train = data[output_columns]
    RSS_train = data.drop(columns=output_columns)

    # 2) Load or train
    if os.path.exists(model_filename) and not force_retrain:
        print(f"Model '{model_filename}' found. Loading instead of retraining.")
        model = load_model(model_filename)
    else:
        print(f"No model found for '{model_filename}' or force_retrain=True.")
        print("Defining and training a new model...")

        model = Sequential([
            Dense(15, input_shape=(RSS_train.shape[1],), activation="relu"),
            Dense(15, activation="relu"),
            Dense(XYZ_train.shape[1])  # X, Y, Z
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Train
        model.fit(
            RSS_train, XYZ_train,
            epochs=epochs,
            batch_size=250,
            verbose=1
        )
        model.save(model_filename)

    # 3) Predict on test data
    data_test = pd.read_csv(test_filename)
    RSS_test = data_test.drop(columns=output_columns)
    predictions = model.predict(RSS_test)

    # 4) Descale predictions
    with open(scaler, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    descaled = loaded_scaler.inverse_transform(
        np.hstack([np.zeros_like(RSS_test), predictions])
    )[:, -3:]  # keep only X, Y, Z

    # 5) Save predictions with the same base name + test_noise
    # e.g. "full_model_rss_n=50_noise=0.0_seed=2_test_noise=0.0.csv"
    base_name_no_ext = os.path.splitext(os.path.basename(model_filename))[0]
    pred_filename = f"{base_name_no_ext}_test_noise={mah_noise}.csv"
    prediction_filepath = os.path.join(prediction_dir, pred_filename)

    pd.DataFrame(descaled, columns=output_columns).to_csv(
        prediction_filepath, index=False
    )
    print(f"Predictions saved to: {prediction_filepath}")

    # 6) Evaluate final training loss
    final_loss = model.evaluate(RSS_train, XYZ_train, verbose=0)
    print(f"Final loss on training data for {base_model_name}: {final_loss}\n")

    return model

# ----------------------------------------------------------------------
# Main loop over (n, seed) and test noise
# ----------------------------------------------------------------------
sample_sizes = range(50, 1501, 50)
seeds = range(10)
noise_levels = [0.0]

for n in sample_sizes:
    for seed in seeds:
        # This file must be named with the seed in it:
        # e.g. 3D-Data/Measured_Normalized/rss_n=50_noise=0.0_seed=2.csv
        data_filename = f"3D-Data/Measured_Normalized/rss_n={n}_noise=0.0_seed={seed}.csv"

        for noise_level in noise_levels:
            test_filename = f"3D-Data/Measured_Normalized/gnd_n=50_noise={noise_level}.csv"
            train_or_load_model(
                data_filename=data_filename,
                test_filename=test_filename,
                force_retrain=False,
                epochs=10000,
                mah_noise=noise_level,
                scaler="scaler_absolute.pkl"
            )
