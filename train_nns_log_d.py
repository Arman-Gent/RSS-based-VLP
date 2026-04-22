import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Define the trilateration functions (same as before)
def mse_error(pos, beacon_coords, distances):
    predicted_distances = np.linalg.norm(beacon_coords - pos, axis=1)
    mse = np.mean((predicted_distances - distances)**2)
    return mse

def trilaterate_mse(distances, beacon_coords=np.array([[0,0,5],[-2,-2,5],[-2,2,5],[2,-2,5],[2,2,5]])):
    initial_guess = np.mean(beacon_coords, axis=0)
    bounds = [(None, None), (None, None), (None, 5)]
    result = minimize(mse_error, initial_guess, args=(beacon_coords, distances), method="L-BFGS-B", bounds=bounds)
    if result.success:
        return result.x
    else:
        raise ValueError("Trilateration failed: " + result.message)

# Configure GPU (if available)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
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
    model_dir="models/nns_log_d",
    prediction_dir="predictions/nns_log_d",
    epochs=1000,
    force_retrain=False,
    scaler="scaler_log_d.pkl",
    mah_noise=0.0
):
    """
    Trains or loads a neural network on log-d data from data_filename,
    then makes predictions on test_filename.
    
    The training CSV should have output columns: ["D0", "D1", "D2", "D3", "D4"].
    The model filename is derived from data_filename (which includes the seed),
    so each (n, seed) combination is unique.
    
    After prediction, the outputs are descaled, exponentiated (to convert log-distances
    into linear distances), and then trilaterated using a fixed beacon configuration.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Build a unique model filename (e.g., full_model_rss_n=100_noise=0.0_seed=2.h5)
    base_model_name = os.path.basename(data_filename).replace('.csv', '')
    model_filename = os.path.join(model_dir, f"full_model_{base_model_name}.h5")
    
    # 1) Load training data
    data = pd.read_csv(data_filename)
    output_columns = ["D0", "D1", "D2", "D3", "D4"]
    D_train = data[output_columns]
    RSS_train = data.drop(columns=output_columns)
    
    # 2) Load or train a new model
    if os.path.exists(model_filename) and not force_retrain:
        print(f"Model '{model_filename}' found. Loading...")
        model = load_model(model_filename)
    else:
        print(f"Training new model for {base_model_name}...")
        model = Sequential([
            Dense(15, input_shape=(RSS_train.shape[1],), activation="relu"),
            Dense(15, activation="relu"),
            Dense(D_train.shape[1])  # Predicting D0...D4
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(RSS_train, D_train, epochs=epochs, batch_size=250, verbose=1)
        model.save(model_filename)
    
    # 3) Predict on test data
    data_test = pd.read_csv(test_filename)
    # For test data, drop the same output columns ("D0".. "D4")
    RSS_test = data_test.drop(columns=output_columns)
    predictions = model.predict(RSS_test)
    
    # 4) Descale predictions
    with open(scaler, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    # Assume scaler was fitted on data with shape (..., num_features + 5)
    # We prepend zeros so that the inverse transform yields meaningful D0..D4 values
    log_distances = loaded_scaler.inverse_transform(
        np.hstack([np.zeros_like(RSS_test), predictions])
    )[:, -5:]  # last 5 columns correspond to D0..D4 (still in log scale)
    # Exponentiate to get linear distances
    linear_distances = np.exp(log_distances)
    
    print("First 5 sets of linear distances:\n", linear_distances[:5])
    
    # 5) Trilaterate each row to obtain (X, Y, Z)
    beacon_coords = np.array([
        [0, 0, 5],
        [-2, -2, 5],
        [-2,  2, 5],
        [ 2, -2, 5],
        [ 2,  2, 5]
    ])
    xyz_coordinates = []
    for distance_set in linear_distances:
        xyz = trilaterate_mse(distance_set, beacon_coords=beacon_coords)
        xyz_coordinates.append(xyz)
    
    # 6) Save predictions
    # Create prediction filename that includes seed and test noise info, e.g.
    # "full_model_rss_n=100_noise=0.0_seed=2_test_noise=0.0.csv"
    base_name_no_ext = os.path.splitext(os.path.basename(model_filename))[0]
    pred_filename = f"{base_name_no_ext}_test_noise={mah_noise}.csv"
    prediction_filepath = os.path.join(prediction_dir, pred_filename)
    pd.DataFrame(xyz_coordinates, columns=["X", "Y", "Z"]).to_csv(prediction_filepath, index=False)
    print(f"Predictions saved to: {prediction_filepath}")
    
    # 7) Evaluate final training loss
    final_loss = model.evaluate(RSS_train, D_train, verbose=0)
    print(f"Final training loss for {base_model_name}: {final_loss}\n")
    
    return model

# ----------------------------------------------------------------------
# Main loop: loop over sample sizes, seeds, and test noise levels
# ----------------------------------------------------------------------
sample_sizes = range(50, 1501, 50)  # e.g., 50, 100, ..., 1500
seeds = range(10)
noise_levels = [0.0]

for n in sample_sizes:
    for seed in seeds:
        # Training data filename must include the seed:
        # e.g., "3D-Data/Measured_log_d_Normalized/rss_n=100_noise=0.0_seed=2.csv"
        data_filename = f"3D-Data/Measured_log_d_Normalized/rss_n={n}_noise=0.0_seed={seed}.csv"
        for noise_level in noise_levels:
            test_filename = f"3D-Data/Measured_log_d_Normalized/gnd_n=50_noise={noise_level}.csv"
            train_or_load_model(
                data_filename=data_filename,
                test_filename=test_filename,
                force_retrain=False,
                epochs=20000,
                mah_noise=noise_level,
                scaler="scaler_log_d.pkl"
            )

