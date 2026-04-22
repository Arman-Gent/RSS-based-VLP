
import numpy as np
import pandas as pd
import torch
import os
import pickle
from kan import KAN
from scipy.optimize import minimize

# Disable dynamo to avoid import errors
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Device setup (CPU only)
device = torch.device('cpu')

# Trilateration functions
def mse_error(pos, beacon_coords, distances):
    predicted_distances = np.linalg.norm(beacon_coords - pos, axis=1)
    return np.mean((predicted_distances - distances)**2)

def trilaterate_mse(distances, beacon_coords=np.array([[0,0,5],[-2,-2,5],[-2,2,5],[2,-2,5],[2,2,5]])):
    initial_guess = np.mean(beacon_coords, axis=0)
    bounds = [(None, None), (None, None), (None, 5)]
    result = minimize(mse_error, initial_guess, args=(beacon_coords, distances), method="L-BFGS-B", bounds=bounds)
    if result.success:
        return result.x
    else:
        raise ValueError("Trilateration failed: " + result.message)

# Function to train KAN models on provided datasets
def train_kan_model(
    data_filename,
    test_filename,
    model_dir="models/KANs_log_d",
    prediction_dir="predictions/KANs_log_d",
    scaler_path="scaler_log_d.pkl",
    grids=np.array([3, 10]),
    steps=30000,
    k=3,
    batch_size=128
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    base_model_name = os.path.basename(data_filename).replace('.csv', '')
    ckpt_path = os.path.join(model_dir, f"KAN_model_{base_model_name}")
    os.makedirs(ckpt_path, exist_ok=True)

    # Load training data
    df_train = pd.read_csv(data_filename)

    # Dynamically set input/output columns
    output_columns = ['D0', 'D1', 'D2', 'D3', 'D4']
    input_columns = [col for col in df_train.columns if col not in output_columns]

    X_train = df_train[input_columns].values.astype(np.float32)
    Y_train = df_train[output_columns].values.astype(np.float32)

    # Load scaler
    with open(scaler_path, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    # Training dataset tensors
    dataset = {
        "train_input": torch.tensor(X_train).to(device),
        "train_label": torch.tensor(Y_train).to(device),
        "test_input": torch.tensor(X_train).to(device),  # same as train for internal validation
        "test_label": torch.tensor(Y_train).to(device)
    }

    # Model architecture
    input_dim = len(input_columns)
    output_dim = len(output_columns)
    model_width = [input_dim, 3, output_dim]

    for i, grid_size in enumerate(grids):
        if i == 0:
            model = KAN(width=model_width, grid=grid_size, k=k, seed=1,
                        device=device, ckpt_path=ckpt_path, auto_save=False)
        else:
            model = model.refine(grid_size)

        model.auto_save = True
        model.fit(dataset, opt="Adam", steps=steps, lr=0.001, batch=batch_size)
        model.auto_save = False

    print(f"✅ Model trained and saved for {base_model_name}.")

    # Load final trained model
    final_version_prefix = "0.3_state"
    checkpoint_file = os.path.join(ckpt_path, final_version_prefix)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.eval()

    # Now load and use the dense test data
    df_test = pd.read_csv(test_filename)
    X_test = df_test[input_columns].values.astype(np.float32)
    test_tensor = torch.tensor(X_test).to(device)

    # Predict and rescale predictions on TEST data
    predictions = model(test_tensor).detach().numpy()
    descaled_log_distances = loaded_scaler.inverse_transform(
        np.hstack([np.zeros((predictions.shape[0], len(input_columns))), predictions])
    )[:, -5:]

    linear_distances = np.exp(descaled_log_distances)

    # Trilaterate each row
    beacon_coords = np.array([[0, 0, 5], [-2, -2, 5], [-2, 2, 5], [2, -2, 5], [2, 2, 5]])
    xyz_coordinates = [trilaterate_mse(d, beacon_coords) for d in linear_distances]

    # Save predictions
    predictions_df = pd.DataFrame(xyz_coordinates, columns=["X", "Y", "Z"])
    prediction_filepath = os.path.join(prediction_dir, f"KAN_predictions_{base_model_name}.csv")
    predictions_df.to_csv(prediction_filepath, index=False)

    print(f"✅ Predictions saved for {base_model_name} at {prediction_filepath}.")

# Main loop to train on multiple datasets
sample_sizes = range(850, 999, 50)
seeds = range(10)
noise_levels = [0.0]

for n in sample_sizes:
    for seed in seeds:
        data_filename = f"3D-Data/Measured_log_d_Normalized/rss_n={n}_noise=0.0_seed={seed}.csv"

        for noise_level in noise_levels:
            test_filename = f"3D-Data/Measured_log_d_Normalized/gnd_n=50_noise={noise_level}.csv"
            train_kan_model(
                data_filename=data_filename,
                test_filename=test_filename,
                scaler_path="scaler_log_d.pkl"
            )
