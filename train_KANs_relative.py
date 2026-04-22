
import numpy as np
import pandas as pd
import torch
import os
import pickle
from kan import KAN
from sklearn.preprocessing import StandardScaler

# Disable dynamo to avoid import errors
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Device setup (CPU only)
device = torch.device('cpu')

# Function to train KAN models on provided datasets
def train_kan_model(
    data_filename,
    test_filename,
    model_dir="models/KANs_relative",
    prediction_dir="predictions/KANs_relative",
    scaler_path="scaler_relative.pkl",
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

    # Load data
    df_train = pd.read_csv(data_filename)
    df_test = pd.read_csv(test_filename)

    # Dynamically set input/output columns
    output_columns = ['X', 'Y', 'Z']
    input_columns = [col for col in df_train.columns if col not in output_columns]

    X_train = df_train[input_columns].values.astype(np.float32)
    Y_train = df_train[output_columns].values.astype(np.float32)

    X_test = df_test[input_columns].values.astype(np.float32)

    # Load scaler
    with open(scaler_path, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    # Dataset tensors (only for training phase)
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

    train_losses, test_losses = [], []

    for i, grid_size in enumerate(grids):
        if i == 0:
            model = KAN(width=model_width, grid=grid_size, k=k, seed=1,
                        device=device, ckpt_path=ckpt_path, auto_save=False)
        else:
            model = model.refine(grid_size)

        model.auto_save = True
        results = model.fit(dataset, opt="Adam", steps=steps, lr=0.001, batch=batch_size)

        train_losses += results['train_loss']
        test_losses += results['test_loss']

        model.auto_save = False

    print(f"✅ Model trained and saved for {base_model_name}.")

    # Load final trained model
    final_version_prefix = "0.3_state"
    checkpoint_file = os.path.join(ckpt_path, final_version_prefix)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.eval()

    # Predict and rescale predictions on the TEST data (gnd)
    test_tensor = torch.tensor(X_test).to(device)
    predictions = model(test_tensor).detach().cpu().numpy()
    descaled_predictions = loaded_scaler.inverse_transform(
        np.hstack([np.zeros((predictions.shape[0], len(input_columns))), predictions])
    )[:, -3:]

    # Save predictions
    predictions_df = pd.DataFrame(descaled_predictions, columns=output_columns)
    prediction_filepath = os.path.join(prediction_dir, f"KAN_predictions_{base_model_name}.csv")
    predictions_df.to_csv(prediction_filepath, index=False)

    print(f"✅ Predictions saved for {base_model_name} at {prediction_filepath}.")

# Main loop to train on multiple datasets
sample_sizes = range(50, 1501, 50)
seeds = range(10)
noise_levels = [0.0]

for n in sample_sizes:
    for seed in seeds:
        data_filename = f"3D-Data/Measured_relative_Normalized/relative_rss_n={n}_noise=0.0_seed={seed}.csv"

        for noise_level in noise_levels:
            test_filename = f"3D-Data/Measured_relative_Normalized/relative_gnd_n=50_noise={noise_level}.csv"
            train_kan_model(
                data_filename=data_filename,
                test_filename=test_filename,
                scaler_path="scaler_relative.pkl"
            )
