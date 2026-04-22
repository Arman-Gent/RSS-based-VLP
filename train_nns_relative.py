
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# GPU configuration
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
    model_dir="models/nns_relative",
    prediction_dir="predictions/nns_relative",
    epochs=1000,
    force_retrain=False,
    scaler="scaler_relative.pkl",
    mah_noise=0.0
):
    """
    Trains or loads an NN model using relative data from data_filename,
    then makes predictions on test_filename.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    base_model_name = os.path.basename(data_filename).replace('.csv', '')
    model_filename = os.path.join(model_dir, f"full_model_{base_model_name}.h5")

    # Load training data
    data = pd.read_csv(data_filename)
    output_columns = ["X", "Y", "Z"]
    XYZ_train = data[output_columns]
    RSS_train = data.drop(columns=output_columns)

    # Load or train the model
    if os.path.exists(model_filename) and not force_retrain:
        print(f"Loading existing model: '{model_filename}'")
        model = load_model(model_filename)
    else:
        print(f"Training new model for: '{base_model_name}'")
        model = Sequential([
            Dense(15, input_shape=(RSS_train.shape[1],), activation="relu"),
            Dense(15, activation="relu"),
            Dense(XYZ_train.shape[1])
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit(
            RSS_train, XYZ_train,
            epochs=epochs,
            batch_size=250,
            verbose=1
        )

        model.save(model_filename)

    # Predict on test data
    data_test = pd.read_csv(test_filename)
    RSS_test = data_test.drop(columns=output_columns)
    predictions = model.predict(RSS_test)

    # Descale predictions
    with open(scaler, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    descaled_predictions = loaded_scaler.inverse_transform(
        np.hstack([np.zeros_like(RSS_test), predictions])
    )[:, -3:]

    # Save predictions
    pred_filename = os.path.join(
        prediction_dir,
        f"{base_model_name}_test_noise={mah_noise}.csv"
    )
    pd.DataFrame(descaled_predictions, columns=output_columns).to_csv(
        pred_filename, index=False
    )
    print(f"Predictions saved: {pred_filename}")

    # Training loss
    final_loss = model.evaluate(RSS_train, XYZ_train, verbose=0)
    print(f"Training loss ({base_model_name}): {final_loss}\n")

    return model

# Main loop
sample_sizes = range(50, 1501, 50)
seeds = range(10)
noise_levels = [0.0]

for n in sample_sizes:
    for seed in seeds:
        data_filename = f"3D-Data/Measured_relative_Normalized/relative_rss_n={n}_noise=0.0_seed={seed}.csv"
        for noise_level in noise_levels:
            test_filename = f"3D-Data/Measured_relative_Normalized/relative_gnd_n=50_noise={noise_level}.csv"
            train_or_load_model(
                data_filename=data_filename,
                test_filename=test_filename,
                force_retrain=False,
                epochs=20000,
                mah_noise=noise_level,
                scaler="scaler_relative.pkl"
            )
