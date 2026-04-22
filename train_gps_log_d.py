import numpy as np
import gpflow
import matplotlib.pyplot as plt
import os
import scipy as sp
import pandas as pd
import tensorflow as tf
import ipywidgets as widgets
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize



def mse_error (pos, beacon_coords, distances):

    predicted_distances = np.linalg.norm(beacon_coords - pos , axis=1)
    mse = np.mean((predicted_distances - distances)**2)
    return(mse)




def trilaterate_mse(distances,beacon_coords=np.array([[0,0,5],[-2,-2,5],[-2,2,5],[2,-2,5],[2,2,5]])):
    initial_guess = np.mean(beacon_coords,axis=0)
    bounds = [(None, None), (None, None), (None, 5)]

    result = minimize (mse_error, initial_guess, args=(beacon_coords,distances),method= "L-BFGS-B",bounds=bounds)


    if result.success:
        return(result.x)
    else:
        raise ValueError("You Trolled" + result.message)
    





categories = ['Measured_log_d_Normalized']
noise_levels = [0.0]
ns = range(50, 1501, 50)
seeds = range(10)  # e.g., 0..9

base_dir = '3D-Data'

# ----------------------------------------------------------------------
# 2) Load data into a dictionary keyed by (category, noise_level, n, seed)
#    This time, naming convention is "rss_n={n}_noise={noise_level}_seed={seed}.csv"
# ----------------------------------------------------------------------
train_dict = {}
for category in categories:
    for noise_level in noise_levels:
        for n in ns:
            for seed in seeds:
                file_name = f"rss_n={n}_noise={noise_level}_seed={seed}.csv"
                file_path = os.path.join(base_dir, category, file_name)

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    train_dict[(category, noise_level, n, seed)] = df
                else:
                    print(f"No such file: {file_path}")

# ----------------------------------------------------------------------
# 3) Train function for each (category, noise, n, seed) combination
# ----------------------------------------------------------------------
def gp_trainer(train_dict=train_dict):
    gp_dict = {}
    for (category, noise_level, n, seed), df in train_dict.items():
        train_input = df[["RSS0", "RSS1", "RSS2", "RSS3", "RSS4"]]
        train_output = df[["D0", "D1", "D2", "D3", "D4"]]

        model = gpflow.models.GPR(
            data=(train_input.values, train_output.values),
            kernel=gpflow.kernels.SquaredExponential(lengthscales=(0.2,0.2,0.2,0.2,0.2)),
            noise_variance=1e-4
        )
        gpflow.set_trainable(model.likelihood, True)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        gp_dict[(category, noise_level, n, seed)] = model
        print(f"Trained model for category={category}, noise={noise_level}, n={n}, seed={seed}")
    return gp_dict

# ----------------------------------------------------------------------
# 4) Train and save all models
# ----------------------------------------------------------------------
gp_models = gp_trainer(train_dict)

with open('gp_models_log_d.pkl', 'wb') as file:
    pickle.dump(gp_models, file)

print("Finished training all GP 'log_d' models (one per seed).")


with open('gp_models_log_d.pkl', 'rb') as file:
    gp_models = pickle.load(file)




test_dict={}
m=50
noise_levels= [0.]
for category in categories:
    for noise_level in noise_levels:
        file_name= f"gnd_n={m}_noise={noise_level}.csv"
        file_path = os.path.join(base_dir,category,file_name)
        if os.path.exists(file_path):
            test_dict[(category,noise_level)] = pd.read_csv(file_path)
        else:
            print('no such file', category)
            

def make_predictions(
    gp_model,
    test_data,
    prediction_save_directory="predictions/gps_log_d",
    n=9,
    noise_level=0.0,
    seed=0,
    scaler="scaler_log_d.pkl"
):
    """
    Make predictions for the given GP log-d model and test data, then save to CSV.
    Includes `seed` in the output filename to avoid overwriting different models.
    """
    import pickle
    import numpy as np
    import os

    # These are the columns in the test_data to drop (the 5 distances in log-space)
    distance_columns_log = ["D0", "D1", "D2", "D3", "D4"]

    # Build a filename that includes the seed
    filename = f"rss_n={n}_noise={noise_level}_seed={seed}.csv"

    # Prepare the test inputs (RSS0..RSS4, etc.) by dropping the D0..D4 columns
    RSS_test = test_data.drop(distance_columns_log, axis=1).values

    # Make predictions
    mean, covariance = gp_model.predict_f(RSS_test)

    # Load your log-d scaler
    with open(scaler, 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    # Descale the predictions (log-distances), then exponentiate to get linear distances
    predictions = mean
    descaled_predictions = loaded_scaler.inverse_transform(
        np.hstack([np.zeros_like(RSS_test), predictions])
    )[:, -5:]  # last 5 columns correspond to D0..D4 in linear scale, still log space
    linear_distances = np.exp(descaled_predictions)

    # Trilaterate each row of linear distances to get (X, Y, Z)
    xyz_coordinates = []
    beacon_coords = np.array([
        [0, 0, 5],
        [-2, -2, 5],
        [-2,  2, 5],
        [ 2, -2, 5],
        [ 2,  2, 5]
    ])
    for distance_set in linear_distances:
        xyz = trilaterate_mse(distance_set, beacon_coords=beacon_coords)
        xyz_coordinates.append(xyz)
    
    predictions_df = pd.DataFrame(xyz_coordinates, columns=["X", "Y", "Z"])

    # Ensure directory exists
    if not os.path.exists(prediction_save_directory):
        os.makedirs(prediction_save_directory)

    # Save predictions
    prediction_save_path = os.path.join(prediction_save_directory, filename)
    predictions_df.to_csv(prediction_save_path, index=False)
    print(f"Saved predictions to {prediction_save_path}")

    return predictions_df
# i is (category, noise_level, n, seed) for gp_models
for i in gp_models.keys():
    for j in test_dict.keys():
        make_predictions(
            gp_models[i],
            test_dict[j],
            n=i[2],
            noise_level=j[1],   # noise for the test data
            seed=i[3],         # pass the seed from the model key
            prediction_save_directory="predictions/gps_log_d"
        )
