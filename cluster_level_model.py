import os
import pandas as pd
from sklearn import preprocessing
import torch
import numpy as np
from os.path import join
import torch.utils.data as data_utils
import src.autoencoder as autoencoder
import src.constants as const
from config import config as conf
from src.evaluation import Evaluation
from datetime import datetime

window_size = 18
hidden_size = 35
batch_size = 60
learning_rate = 4.507664256956887e-05
num_layers = 3

# Pre-requisites
min_max_scaler = preprocessing.MinMaxScaler()

# setting seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# List all the microservices in SN dataset
dataset_path = const.DATASET_LOCATION

# Read normal data.
normal_files = ["preprocess_normal","preprocess_normal2","preprocess_normal3","preprocess_normal4"]
normal_df_list = [pd.read_csv(dataset_path+'/'+normal_file+".csv") for normal_file in normal_files]
normal_df = pd.concat(normal_df_list)
normal_df = normal_df.astype(float)

# Read anomaly data. Merge 2 anomaly files.
anomaly_files = ["preprocess_usersurge","preprocess_other_anoms","preprocess_usersurge_recollect","preprocess_other_anoms2","preprocess_usersurge3","preprocess_other_anoms3","preprocess_usersurge4_recollect","preprocess_other_anoms4"]
anomaly_df_list = [pd.read_csv(dataset_path+'/'+anom_file+".csv") for anom_file in anomaly_files]
anomaly_df = pd.concat(anomaly_df_list)
# Separate out the anomaly labels before normalisation/standardization
anomaly_df_labels = anomaly_df['Normal/Attack']
anomaly_df = anomaly_df.drop(['Normal/Attack'], axis=1)
anomaly_df = anomaly_df.astype(float)

# Normalise/ standardize the normal and anomaly dataframe
full_df = pd.concat([normal_df, anomaly_df])
min_max_scaler.fit(full_df)

normal_df_values = normal_df.values
normal_df_values_scaled = min_max_scaler.transform(normal_df_values)
normal_df_scaled = pd.DataFrame(normal_df_values_scaled)

anomaly_df_values = anomaly_df.values
anomaly_df_values_scaled = min_max_scaler.transform(anomaly_df_values)
anomaly_df_scaled = pd.DataFrame(anomaly_df_values_scaled)

# Preparing the datasets for training and testing using AutoEncoder
t1 = np.arange(window_size)[None, :]
merge_points = []
for df in anomaly_df_list:
    merge_points.append(len(df))

t2_for_merge_points = []
cumul_point = 0
for k in range(len(merge_points)):
    t2_for_merge_points.append(np.arange(anomaly_df_scaled.shape[0] - window_size + 1)[cumul_point:cumul_point+merge_points[k]-window_size+1, None])
    cumul_point += merge_points[k]

t3_for_merge_points = []
for t2_sublist in t2_for_merge_points:
    t3_for_merge_points.append(t1 + t2_sublist)

# Merge all elements inside t3_for_merge_points
t3_final = t3_for_merge_points[0]
for i in range(1,len(t3_for_merge_points)):
    t3_final = np.concatenate((t3_final, t3_for_merge_points[i]), axis=0)

# Preparing the datasets for training and testing using AutoEncoder
windows_normal = normal_df_scaled.values[np.arange(window_size)[None, :] + np.arange(normal_df_scaled.shape[0] - window_size)[:, None]]
# windows_anomaly = anomaly_df_scaled.values[np.arange(window_size)[None, :] + np.arange(anomaly_df_scaled.shape[0] - window_size)[:, None]]
windows_anomaly = anomaly_df_scaled.values[t3_final]

w_size = windows_normal.shape[1] * windows_normal.shape[2] # w_size is the input window size
z_size = hidden_size # z_size is the latent size

windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 * windows_normal.shape[0])):]

# Create batches of training and testing data
train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], w_size]))
), batch_size=batch_size, shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], w_size]))
), batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_anomaly).float().view(([windows_anomaly.shape[0], w_size]))
), batch_size=batch_size, shuffle=False, num_workers=0)

# Initialise the AutoEncoder model
autoencoder_model = autoencoder.AutoEncoder(in_size=w_size, latent_size=z_size, num_layers=num_layers)
# Start training and save the best model, i.e. the model with the least validation loss
model_path = const.MODEL_LOCATION
model_name = join(model_path, "ae_model_{}.pth".format(2))  # parameterize the run number
start_time = datetime.now()
autoencoder.training(conf.N_EPOCHS, autoencoder_model, train_loader, val_loader, learning_rate, model_name)
end_time = datetime.now()
print("Training time :", end_time-start_time)

# Load the model
checkpoint = torch.load(model_name)
autoencoder_model.encoder.load_state_dict(checkpoint['encoder'])
autoencoder_model.decoder.load_state_dict(checkpoint['decoder'])

# Use the trained model to obtain predictions for the test set
results = autoencoder.testing(autoencoder_model, test_loader)
y_pred_for_test_set = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(), results[-1].flatten().detach().cpu().numpy()])

# Process the actual labels. We have to skip the merge points for labels as well
windows_labels = anomaly_df_labels.values[t3_final]
processed_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

thresholding_percentile = 100 - (((processed_test_labels.count(1.0)) / (len(processed_test_labels))) * 100)

# Obtain threshold based on pth percentile of the mean squared error
threshold = np.percentile(y_pred_for_test_set, [thresholding_percentile])[0]  # 90th percentile

# Map the predictions to anomaly labels after applying the threshold
predicted_labels = []
for val in y_pred_for_test_set:
    if val > threshold:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

# Evaluate the predicted_labels against the actual labels
print([int(label) for label in processed_test_labels])
print([round(label,1) for label in predicted_labels])
test_eval = Evaluation(processed_test_labels, predicted_labels)
test_eval.print()

print("debug")
