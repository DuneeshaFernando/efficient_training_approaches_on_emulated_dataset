import pandas as pd
from sklearn import preprocessing
from itertools import combinations
from scipy.spatial import distance
from kruskal_algo import *
from mst_using_kruskal import *
import numpy as np

min_max_scaler = preprocessing.MinMaxScaler()

# Obtain the 4 distributions for the 7 machines w/o concept drift
normal_dataset_path = '../processed_data/'
machine_files = ['fd_normal.csv', 'fd_normal2.csv', 'fd_normal3.csv', 'fr_normal.csv', 'fr_normal2.csv', 'fr_normal3.csv', 'preprocess_normal.csv', 'preprocess_normal2.csv', 'preprocess_normal3.csv', 'preprocess_normal4.csv']
# Following machine_files lists are used when obtaining similarity chains for individual clusters
# machine_files = ['fd_normal.csv', 'fd_normal2.csv']
# machine_files = [fr_normal.csv', 'fr_normal2.csv']
# machine_files = ['preprocess_normal.csv', 'preprocess_normal2.csv', 'preprocess_normal3.csv']
selected_cols = ['total_disk_write_throughput','cpu_usage','request_throughput','vsize'] # Although total_disk_read_throughput is a metric with high variance of mean, I removed it from this list as it is an all-zero column for fr microservices.
normal_df_list = [pd.read_csv(normal_dataset_path + normal_data_file, usecols=selected_cols) for normal_data_file in machine_files]

# Create dictionary by numbering machines in machine_files
machine_diction = {}
machine_diction_index = 0
for machine in machine_files:
    machine_diction[machine_diction_index]=machine.split(".csv")[0]

# Normalise the dataset
# First concat all dataframes to fit
normal_df = pd.concat(normal_df_list)
normal_df = normal_df.astype(float)

min_max_scaler.fit(normal_df)

pairwise_jsd_list = []
# Obtain pair-wise combinations of machines
for comb in combinations([i for i in range(len(machine_files))], 2):
    machine_1_df = pd.read_csv(normal_dataset_path + machine_files[comb[0]], usecols=selected_cols)
    machine_1_df_values = machine_1_df.values
    machine_1_df_values_scaled = min_max_scaler.transform(machine_1_df_values)

    machine_2_df = pd.read_csv(normal_dataset_path + machine_files[comb[1]], usecols=selected_cols)
    machine_2_df_values = machine_2_df.values
    machine_2_df_values_scaled = min_max_scaler.transform(machine_2_df_values)

    machine_1_col1 = pd.cut(machine_1_df_values_scaled[:, 0], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_1_col2 = pd.cut(machine_1_df_values_scaled[:, 1], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_1_col3 = pd.cut(machine_1_df_values_scaled[:, 2], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_1_col4 = pd.cut(machine_1_df_values_scaled[:, 3], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    # machine_1_col5 = pd.cut(machine_1_df_values_scaled[:, 3], [round(i * 0.1, 2) for i in range(11)]).value_counts()

    machine_1_col1[:] = machine_1_col1[:] / machine_1_col1.sum()
    machine_1_col2[:] = machine_1_col2[:] / machine_1_col2.sum()
    machine_1_col3[:] = machine_1_col3[:] / machine_1_col3.sum()
    machine_1_col4[:] = machine_1_col4[:] / machine_1_col4.sum()
    # machine_1_col5[:] = machine_1_col5[:] / machine_1_col5.sum()

    machine_2_col1 = pd.cut(machine_2_df_values_scaled[:, 0], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_2_col2 = pd.cut(machine_2_df_values_scaled[:, 1], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_2_col3 = pd.cut(machine_2_df_values_scaled[:, 2], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    machine_2_col4 = pd.cut(machine_2_df_values_scaled[:, 3], [round(i * 0.1, 2) for i in range(11)]).value_counts()
    # machine_2_col5 = pd.cut(machine_2_df_values_scaled[:, 4], [round(i * 0.1, 2) for i in range(11)]).value_counts()

    machine_2_col1[:] = machine_2_col1[:] / machine_2_col1.sum()
    machine_2_col2[:] = machine_2_col2[:] / machine_2_col2.sum()
    machine_2_col3[:] = machine_2_col3[:] / machine_2_col3.sum()
    machine_2_col4[:] = machine_2_col4[:] / machine_2_col4.sum()
    # machine_2_col5[:] = machine_2_col5[:] / machine_2_col5.sum()

    # Fill nan values with 0. This is useful if we consider total_disk_read_throughput which is an all-zero column for fr microservices.
    machine_1_col1[:] = machine_1_col1[:].fillna(0)
    machine_2_col1[:] = machine_2_col1[:].fillna(0)

    data_machine_1 = pd.concat([machine_1_col1, machine_1_col2, machine_1_col3, machine_1_col4], axis=1)
    data_machine_2 = pd.concat([machine_2_col1, machine_2_col2, machine_2_col3, machine_2_col4], axis=1)

    np_machine_1 = data_machine_1.to_numpy()
    np_machine_2 = data_machine_2.to_numpy()

    dist = distance.jensenshannon(np_machine_1, np_machine_2, axis=0)
    # Fill nan values in array with 0
    dist_filled = np.nan_to_num(dist, nan=0)

    pairwise_jsd_list.append([comb[0], comb[1], dist_filled])

print(pairwise_jsd_list)

# This 1st code is to obtain Kruskal based clusters

g = Graph(len(machine_files))
for entry in pairwise_jsd_list:
    print(entry)
    d1 = entry[2][0] ** 2
    d2 = entry[2][1] ** 2
    d3 = entry[2][2] ** 2
    d4 = entry[2][3] ** 2
    # d5 = entry[2][4] ** 2
    d = np.sqrt(d1 + d2 + d3 + d4) # + d5)
    g.addEdge(entry[0], entry[1], d)

cluster_dict, mach_cluster_dict = g.KruskalMSTClusters(K=3)

for i in range(len(cluster_dict)):
    new_list = []
    for item in cluster_dict[i]:
        new_list.append(machine_files[item].strip('.csv'))
    cluster_dict[i]=new_list
print(cluster_dict)

for i in range(len(mach_cluster_dict)):
    mach_cluster_dict[machine_files[i]] = mach_cluster_dict.pop(str(i))
print(mach_cluster_dict)

# This 2nd code is to obtain the similarity chains within individual clusters. Select correct machine_files list from above

mstg = MSTGraph(len(machine_files))
for entry in pairwise_jsd_list:
    print(entry)
    d1 = entry[2][0] ** 2
    d2 = entry[2][1] ** 2
    d3 = entry[2][2] ** 2
    d4 = entry[2][3] ** 2
    # d5 = entry[2][4] ** 2
    d = np.sqrt(d1 + d2 + d3 + d4) # + d5)
    mstg.addEdge(entry[0], entry[1], d)

mst_final_list = mstg.KruskalMST()
print("************")
print(mst_final_list)
print("************")
for item in mst_final_list:
    item[0] = machine_files[item[0]]
    item[1] = machine_files[item[1]]
print(mst_final_list)