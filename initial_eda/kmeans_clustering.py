# Kmeans clustering is performed as an EDA

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

N_CLUSTERS = 3

print("Clustering based on train sets of 10 machines")

# Read from dataframe and create numpy array
machine_list = []
column_value_list = []
df = pd.read_csv('5_selected_columns_for_10_machines.csv')

for i in range(len(df)):
    machine_list.append(df.iloc[i]['Machine'])
    column_value_list.append([df.iloc[i]['total_disk_read_throughput'], df.iloc[i]['total_disk_write_throughput'], df.iloc[i]['cpu_usage'], df.iloc[i]['request_throughput'],df.iloc[i]['vsize']])

print(column_value_list)
print(machine_list)
X = np.array(column_value_list)

# # This 1st section will be used to obtain optimal K for k-means clustering on the SMD dataset
# distortions = []
# N_CLUSTERS = range(1,7)
# for k in N_CLUSTERS:
#     kmeanModel = KMeans(n_clusters=k, random_state=0)
#     kmeanModel.fit(X)
#     distortions.append(kmeanModel.inertia_)
#
# plt.figure(figsize=(16,8))
# plt.plot(N_CLUSTERS, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

# This 2nd section will be used to perform k-means clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(X)
kmeans_labels = kmeans.labels_
print(kmeans_labels)
print(kmeans.cluster_centers_)

for i in range(N_CLUSTERS):
    print('cluster'+str(i))
    arr = np.where(kmeans_labels == i)
    li = arr[0].tolist()
    for j in li:
        print(machine_list[j], end = ' ')
    print("\n")