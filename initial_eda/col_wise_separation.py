import pandas as pd
from sklearn import preprocessing
nodewise_data_path = '../processed_data/'

machine_file_list = ['fd_normal.csv', 'fd_normal2.csv','fd_normal3.csv', 'fr_normal.csv', 'fr_normal2.csv','fr_normal3.csv', 'preprocess_normal.csv', 'preprocess_normal2.csv', 'preprocess_normal3.csv', 'preprocess_normal4.csv']

# # This 1st section is to count all zero occurrences
#
# for machine_file in machine_file_list:
#     df = pd.read_csv(nodewise_data_path+machine_file)
#     describe_df = df.describe()
#     describe_df.to_csv('../initial_eda/df_describe/' + machine_file)

# Counting no.of all zero columns in df
all_zero_count = {'total_disk_read_throughput':0,'total_disk_write_throughput':0,'rss':0,'vsize':0,'cpu_usage':0,'rx_bytes_per_ns':0,'tx_bytes_per_ns':0,'latency_p50':0, 'latency_p90':0, 'latency_p99':0, 'request_throughput':0, 'errors_per_ns':0}
col_id_to_name_mapping = ['total_disk_read_throughput','total_disk_write_throughput','rss','vsize','cpu_usage','rx_bytes_per_ns','tx_bytes_per_ns','latency_p50','latency_p90','latency_p99','request_throughput','errors_per_ns']
for i in range(12):
    for machine_file in machine_file_list:
        df = pd.read_csv('../initial_eda/df_describe/'+machine_file)
        min_val = df.iloc[3][col_id_to_name_mapping[i]]
        max_val = df.iloc[7][col_id_to_name_mapping[i]]
        if (max_val-min_val)==0:
            all_zero_count[col_id_to_name_mapping[i]]+=1
print(all_zero_count)

# Following are the results obtained from the above step
# {'total_disk_read_throughput': 2, 'total_disk_write_throughput': 0, 'rss': 0, 'vsize': 0, 'cpu_usage': 0, 'rx_bytes_per_ns': 0, 'tx_bytes_per_ns': 0, 'latency_p50': 0, 'latency_p90': 0, 'latency_p99': 0, 'request_throughput': 0, 'errors_per_ns': 10}
# We can ignore errors_per_ns col altogether and total_disk_read_throughput column later on.

# final_df = {'fd_normal':None, 'fd_normal2':None, 'fd_normal3':None, 'fr_normal':None, 'fr_normal2':None, 'fr_normal3':None, 'preprocess_normal':None, 'preprocess_normal2':None, 'preprocess_normal3':None, 'preprocess_normal4':None}
# # Rather than drawing individual boxplots, form a dataframe per column
# for i in range(11):
#     print("column_"+col_id_to_name_mapping[i])
#     for machine_file in machine_file_list:
#         df = pd.read_csv(nodewise_data_path + machine_file)
#         final_df[machine_file.split('.csv')[0]] = df[col_id_to_name_mapping[i]]
#     result = pd.DataFrame(final_df)
#     result.to_csv('../initial_eda/col_wise_dfs/' + 'column_' + col_id_to_name_mapping[i] + '.csv', index=False)

# Standardize each column of the col_wise_dfs in the initial_eda folder before choosing the metrics using variance of mean
min_max_scaler = preprocessing.MinMaxScaler()

for i in range(11):
    df = pd.read_csv('col_wise_dfs/column_'+col_id_to_name_mapping[i]+'.csv')
    x_scaled = min_max_scaler.fit_transform(df)
    scaled_df = pd.DataFrame(x_scaled)
    scaled_df.columns = df.columns
    # scaled_df.to_csv('../standardized_data/standardized_col_wise_dfs/' + 'column_' + col_id_to_name_mapping[i] + '.csv', index=False)

df_list = []
for machine_file in machine_file_list:
    df = pd.read_csv('df_describe/' + machine_file)
    df_list.append(df.iloc[[1], 1:])
final_df = pd.concat(df_list, ignore_index=True)
final_df.insert(0,"Machine",[f.strip('.csv') for f in machine_file_list])
# final_df.to_csv('mean_analysis.csv', index=False)

df = pd.read_csv('mean_analysis.csv')
selected_column_list = ['Machine','total_disk_read_throughput','total_disk_write_throughput','cpu_usage','request_throughput','vsize']
sub_df = df[selected_column_list]
sub_df.to_csv('5_selected_columns_for_10_machines.csv', index=False)