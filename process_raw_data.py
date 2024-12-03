# Process the collected raw data and store them in processed_data folder
# Read the csvs, and remove the container_x and time_ columns
import pandas as pd

# df = pd.read_csv("collected_data/merged_timeseries_fr_normal3.csv")
# df = df.drop(['container_x', 'time_'], axis=1)
# df.to_csv("processed_data/fr_normal3.csv", index=False)

# anom_df = pd.read_csv("collected_data/merged_timeseries_preprocess_usersurge4_recollect.csv")
# anom_df = anom_df.drop(['container_x', 'time_'], axis=1)
# anom_df['Normal/Attack'] = 0
# anom_df.to_csv("processed_data/preprocess_usersurge4_recollect.csv", index=False)

df = pd.read_csv("processed_data/preprocess_usersurge4_recollect.csv")
# df2 = pd.read_csv("processed_data/preprocess_usersurge_recollect.csv")
print("debug")