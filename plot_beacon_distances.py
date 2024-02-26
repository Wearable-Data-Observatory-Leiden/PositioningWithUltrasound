import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(csv_file_path, header_line):
    # Read the CSV file with the determined delimiter and specifying the number of columns to read
    data_beacons = pd.read_csv(csv_file_path, header=None, delimiter=',', usecols=range(7))
    # Get the column names from the first row of the DataFrame
    headers = header_line.split(',')[:7]
    # Use the column names for the DataFrame
    data_beacons.columns = headers
    data_beacons = data_beacons.fillna('')
    return data_beacons
def calculate_beacon_indices(data_beacons):
    consecutive_empty_count = 0
    beacon2_index = None
    for index, row in data_beacons.iterrows():
        consecutive_empty_count = consecutive_empty_count + 1 if row['P1s1'] == '' else 0
        if consecutive_empty_count == 3: 
            beacon2_index = index - 1  # Index of the first empty row in the consecutive sequence
            break
    consecutive_empty_count = 0
    beacon3_index = None
    for index, row in data_beacons.iloc[beacon2_index:].iterrows():
        consecutive_empty_count = consecutive_empty_count + 1 if row['P3s2'] == '' else 0
        if consecutive_empty_count == 4: 
            beacon3_index = index - 3  # Index of the first empty row in the consecutive sequence
            break
    return beacon2_index, beacon3_index
def aggregate_non_empty(series):
    return series[series != ''].iloc[-1] if any(series != '') else ''

# Main code

# read in the data and find the moment where the beacons are initialized, and at what distance
csv_file_path = 'test_data.csv'
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()
data_beacons = read_data(csv_file_path, header_line)
beacon2_index, beacon3_index = calculate_beacon_indices(data_beacons)

# read in the data after beacon initialization and prepare the data in compressed rows
data = pd.read_csv(csv_file_path, skiprows=beacon3_index, header=None, delimiter=',', usecols=range(13))
headers = header_line.split(',')[:13]
data.columns = headers
data = data.round()
data = data.fillna('')
grouped_data = data.groupby(data.index // 4).agg({col: aggregate_non_empty for col in data.columns}).reset_index()
print(grouped_data)

beacon_numbers = [1, 2, 3]
for beacon_num in beacon_numbers:
    # Convert columns to numeric if needed
    grouped_data[f'P{beacon_num}s1'] = pd.to_numeric(grouped_data[f'P{beacon_num}s1'], errors='coerce')
    grouped_data[f'P{beacon_num}s2'] = pd.to_numeric(grouped_data[f'P{beacon_num}s2'], errors='coerce')
    
    # Plotting the values for each beacon
    plt.figure()  # Create a new figure for each plot
    plt.plot(grouped_data.index, grouped_data[f'P{beacon_num}s1'], marker='o', markersize=2, label=f'P{beacon_num}s1')
    plt.plot(grouped_data.index, grouped_data[f'P{beacon_num}s2'], marker='o', markersize=2, color='orange', label=f'P{beacon_num}s2')
    plt.xlabel('Index')
    plt.ylabel('Sensor Values')
    plt.title(f'Plot of Beacon {beacon_num}')
    plt.legend()
    plt.show()

