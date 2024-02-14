import pandas as pd
import numpy as np
# Specify the path to your CSV file
csv_file_path = 'test_data.csv'

# Read the header from the CSV file
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()

# Read the CSV file with the determined delimiter and specifying the number of columns to read
data_beacons = pd.read_csv(csv_file_path, header=None, delimiter=',', usecols=range(7))

# Get the column names from the first row of the DataFrame
headers = header_line.split(',')[:7]

# Use the column names for the DataFrame
data_beacons.columns = headers
data_beacons = data_beacons.fillna('')


# Iterate over the DataFrame and check for three consecutive empty rows in column 'P1s1'
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
print(f"Index B2: {beacon2_index}")
print(f"Index B3: {beacon3_index}")

last_4_values_P1S1 = data_beacons.loc[:beacon2_index][data_beacons['P1s1'] != ''].tail(4)
last_4_values_P2S2 = data_beacons.loc[:beacon2_index][data_beacons['P2s2'] != ''].tail(4)

last_4_values_P1S2 = data_beacons.loc[:beacon3_index][data_beacons['P1s2'] != ''].tail(4)
last_4_values_P2S1 = data_beacons.loc[:beacon3_index][data_beacons['P2s1'] != ''].tail(4)
last_4_values_P3S1 = data_beacons.loc[:beacon3_index][data_beacons['P3s1'] != ''].tail(4)
last_4_values_P3S2 = data_beacons.loc[:beacon3_index][data_beacons['P3s2'] != ''].tail(4)

numeric_values = pd.to_numeric(last_4_values_P1S1['P1s1'], errors='coerce')
avg_P1S1 = numeric_values.mean()
numeric_values = pd.to_numeric(last_4_values_P2S2['P2s2'], errors='coerce')
avg_P2S2 = numeric_values.mean()

numeric_values = pd.to_numeric(last_4_values_P1S2['P1s2'], errors='coerce')
avg_P1S2 = numeric_values.mean()
numeric_values = pd.to_numeric(last_4_values_P2S1['P2s1'], errors='coerce')
avg_P2S1 = numeric_values.mean()
numeric_values = pd.to_numeric(last_4_values_P3S1['P3s1'], errors='coerce')
avg_P3S1 = numeric_values.mean()
numeric_values = pd.to_numeric(last_4_values_P3S2['P3s2'], errors='coerce')
avg_P3S2 = numeric_values.mean()

print(f"Avg value p1s1: {avg_P1S1}, p2s2: {avg_P2S2},p1s2: {avg_P1S2}, p2s1: {avg_P2S1},p3s1: {avg_P3S1}, p3s2: {avg_P3S2}")

data = pd.read_csv(csv_file_path, skiprows=75, header=None, delimiter=',', usecols=range(7))
# Get the column names from the first row of the DataFrame
headers = header_line.split(',')[:7]

# Use the column names for the DataFrame
data.columns = headers

# Round the numbers to the nearest integer
data = data.round()
data = data.fillna('')
# Define a function to aggregate non-empty values into a string without brackets
def aggregate_non_empty(x):
    return ', '.join(str(value) for value in x if value)

# Group the DataFrame by every 4 rows and aggregate the values without brackets
grouped_data = data.groupby(data.index // 4).agg(aggregate_non_empty)

print(grouped_data)