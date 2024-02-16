import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import animation 
import queue

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
def calculate_averages(data_beacons, beacon2_index, beacon3_index):
    mask_beacon2 = data_beacons.index <= beacon2_index
    mask_beacon3 = data_beacons.index <= beacon3_index

    last_4_values_P1S1 = data_beacons.loc[mask_beacon2 & (data_beacons['P1s1'] != '')].tail(4)
    last_4_values_P2S2 = data_beacons.loc[mask_beacon2 & (data_beacons['P2s2'] != '')].tail(4)

    last_4_values_P1S2 = data_beacons.loc[mask_beacon3 & (data_beacons['P1s2'] != '')].tail(4)
    last_4_values_P2S1 = data_beacons.loc[mask_beacon3 & (data_beacons['P2s1'] != '')].tail(4)
    last_4_values_P3S1 = data_beacons.loc[mask_beacon3 & (data_beacons['P3s1'] != '')].tail(4)
    last_4_values_P3S2 = data_beacons.loc[mask_beacon3 & (data_beacons['P3s2'] != '')].tail(4)

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

    return avg_P1S1, avg_P2S2, avg_P1S2, avg_P2S1, avg_P3S1, avg_P3S2
def add_new_column1(data):
    new_column_values = []
    previous_value = None
    for index,row in data.iterrows():
        if row['P1s1'] != '' and 0.1 < float(row['P1s1']) < avg_P1S1 - 5:
            new_column_values.append(row['P1s1'])
        elif row['P1s2'] != '' and 0.1 < float(row['P1s2']) < avg_P1S2 - 5:
            new_column_values.append(row['P1s2'])
        else:
            new_column_values.append(previous_value)
        previous_value = new_column_values[-1]
    data.insert(8, 'B1->T', new_column_values)
    return data
def add_new_column2(data):
    new_column_values = []
    previous_value = None
    for index,row in data.iterrows():
        if row['P2s1'] != '' and 0.1 < float(row['P2s1']) < avg_P2S1 - 5:
            new_column_values.append(row['P2s1'])
        elif row['P2s2'] != '' and 0.1 < float(row['P2s2']) < avg_P2S2 - 5:
            new_column_values.append(row['P2s2'])
        else:
            new_column_values.append(previous_value)
        previous_value = new_column_values[-1]
    data.insert(9, 'B2->T', new_column_values)
    return data
def add_new_column3(data):
    new_column_values = []
    previous_value = None
    for index,row in data.iterrows():
        if row['P3s1'] != '' and 0.1 < float(row['P3s1']) < avg_P3S1 - 5:
            new_column_values.append(row['P3s1'])
        elif row['P3s2'] != '' and 0.1 < float(row['P3s2']) < avg_P3S2 - 5:
            new_column_values.append(row['P3s2'])
        else:
            new_column_values.append(previous_value)
        previous_value = new_column_values[-1]
    data.insert(10, 'B3->T', new_column_values)
    return data
def difference_columns(df, column1, column2):
    #values_column1 = df[column1].loc[df[column1] != '']
    #values_column2 = df[column2].loc[df[column2] != '']
    values_column1 = df[column1].dropna().iloc[:6]
    values_column2 = df[column2].dropna().iloc[:6]

    numeric_values_column1 = pd.to_numeric(values_column1, errors='coerce')
    avg_B = numeric_values_column1.mean()

    numeric_values_column2 = pd.to_numeric(values_column2, errors='coerce')
    avg_T = numeric_values_column2.mean()

    return avg_B - avg_T
def get_measurements(grouped_data):
    measurements = []
    for col in ['P4s1', 'P4s2', 'P4s3', 'P4s4', 'P4s5', 'P4s6']:
        for _, row in grouped_data.iterrows():
            value = row[col]
            if value != '':
                measurements.append(value)
                break  # Exit the loop once a non-empty value is found
    return measurements
def check_error(entry,measurements,beacon1,beacon2,beacon3):
    error1 = measurements[entry[0]-1] - beacon1 + 30
    error2 = measurements[entry[1]-1] - beacon2 + 20
    error3 = measurements[entry[2]-1] - beacon3 + 10
    # print(entry)
    # print(f"Beacon1:{beacon1}, measurement: {measurements[entry[0]-1]}")
    # print(f"Error 1: {error1}")
    # print(f"Error 2: {error2}")
    # print(f"Error 3: {error3}")
    total_error = abs(error1) + abs(error2) + abs(error3) 
    # print(f"Total error: {total_error}")
    return(total_error)
def find_lowest_error(all_errors):
    min_error = min(all_errors)
    min_error_index = all_errors.index(min_error)
    return min_error_index
orientations = [
(1, 2, 3),(1, 2, 4),(1, 2, 5),(1, 2, 6),(1, 3, 4),(1, 3, 5),(1, 3, 6),(1, 4, 5),(1, 4, 6),(1, 5, 6),
(2, 3, 1),(2, 3, 4),(2, 3, 5),(2, 3, 6),(2, 4, 1),(2, 4, 5),(2, 4, 6),(2, 5, 1),(2, 5, 6),(2, 6, 1),
(3, 1, 2),(3, 4, 1),(3, 4, 2),(3, 4, 5),(3, 4, 6),(3, 5, 1),(3, 5, 2),(3, 5, 6),(3, 6, 1),(3, 6, 2),
(4, 1, 2),(4, 1, 3),(4, 2, 3),(4, 5, 1),(4, 5, 2),(4, 5, 3),(4, 5, 6),(4, 6, 1),(4, 6, 2),(4, 6, 3),
(5, 1, 2),(5, 1, 3),(5, 1, 4),(5, 2, 3),(5, 2, 4),(5, 3, 4),(5, 6, 1),(5, 6, 2),(5, 6, 3),(5, 6, 4),
(6, 1, 2),(6, 1, 3),(6, 1, 4),(6, 1, 5),(6, 2, 3),(6, 2, 4),(6, 2, 5),(6, 3, 4),(6, 3, 5),(6, 4, 5)]
# Main code
csv_file_path = 'test_data.csv'
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()
data_beacons = read_data(csv_file_path, header_line)
beacon2_index, beacon3_index = calculate_beacon_indices(data_beacons)
avg_P1S1, avg_P2S2, avg_P1S2, avg_P2S1, avg_P3S1, avg_P3S2 = calculate_averages(data_beacons, beacon2_index, beacon3_index)
# Read your CSV file and perform necessary preprocessing
data = pd.read_csv(csv_file_path, skiprows=beacon3_index, header=None, delimiter=',', usecols=range(13))
headers = header_line.split(',')[:13]
data.columns = headers
data = data.round(1)
data = data.fillna('')
grouped_data = data.groupby(data.index // 4).agg({col: aggregate_non_empty for col in data.columns}).reset_index()
grouped_data = add_new_column1(grouped_data)
grouped_data = add_new_column2(grouped_data)
grouped_data = add_new_column3(grouped_data)
dif1 = difference_columns(grouped_data,'B1->T','P4s1')
dif2 = difference_columns(grouped_data,'B2->T','P4s3')
dif3 = difference_columns(grouped_data,'B3->T','P4s5')
print(f"Dif1:{dif1} dif2:{dif2} dif3:{dif3}")
print(grouped_data)
# orientation calculations
measurements = get_measurements(grouped_data)
column_name = 'B1->T'
first_six_entries = grouped_data[column_name].iloc[:6]
beacon1 = first_six_entries.mean()
print(f"Beacon 1 {beacon1}")

column_name = 'B2->T'
first_six_entries = grouped_data[column_name].iloc[:6]
beacon2 = first_six_entries.mean()
print(f"Beacon 2 {beacon2}")

column_name = 'B3->T'
first_six_entries = grouped_data[column_name].iloc[:6]
beacon3 = first_six_entries.mean()
print(f"Beacon 3 {beacon3}")

print(measurements)
all_errors = []
for entry in orientations:
    error = check_error(entry,measurements,beacon1,beacon2,beacon3)
    all_errors.append(error)
    print("Error for", entry, ":", error)  # Debugging step

print("Error for entry 5:", orientations[5], ":", all_errors[5])  # Debugging step
# Find the index of the combination with the lowest error
combination_index = find_lowest_error(all_errors)

# Retrieve the combination with the lowest error
combination = orientations[combination_index]

print(f"Best combination: {combination}, total error: {all_errors[find_lowest_error(all_errors)]}")