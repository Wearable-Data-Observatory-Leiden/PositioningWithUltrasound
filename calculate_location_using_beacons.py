import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def trilateration(r1,r2,r3):
    P1 = np.array([0, 0, 0])   # beacon locations
    P2 = np.array([180, 0, 0])
    P3 = np.array([100, 230, 0])    

    p1 = np.array([0, 0, 0])
    p2 = np.array([P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]])
    p3 = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])
    v1 = p2 - p1
    v2 = p3 - p1

    Xn = (v1)/np.linalg.norm(v1)

    tmp = np.cross(v1, v2)

    Zn = (tmp)/np.linalg.norm(tmp)

    Yn = np.cross(Xn, Zn)

    i = np.dot(Xn, v2)
    d = np.dot(Xn, v1)
    j = np.dot(Yn, v2)

    X = round(((r1**2)-(r2**2)+(d**2))/(2*d),1)
    Y = round((((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(X)), 1)
    Z1 = round(np.sqrt(max(0, r1**2-X**2-Y**2)), 1)
    Z2 = round(-Z1, 1)
    
    K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    K2 = P1 + X * Xn + Y * Yn + Z2 * Zn
 
    if K1[2] < 0:
        return K1
    elif K2[2] < 0:
        return K2
    else:
        #print("Error: Trilateration failed. Target point cannot be determined.")
        return None
def calculate_location(row):
    r1, r2, r3 = pd.to_numeric(row['1 final']), pd.to_numeric(row['2 final']), pd.to_numeric(row['3 final'])  # Extract r1, r2, r3 from the row
    return trilateration(r1, r2, r3)
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
def add_column_B1T(data):
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
def add_column_B2T(data):
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
def add_column_B3T(data):
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
    values_column1 = df[column1].loc[df[column1] != '']
    values_column2 = df[column2].loc[df[column2] != '']

    numeric_values_column1 = pd.to_numeric(values_column1, errors='coerce').head(5)
    avg_B = numeric_values_column1.mean()

    numeric_values_column2 = pd.to_numeric(values_column2, errors='coerce').head(5)
    avg_T = numeric_values_column2.mean()

    return avg_B - avg_T
def aggregate_non_empty(series):
    return series[series != ''].iloc[-1] if any(series != '') else ''
def calculate_movement(point1, point2):
    if point1 is None or isinstance(point1, int):
        # If point1 is an integer, return (0, 0, 0)
        return (0, 0, 0)
    else:
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distance
def fillna_with_tuple(series, value):
    return series.fillna(pd.Series([value] * len(series), index=series.index))


# Main code

# read in the data and find the moment where the beacons are initialized, and at what distance
csv_file_path = 'test_data.csv'
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()
data_beacons = read_data(csv_file_path, header_line)
beacon2_index, beacon3_index = calculate_beacon_indices(data_beacons)
avg_P1S1, avg_P2S2, avg_P1S2, avg_P2S1, avg_P3S1, avg_P3S2 = calculate_averages(data_beacons, beacon2_index, beacon3_index)

# read in the data after beacon initialization and prepare the data in compressed rows
data = pd.read_csv(csv_file_path, skiprows=beacon3_index, header=None, delimiter=',', usecols=range(13))
headers = header_line.split(',')[:13]
data.columns = headers
data = data.round()
data = data.fillna('')
grouped_data = data.groupby(data.index // 4).agg({col: aggregate_non_empty for col in data.columns}).reset_index()

# Make new columns that compresses the data of the 2 sensors on each beacon
# if measurement is equal to beacon initialization, last measurement is used 
grouped_data = add_column_B1T(grouped_data)
grouped_data = add_column_B2T(grouped_data)
grouped_data = add_column_B3T(grouped_data)

# Calculate the difference in measurements between the sensor pairs
dif1 = difference_columns(grouped_data,'B1->T','P4s1')
dif2 = difference_columns(grouped_data,'B2->T','P4s3')
dif3 = difference_columns(grouped_data,'B3->T','P4s5')

# remove the unnececery columns
df = grouped_data.drop(columns=['P1s1','P1s2','P2s1','P2s2','P3s1','P3s2','P4s1','P4s2','P4s3','P4s4','P4s5','P4s6'])

# calculate the location of the tag using all 6 measurements 
distance_sensor_middle = 7
df['1 final'] = round(df['B1->T'] + distance_sensor_middle - (dif1/2))
df['2 final'] = round(df['B2->T'] + distance_sensor_middle - (dif2/2))
df['3 final'] = round(df['B3->T'] + distance_sensor_middle - (dif3/2))
df['Location'] = df.apply(calculate_location, axis=1)

# add the a movement column and show the results
df = df.drop(columns=['B1->T','B2->T','B3->T'])
df = df[df['Location'].notna()]
df['Movement'] = df['Location'].diff().fillna(0).apply(
    lambda x: calculate_movement(x, (0, 0, 0)))
print(df)