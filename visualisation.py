import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import animation 
import queue


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
    r1, r2, r3 = pd.to_numeric(row[11]), pd.to_numeric(row[12]), pd.to_numeric(row[13])  # Extract r1, r2, r3 from the row
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

    numeric_values_column1 = pd.to_numeric(values_column1, errors='coerce')
    avg_B = numeric_values_column1.mean()

    numeric_values_column2 = pd.to_numeric(values_column2, errors='coerce')
    avg_T = numeric_values_column2.mean()

    return avg_B - avg_T
def aggregate_non_empty(series):
    return series[series != ''].iloc[-1] if any(series != '') else ''
def extract_points(row):
    point_array = row['Location']
    if isinstance(point_array, np.ndarray) and point_array.shape == (3,):
        return tuple(point_array.tolist())  # Convert array to tuple
    else:
        return ([0,0,0])  # Return an empty tuple if the point is not valid
def init():
    sc._offsets3d = ([], [], [])
    return sc,
def update(frame):
    frame = frame + 1
    if frame < number_of_points_shown:
        current_points = all_points_over_time[:frame]
    else:
        current_points = all_points_over_time[frame-number_of_points_shown:frame]
    if current_points:
        x, y, z = zip(*current_points)
        sc._offsets3d = (x, y, z)
    else:
        sc._offsets3d = ([], [], [])
    for txt in ax.texts:
        txt.remove()
    ax.text(0.95, 0.95, 0, f'Frame: {frame}', transform=ax.transAxes, ha='right', va='top', fontsize=12)  # Specify the text and add fontsize
    return sc,

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

# calculate the location of the tag using only the beacon measurements
distance_sensor_middle = 7
df['1 final'] = round(df['B1->T'] + distance_sensor_middle - (dif1/2))
df['2 final'] = round(df['B2->T'] + distance_sensor_middle - (dif2/2))
df['3 final'] = round(df['B3->T'] + distance_sensor_middle - (dif3/2))
df['Location'] = df.apply(calculate_location, axis=1)

# check to see if the dataframe is printed correctly
new_df = df.drop(columns=['B1->T','B2->T','B3->T'])
new_df = new_df[new_df['Location'].notna()]
print(new_df)

# grab all the location points from the dataframe
all_points_over_time = new_df.apply(extract_points, axis=1).tolist()

# # check if the points are working, only used for debugging
# first_5_points = all_points_over_time[:5] 
# for point in first_5_points:
#     print(f"Point: {point}  type: {type(point)}")

# built the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim([0, 350])
ax.set_ylim([0, 350])
ax.set_zlim([-180, 0])
ax.autoscale(enable=False)
number_of_points_shown = 5
data_queue = queue.Queue()

# Create and show animation
animation = FuncAnimation(fig, update, frames=len(all_points_over_time), init_func=init, interval=500, blit=False)
plt.show()