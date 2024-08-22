import pandas as pd
import numpy as np


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
def calibration_indexes(df, num_steps=36):
    indexes = []
    step_count = 0
    columns_with_space = df.columns[df.iloc[0] == '']
    for index, row in df.iterrows():
        current = df.columns[row == '']
        if not current.equals(columns_with_space):
            indexes.append(index)
            columns_with_space = current
            step_count += 1
        if step_count == num_steps: break
    return indexes
        

order = [['P1s1', 'P4s1'], ['P1s2', 'P4s1'], ['P2s1', 'P4s3'], ['P2s2', 'P4s3'], ['P3s1', 'P4s5'], ['P3s2', 'P4s5'], 
         ['P1s1', 'P4s2'], ['P1s2', 'P4s2'], ['P2s1', 'P4s4'], ['P2s2', 'P4s4'], ['P3s1', 'P4s6'], ['P3s2', 'P4s6'], 
         ['P1s1', 'P4s3'], ['P1s2', 'P4s3'], ['P2s1', 'P4s5'], ['P2s2', 'P4s5'], ['P3s1', 'P4s1'], ['P3s2', 'P4s1'], 
         ['P1s1', 'P4s4'], ['P1s2', 'P4s4'], ['P2s1', 'P4s6'], ['P2s2', 'P4s6'], ['P3s1', 'P4s2'], ['P3s2', 'P4s2'], 
         ['P1s1', 'P4s5'], ['P1s2', 'P4s5'], ['P2s1', 'P4s1'], ['P2s2', 'P4s1'], ['P3s1', 'P4s3'], ['P3s2', 'P4s3'], 
         ['P1s1', 'P4s6'], ['P1s2', 'P4s6'], ['P2s1', 'P4s2'], ['P2s2', 'P4s2'], ['P3s1', 'P4s4'], ['P3s2', 'P4s4']]

# Main code

# read in the data and find the moment where the beacons are initialized
# check if the distance measured is 3 meter and save this part of the csv file as test1_beacon_init
csv_file_path = 'test_data-12-04_final_1.csv'
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()
data_beacons = read_data(csv_file_path, header_line)
beacon2_index, beacon3_index = calculate_beacon_indices(data_beacons)
avg_P1S1, avg_P2S2, avg_P1S2, avg_P2S1, avg_P3S1, avg_P3S2 = calculate_averages(data_beacons, beacon2_index, beacon3_index)
# print("The beacons should be 3 meters apart from each other, here are the measured distances:")
# print(f"Beacon 1, tag 1(1->2): {avg_P1S1}")
# print(f"Beacon 2, tag 1(2->1): {avg_P2S2}")
# print(f"Beacon 1, tag 2(1->3): {avg_P1S2}")
# print(f"Beacon 3, tag 1(3->1): {avg_P3S1}")
# print(f"Beacon 2, tag 2(2->3): {avg_P2S1}")
# print(f"Beacon 3, tag 2(3->2): {avg_P3S2}")
data_beacons_slice = data_beacons.iloc[:beacon3_index]
data_beacons_slice.to_csv('test1_beacon_init-12-04.csv') 


# read in the data after beacon initialization and prepare the data in compressed rows
data = pd.read_csv(csv_file_path, skiprows=beacon3_index, header=None, delimiter=',', usecols=range(13))
headers = header_line.split(',')[:13]
data.columns = headers
data = data.round(1)
data = data.fillna('')
grouped_data = data.groupby(data.index // 4).agg({col: aggregate_non_empty for col in data.columns}).reset_index()

# get the indexes of the calibration steps and save that to a csv file
indexes = calibration_indexes(grouped_data)
data_sensor_pairs = grouped_data.iloc[:indexes[35]]
data_sensor_pairs.to_csv('test1_sensors_init-12-04.csv')
data_sensor_pairs_filtered = data_sensor_pairs


sensor_pairs = []
for x in range(0,36):
    if (x == 0): start_row = 0      # only required for the first iteration of this loop
    else: start_row = indexes[x-1]  # find start   
    end_row = indexes[x]            # find end
    beacon = order[x][0]            # find the sensors in the order list
    tag = order[x][1]
    data_sensor_pairs_filtered.loc[start_row:end_row] = data_sensor_pairs_filtered.loc[start_row:end_row, [beacon, tag]]
    max_count = 0

    for q in range(start_row, end_row): # only take the last 25 measurements
        if q+25 < end_row:
            data_sensor_pairs_filtered.loc[q, beacon] = np.nan
            data_sensor_pairs_filtered.loc[q, tag] = np.nan

    for y in range(100,220):            # get the most likely distance
        count = 0
        for value in data_sensor_pairs_filtered.loc[start_row:end_row,beacon]:
            try:
                value = float(value) 
            except ValueError:
                continue
            if y - 10 < value and value < y + 10:  
                count += 1
        if count > max_count:
            max_count = count
            avg = y
    # if x == 1:  # hardcode faulty measurements for this test
    #     avg = 184

    for z in range(start_row, end_row): # remove data sets far from expected distance
        if not (avg - 10 < data_sensor_pairs_filtered.loc[z, beacon] < avg + 10):
            data_sensor_pairs_filtered.loc[z, beacon] = np.nan
            data_sensor_pairs_filtered.loc[z, tag] = np.nan

    beacon_filtered = data_sensor_pairs_filtered.loc[start_row:end_row-1, beacon]
    tag_filtered = data_sensor_pairs_filtered.loc[start_row:end_row-1, tag]

    difference = beacon_filtered.mean() - tag_filtered.mean()
    average = (beacon_filtered.mean() + tag_filtered.mean())/2

    sensor_pairs.append([beacon,tag, round(difference,1),round(beacon_filtered.mean(),1),round(tag_filtered.mean(),1),round(average,1)])



data_sensor_pairs.to_csv('test1_sensors_init_filtered-12-04.csv')

   
# difference
row_names = sorted(set(pair[1] for pair in sensor_pairs))
col_names = sorted(set(pair[0] for pair in sensor_pairs))
df_sensors = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_sensors.at[pair[1], pair[0]] = pair[2]
#print("Difference between the beacon sensor and the tag sensors facing each other")
#print(df_sensors)
df_sensors.to_csv('results1_difference-12-04.csv')

# avg_beacon
df_beacon_mean = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_beacon_mean.at[pair[1], pair[0]] = pair[3]
#print("Distance measured by the beacon")
#print(df_beacon_mean)
df_beacon_mean.to_csv('results1_beacons_mean-12-04.csv')

# avg_tag
df_tag_mean = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_tag_mean.at[pair[1], pair[0]] = pair[4]
#print("Distance measured by the tag")
#print(df_tag_mean)
df_tag_mean.to_csv('results1_tag_mean-12-04.csv')

# beacon+tag/2
df_avg = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_avg.at[pair[1], pair[0]] = pair[5]
#print("Avg distance of the tag measurement and the beacon measurement")
#print(df_avg)
df_avg.to_csv('results1_avg_beacon+tag-12-04.csv')



# this is where the tracking of the tag starts
row_start_of_tracking = 8029
data_tracking = pd.read_csv(csv_file_path, skiprows=row_start_of_tracking, header=None, delimiter=',', usecols=range(13))
data_tracking.to_csv('test1_tracking-12-04.csv')
headers = header_line.split(',')[:13]
data_tracking.columns = headers
data_tracking = data_tracking.fillna('')
grouped_tracking = data_tracking.groupby(data_tracking.index // 4).agg({col: aggregate_non_empty for col in data_tracking.columns}).reset_index()

# Make new columns that compresses the data of the 2 sensors on each beacon
# if measurement is equal to beacon initialization, last measurement is used 
grouped_tracking = add_column_B1T(grouped_tracking)
grouped_tracking = add_column_B2T(grouped_tracking)
grouped_tracking = add_column_B3T(grouped_tracking)

# Calculate the difference in measurements between the sensor pairs
dif1 = difference_columns(grouped_tracking,'B1->T','P4s1')
dif2 = difference_columns(grouped_tracking,'B2->T','P4s3')
dif3 = difference_columns(grouped_tracking,'B3->T','P4s5')

# remove the unnececery columns
df = grouped_tracking.drop(columns=['P1s1','P1s2','P2s1','P2s2','P3s1','P3s2','P4s1','P4s2','P4s3','P4s4','P4s5','P4s6'])

# calculate the location of the tag using only the beacon measurements
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