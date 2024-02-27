import pandas as pd

def read_data(csv_file_path, header_line):
    # Read the CSV file with the determined delimiter and specifying the number of columns to read
    data = pd.read_csv(csv_file_path,skiprows=1, header=None, delimiter=',', usecols=range(13))
    # Get the column names from the first row of the DataFrame
    headers = header_line.split(',')[:13]
    data.columns = headers
    data = data.fillna('')
    return data
def aggregate_non_empty(series):
    return series[series != ''].iloc[-1] if any(series != '') else ''
order = [['P1s1', 'P4s1'], ['P1s2', 'P4s1'], ['P2s1', 'P4s3'], ['P2s2', 'P4s3'], ['P3s1', 'P4s5'], ['P3s2', 'P4s5'], 
         ['P1s1', 'P4s2'], ['P1s2', 'P4s2'], ['P2s1', 'P4s4'], ['P2s2', 'P4s4'], ['P3s1', 'P4s6'], ['P3s2', 'P4s6'], 
         ['P1s1', 'P4s3'], ['P1s2', 'P4s3'], ['P2s1', 'P4s5'], ['P2s2', 'P4s5'], ['P3s1', 'P4s1'], ['P3s2', 'P4s1'], 
         ['P1s1', 'P4s4'], ['P1s2', 'P4s4'], ['P2s1', 'P4s6'], ['P2s2', 'P4s6'], ['P3s1', 'P4s2'], ['P3s2', 'P4s2'], 
         ['P1s1', 'P4s5'], ['P1s2', 'P4s5'], ['P2s1', 'P4s1'], ['P2s2', 'P4s1'], ['P3s1', 'P4s3'], ['P3s2', 'P4s3'], 
         ['P1s1', 'P4s6'], ['P1s2', 'P4s6'], ['P2s1', 'P4s2'], ['P2s2', 'P4s2'], ['P3s1', 'P4s4'], ['P3s2', 'P4s4']]
last_calibration_row = 4038

# open the file
csv_file_path = 'test_data.csv'
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()
data = read_data(csv_file_path, header_line)

# read only till the calibrations and compress the rows
data = data.iloc[:last_calibration_row-2]
grouped_data = data.groupby(data.index // 4).agg({col: aggregate_non_empty for col in data.columns}).reset_index()
df = grouped_data.drop(columns=['index','time'])

# find the indexes for each new calibration step using the blankspaces
indexes = []
columns_with_space = df.columns[df.iloc[0] == '']
for index, row in df.iterrows():
    current = df.columns[row == '']
    if not current.equals(columns_with_space):
        indexes.append(index)
        columns_with_space = current

# add the last index number so that you have all 36 options
last_index_number = len(df.index) - 1
indexes.append(last_index_number)

# for each step in the calibration find the 2 groups of measurements
# the key things to get from these measurements are the following:
# avg_beacon, avg_tag, beacon-tag, beacon+tag/2
sensor_pairs = []
range_around_median_still_valid = 10
for x in range(0,36):
    if (x == 0): start_row = 0      # only required for the first iteration of this loop
    else: start_row = indexes[x-1]  # find start   
    end_row = indexes[x]            # find end
    beacon = order[x][0]            # find the sensors in the order list
    tag = order[x][1]
    data_beacon = df.loc[start_row:end_row-1, beacon]
    filtered_data_beacon = data_beacon[(data_beacon >= data_beacon.median() - range_around_median_still_valid) & 
                                    (data_beacon <= data_beacon.median() + range_around_median_still_valid)]
    data_tag = df.loc[start_row:end_row-1, tag]
    filtered_data_tag = data_tag[(data_tag >= data_tag.median() - range_around_median_still_valid) & 
                                    (data_tag <= data_tag.median() + range_around_median_still_valid)]    
    difference = filtered_data_beacon.mean() - filtered_data_tag.mean()
    avg = (filtered_data_beacon.mean() + filtered_data_tag.mean())/2
    sensor_pairs.append([beacon,tag, round(difference,1),round(filtered_data_beacon.mean(),1),round(filtered_data_tag.mean(),1),round(avg,1)])

# set the collected data in new dataframes so they are easily interpreted

# difference
row_names = sorted(set(pair[1] for pair in sensor_pairs))
col_names = sorted(set(pair[0] for pair in sensor_pairs))
df_sensors = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_sensors.at[pair[1], pair[0]] = pair[2]
print("Difference between the beacon sensor and the tag sensors facing each other")
print(df_sensors)

# avg_beacon
df_beacon_mean = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_beacon_mean.at[pair[1], pair[0]] = pair[3]
print("Distance measured by the beacon")
print(df_beacon_mean)

# avg_tag
df_tag_mean = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_tag_mean.at[pair[1], pair[0]] = pair[4]
print("Distance measured by the tag")
print(df_tag_mean)

# beacon+tag/2
df_avg = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_avg.at[pair[1], pair[0]] = pair[5]
print("Avg distance of the tag measurement and the beacon measurement")
print(df_avg)