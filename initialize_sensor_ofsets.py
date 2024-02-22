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
csv_file_path = 'test_data.csv'
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()
data = read_data(csv_file_path, header_line)
data = data.iloc[:last_calibration_row-2]
grouped_data = data.groupby(data.index // 4).agg({col: aggregate_non_empty for col in data.columns}).reset_index()
df = grouped_data.drop(columns=['index','time'])
indexes = []
columns_with_space = df.columns[df.iloc[0] == '']
for index, row in df.iterrows():
    current = df.columns[row == '']
    if not current.equals(columns_with_space):
        indexes.append(index)
        columns_with_space = current
sensor_pairs = []
for x in range(0,35):
    row = indexes[x]
    beacon = order[x][0]
    tag = order[x][1]
    data_beacon = df.loc[row-5:row-1, beacon]
    data_tag = df.loc[row-5:row-1, tag]
    difference = data_beacon.mean() - data_tag.mean()
    avg = (data_beacon.mean() + data_tag.mean())/2
    sensor_pairs.append([beacon,tag, round(difference,1),round(data_beacon.mean(),1),round(data_tag.mean(),1),round(avg,1)])
row_names = sorted(set(pair[1] for pair in sensor_pairs))
col_names = sorted(set(pair[0] for pair in sensor_pairs))
df_sensors = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_sensors.at[pair[1], pair[0]] = pair[2]
print("Difference between the beacon sensor and the tag sensors facing each other")
print(df_sensors)
df_beacon_mean = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_beacon_mean.at[pair[1], pair[0]] = pair[3]
print("Distance measured by the beacon")
print(df_beacon_mean)
df_tag_mean = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_tag_mean.at[pair[1], pair[0]] = pair[4]
print("Distance measured by the tag")
print(df_tag_mean)
df_avg = pd.DataFrame(index=row_names, columns=col_names)
for pair in sensor_pairs:
    df_avg.at[pair[1], pair[0]] = pair[5]
print("Avg distance of the tag measurement and the beacon measurement")
print(df_avg)