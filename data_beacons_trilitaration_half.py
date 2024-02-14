import pandas as pd
import numpy as np
# trilitaration function that calculates the location the current point
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

def calculate_movement(point1, point2):
    if isinstance(point1, int):
        # If point1 is an integer, return (0, 0, 0)
        return (0, 0, 0)
    else:
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distance

# Define trilateration function for the first set of columns
def calculate_location(row):
    r1, r2, r3 = pd.to_numeric(row[1]), pd.to_numeric(row[3]), pd.to_numeric(row[5])  # Extract r1, r2, r3 from the row
    return trilateration(r1, r2, r3)

# Specify the path to your CSV file
csv_file_path = 'test_data.csv'

# Read the header from the CSV file
with open(csv_file_path, 'r') as file:
    header_line = file.readline().strip()

# Read the CSV file with the determined delimiter and specifying the number of columns to read
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

# Apply the calculate_location function to each row and create a new column
grouped_data['Location1'] = grouped_data.apply(calculate_location, axis=1)

# Selecting desired columns
reduced_data = grouped_data[['P1s1', 'P2s1', 'P3s1', 'Location1']]

# Filter out rows where 'Location1' is not None
reduced_data = reduced_data[reduced_data['Location1'].notna()]

# Define a custom function to fill missing values with tuples
def fillna_with_tuple(series, value):
    return series.fillna(pd.Series([value] * len(series), index=series.index))

# Calculate movement between consecutive points
reduced_data['Movement'] = reduced_data['Location1'].diff().fillna(0).apply(
    lambda x: calculate_movement(x, (0, 0, 0)))

# Fill missing values in the 'Movement' column with tuples
reduced_data['Movement'] = fillna_with_tuple(reduced_data['Movement'], (0, 0, 0))

# Display the DataFrame with the new 'Movement' column
print(reduced_data)