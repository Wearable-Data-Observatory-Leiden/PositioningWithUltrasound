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

# Define trilateration function for the first set of columns
def calculate_location(row):
    r1, r2, r3 = pd.to_numeric(row[1]), pd.to_numeric(row[3]), pd.to_numeric(row[5])  # Extract r1, r2, r3 from the row
    return trilateration(r1, r2, r3)

# Apply the calculate_location function to each row and create a new column
grouped_data['Location1'] = grouped_data.apply(calculate_location, axis=1)

# Define trilateration function for the second set of columns
def calculate_location2(row):
    r1, r2, r3 = pd.to_numeric(row[2]), pd.to_numeric(row[4]), pd.to_numeric(row[6])  # Extract r1, r2, r3 from the row
    return trilateration(r1, r2, r3)

# Apply the calculate_location2 function to each row and create the second new column
grouped_data['Location2'] = grouped_data.apply(calculate_location2, axis=1)

# Display the grouped DataFrame
print(grouped_data.head())
