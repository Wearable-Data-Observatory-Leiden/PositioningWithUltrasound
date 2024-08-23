import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

tracking_file_path = '/home/stanshands/Documents/scriptie/python/pandas/used_results1-12-04(3).csv'
data = pd.read_csv(tracking_file_path, header=None,  skiprows=1, delimiter=',')
data_subset = data.iloc[:, 1:5].dropna()
time_of_start = 129.4
index_to_keep = (data_subset.iloc[:, 0] > time_of_start).idxmax()
moving = data_subset.loc[index_to_keep:]
moving.iloc[:,0]= moving.iloc[:, 0].round(1)

moving.iloc[:,0]= moving.iloc[:, 0]-time_of_start
known_locations = [
    np.array([0, 0, 0]),   # Beacon 1
    np.array([300, 0, 0]),  # Beacon 2
    np.array([150, 260, 0])   # Beacon 3
]
def trilateration(distB1,distB2,distB3):
    r1 = distB1
    r2 = distB2
    r3 = distB3
    P1 = known_locations[0]
    P2 = known_locations[1]
    P3 = known_locations[2]

    p1 = np.array(P1)
    p2 = np.array(P2)
    p3 = np.array(P3)

    v1 = p2 - p1
    v2 = p3 - p1

    Xn = (v1) / np.linalg.norm(v1)

    tmp = np.cross(v1, v2)

    Zn = (tmp) / np.linalg.norm(tmp)

    Yn = np.cross(Xn, Zn)

    i = np.dot(Xn, v2)
    d = np.dot(Xn, v1)
    j = np.dot(Yn, v2)

    X = ((r1**2) - (r2**2) + (d**2)) / (2 * d)
    Y = (((r1**2) - (r3**2) + (i**2) + (j**2)) / (2 * j)) - ((i / j) * (X))
    Z1 = np.sqrt(max(0, r1**2 - X**2 - Y**2))
    Z2 = -Z1

    K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    K2 = P1 + X * Xn + Y * Yn + Z2 * Zn

    if K1[2] > 0:
        return tuple(K1)
    else:
        return tuple(K2)
def circle_location(r, a, b, y, Z):
    theta = 2 * np.pi * (Z / y)
    X = a + r * np.cos(theta)
    Y = b + r * np.sin(theta)
    return X, Y
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
def euclidean_distance_3D(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
r = 43 # radius of the circle
a, b = 150, 86.6  # Center point of the circle
y = 10.45  # Time it takes to complete a circle
Z = 5  # Time at which to compute the location
X, Y = circle_location(r, a, b, y, Z)


num_rows = len(moving) - 45
start_row = 45
rows_used = num_rows-start_row
print(f"Number of points used: {rows_used}")
start_data = moving.iloc[start_row:num_rows]

first_locations = []
actual_locations = []
for row_index, row in start_data.iterrows():
    time = row.iloc[0]
    B1 = row.iloc[1]
    B2 = row.iloc[2]
    B3 = row.iloc[3]
    place = trilateration(B1+6,B2+6,B3+6)
    first_locations.append(place)

total_height = sum(location[2] for location in first_locations)
avg_height = total_height / len(first_locations)
height = avg_height

best_offset = 9999
min_total_diff = 9999
for offset in range (0,10):
    total_diff = 0
    for idx, (row_index, row) in enumerate(start_data.iloc[:num_rows].iterrows()):
        time = row.iloc[0]
        X, Y = circle_location(r, a, b, y, time+offset)
        measured = first_locations[idx]
        diff = euclidean_distance_3D(measured,(X,Y,height))
        total_diff += diff
    if total_diff < min_total_diff:
        best_offset = offset
        min_total_diff = total_diff
print(f"Best offset: {best_offset}")
print(f"Total difference with best offset: {min_total_diff:.2f}")
best_precise_offset = 9999
min_total_diff = 9999
for decimal in range(0,200):
    adjustment = (decimal - 100) * 0.01
    precise_offset = best_offset + adjustment
    total_diff = 0
    for idx, (row_index, row) in enumerate(start_data.iloc[:num_rows].iterrows()):
        time = row.iloc[0]
        X, Y = circle_location(r, a, b, y, time+precise_offset)
        measured = first_locations[idx]
        diff = euclidean_distance_3D(measured,(X,Y,height))
        total_diff += diff
    if total_diff < min_total_diff:
        best_precise_offset = precise_offset
        min_total_diff = total_diff
print(f"Best precise offset: {best_precise_offset}")
print(f"Total difference with best offset: {min_total_diff:.2f}")
print(f"Average error = {min_total_diff/rows_used}")

for row_index, row in start_data.iterrows():
    time = row.iloc[0] + best_precise_offset
    X, Y = circle_location(r, a, b, y, time)
    actual_locations.append((X,Y))

x_values = [location[0] for location in first_locations]
y_values = [location[1] for location in first_locations]
plt.scatter(x_values, y_values,color='green')

x_true = [location[0] for location in actual_locations]
y_true = [location[1] for location in actual_locations]
plt.scatter(x_true, y_true,color='red')

for i in range(len(x_values)):
    plt.plot([x_values[i], x_true[i]], [y_values[i], y_true[i]], color='black')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Plot')

# Show the plot
plt.grid(True)
plt.show()
