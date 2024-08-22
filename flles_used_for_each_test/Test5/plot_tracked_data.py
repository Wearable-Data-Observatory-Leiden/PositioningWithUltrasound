import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

tracking_file_path = 'used_results1-12-04(3).csv'
data = pd.read_csv(tracking_file_path, header=None,  skiprows=1, delimiter=',')
data_subset = data.iloc[:, 2:5].dropna()
known_locations = [
    np.array([0, 0, 0]),   # Beacon 1
    np.array([0, 300, 0]),  # Beacon 2
    np.array([260, 150, 0])   # Beacon 3
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
    
first_place = trilateration(214,131,200)
print(f"First place: {first_place}")
all_locations = []
all_locations_adjusted = []
for row_index, row in data_subset.iterrows():
    B1 = row.iloc[0]
    B2 = row.iloc[1]
    B3 = row.iloc[2]
    place = trilateration(B1,B2,B3)
    place_adjusted = trilateration(B1+6,B2+6,B3+6)
    all_locations.append(place)
    all_locations_adjusted.append(place_adjusted)
    print(f"Place: {place}")

# Plot the known locations in a different color
known_x_values = [location[0] for location in known_locations]
known_y_values = [location[1] for location in known_locations]
plt.scatter(known_x_values, known_y_values, color='red', label='Known Locations', s=100)  

# Annotate the known points with labels
beacon_labels = ['Beacon 1', 'Beacon 2', 'Beacon 3']
for i, label in enumerate(beacon_labels):
    plt.text(known_x_values[i], known_y_values[i], label, fontsize=12, ha='right', va='bottom')

# plot the calculated locations
x_values = [location[0] for location in all_locations]
y_values = [location[1] for location in all_locations]
plt.scatter(x_values, y_values,color='yellow')

# plot the adjusted calculated locations
x_values_adjusted = [location[0] for location in all_locations_adjusted]
y_values_adjusted = [location[1] for location in all_locations_adjusted]
plt.scatter(x_values_adjusted, y_values_adjusted,color='green')

# Add a circle in the center
circle_center = (86.6,150)
circle_diameter = 86
circle_radius = circle_diameter / 2
circle = Circle(circle_center, circle_radius, edgecolor='r', facecolor='none')
plt.gca().add_patch(circle)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Plot')
plt.xlim(-15, 300)
plt.ylim(-15, 300)

# Show the plot
plt.grid(True)
plt.show()

