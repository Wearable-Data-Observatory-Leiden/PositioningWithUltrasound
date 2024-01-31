import serial
import serial.tools.list_ports
import subprocess
import time
import math
import threading
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import animation 
from matplotlib.animation import FuncAnimation
import queue
from queue import Queue
from pynput import keyboard
import numpy as np
# all the global variables get created and initialized
def initialize_global_variables():
    global all_points_over_time, fig, ax, sc, number_of_points_shown, port_and_sensors_list
    global known_locations, avg_distances, threshold, max_distance, start_time, sensor_middle
    global port0,port1,port2,port3, beacon_fail, csv_writer, max_sensors
    global data_queue, time_to_cm, ser1, ser2, ser3, ser4
    
    all_points_over_time = [[]]     # this is where all the points are stored
    number_of_points_shown = 5      # how many points are plotted at the same time
    
    fig = plt.figure()              # Create a 3D plot
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], c='r', marker='o')
    ax.autoscale(enable=False)
    
    ax.set_xlabel('X Label')        # Set labels for each axis
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.set_xlim([0, 250])           # Set axis limits in cm
    ax.set_ylim([0, 250])
    ax.set_zlim([0, 100])
    
    beacon1 = np.array([0, 0, 0])   # beacon locations
    beacon2 = np.array([202, 0, 0])
    beacon3 = np.array([160, 160, 0])    
    known_locations = np.array([beacon1, beacon2, beacon3])

    avg_distances = np.array([0, 0, 0]) # distances for triangulation

    beacon_fail = np.array([False,False,False])
    
    threshold = 20                  # % of how far the measurements are allowed to be apart   
    max_distance = 50               # maximum distance 2 points can be away from each other after 2 measurements

    start_time = time.time()        # Used for timestamps

    board_type = "uno"              # since we are working with the same port and sensors
    port0 = '/dev/ttyACM0'
    port1 = '/dev/ttyACM1'
    port2 = '/dev/ttyACM2'
    port3 = '/dev/ttyACM3'
    sketch_path = "/home/stanshands/Arduino/Ysensors/2sensors/2sensors.ino"
    sketch_path2 = "/home/stanshands/Arduino/Ysensors/3sensors/3sensors.ino"
    port_and_sensors_list = [('/dev/ttyACM0', 2),('/dev/ttyACM1', 2),('/dev/ttyACM2', 2),('/dev/ttyACM3', 3)]
    max_sensors = 3                 # maximum amount of sensors connected to 1 arduino

    upload_arduino_sketch(sketch_path, board_type, port0)
    upload_arduino_sketch(sketch_path, board_type, port1)
    upload_arduino_sketch(sketch_path, board_type, port2)
    upload_arduino_sketch(sketch_path2, board_type, port3)

    ser1 = serial.Serial(port0, baudrate = 115200, timeout=1)    # set up to recieve data from arduino
    ser2 = serial.Serial(port1, baudrate = 115200, timeout=1)
    ser3 = serial.Serial(port2, baudrate = 115200, timeout=1)
    ser4 = serial.Serial(port3, baudrate = 115200, timeout=1)

    time_to_cm =  343 / 10000       # conversion rate in case of normal temperature

    sensor_middle = 4               # distance between the sensor and the middle point of the tag 

    data_queue = queue.Queue()

    csv_writer = None
# trilitaration function that calculates the location the current point
def trilateration():
    r1 = avg_distances[0]
    r2 = avg_distances[1]
    r3 = avg_distances[2]
    P1 = known_locations[0]
    P2 = known_locations[1]
    P3 = known_locations[2]

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

    X = ((r1**2)-(r2**2)+(d**2))/(2*d)
    Y = (((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(X))
    Z1 = np.sqrt(max(0, r1**2-X**2-Y**2))
    Z2 = -Z1

    K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    K2 = P1 + X * Xn + Y * Yn + Z2 * Zn
 
    if K1[2] > 0:
        return K1
    elif K2[2] > 0:
        return K2
    else:
        print("Error: Trilateration failed. Target point cannot be determined.")
        return None
# depending on the amount of ports and sensors creates an apropriate header
def generate_csv_header(port_and_sensors_list):
    header = ["time"]
    for port_num, (port, sensor_count) in enumerate(port_and_sensors_list, start=1):
        for sensor_num in range(1, sensor_count + 1):
            header.append(f"P{port_num}s{sensor_num}")
    return header
# create a csv file and adds the correct header
def create_csv_file(header):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, "test_data.csv")

    with open(file_path, "w", newline='') as data_file:
        csv_writer = csv.writer(data_file)
        csv_writer.writerow(header)
    return file_path
# checks if 2 numbers are within the threshold of each other
def within_threshold(a,b):
    if a == 0 or b == 0: return False, None
    percentage_difference = abs(a - b) / max(a, b) * 100
    if percentage_difference <= threshold:
        return True, (a + b) / 2 
    else:
        return False, None
# uploads the correct sketch to the arduino
def upload_arduino_sketch(sketch_path, board_type, port):
    cli_command = "/home/stanshands/bin/arduino-cli"
    fqbn = f"arduino:avr:{board_type}"
    subprocess.run([cli_command, "compile", "--fqbn", fqbn, sketch_path], capture_output=True, text=True)
    subprocess.run([cli_command, "upload", "--fqbn", fqbn, "--port", port, sketch_path])
# keyboard listener
def on_key_press(key, stop_event,beacon2_stop,beacon3_stop,tag_stop):
    try:
        letter = key.char
        data_queue.put(letter)
        if letter == "s":
            stop_event.set()
        if letter == "t":
            beacon2_stop.set()
        if letter == "a":
            beacon3_stop.set()
        if letter == "n":
            tag_stop.set()
    except AttributeError:
        pass
# The current beacon is measuring the wrong sensor
def switch_beacon(beacon_nr):
    print(f"Now you should switch to a different sensor on beacon {beacon_nr+1}")
    # still need to figure out what logic this needs since 
# gets the measured values and checks which ones are useable
def filter_inputs(b1, t1, b2, t2, b3, t3):
    connections = []
    for idx, (beacon, tag) in enumerate([(b1, t1), (b2, t2), (b3, t3)]):
        if 10 <= (beacon - tag) <= 20:
            connections.append("++")
            avg_distances[idx] = (beacon + tag) / 2
            beacon_fail[idx] = False
        elif 10 <= (beacon - avg_distances[idx]) <= 20:
            connections.append("+-")
            avg_distances[idx] = beacon - 5  # -5 since it fires first
            beacon_fail[idx] = False
        elif 10 <= (tag - avg_distances[idx]) <= 20:
            connections.append("-+")
            avg_distances[idx] = tag + 5  # +5 since it fires later
            if (beacon_fail[idx] == True): 
                switch_beacon(idx)
            else:
                beacon_fail[idx] = True
        else:
            connections.append("--")
    print(f"Connections: {connections}")
# checks if the new calculated point is possible
def check_possible_point(new_point):
    if not isinstance(new_point, np.ndarray) or new_point.shape != (3,):
        print(f"Invalid variable, not a point: {new_point}")
        return False
    else:
        return True
    # this part of the code is not working yet, will do this part later
    numeric_points = [point for point in all_points_over_time if all(isinstance(val, (int, float)) for val in point)]

    if not numeric_points:
        print("No numeric points found in all_points_over_time.")
        return True

    last_point = np.array(numeric_points[-1], dtype=float)  # Convert to float array

    difference_vector = np.array(new_point) - np.array(last_point)
    distance = np.linalg.norm(difference_vector)
    if (distance>max_distance): 
        print("Distance from this point and last is too big")
        return False
    return True
# update the plot when a new point gets added
def update_plot(frame):
    while not data_queue.empty():
        item = data_queue.get()
        if isinstance(item, np.ndarray) and item.shape == (3,):
            new_point = tuple(item)
            print(f"New point: {new_point}.")
        else:
            print(f"Invalid item in the queue: {item}")
            continue
        all_points_over_time[0].append(new_point)
        current_points = all_points_over_time[0][-number_of_points_shown:]
        x, y, z = zip(*current_points)
        sc._offsets3d = (x, y, z)
        sc_static = ax.scatter([], [], [], color='blue')  # The beacons
        sc_static._offsets3d = tuple(map(tuple, known_locations.T))
        fig.canvas.flush_events()
# plan for later on when the check if the new point is valid has been figured out
def update_plot_with_text(frame):
    while not data_queue.empty():
        item = data_queue.get()                         # get the new point form queue
        new_point = tuple(item)                         # put it in correct form
        all_points_over_time[0].append(new_point)       # collection of all point over time
        current_points = all_points_over_time[0][-number_of_points_shown:]
        x, y, z = zip(*current_points)                  # prepare the points to be printed
        sc._offsets3d = (x, y, z)
        sc_static = ax.scatter([], [], [], color='blue')# print the beacons in a different color
        sc_static._offsets3d = tuple(map(tuple, known_locations.T))
                                                        # print the distances between beacons and tag
        ax.text(0.95, 0.95, f"Distance1: {avg_distances[0]}\nDistance2: {avg_distances[1]}\nDistance3: {avg_distances[2]}",
        horizontalalignment='right', verticalalignment='top',   
        transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
 
        fig.canvas.flush_events()                       # make sure the new plot appears on screen
# gets serial and sensor numbers, then sends and recieves the data from the arduino
def communicate_and_process(ser_send, ser_receive, sensornr_send, sensornr_receive, port_nr_1, port_nr_2):
    row1 = [''] * 24
    row2 = [''] * 24    
    csv_entry1 = int(sensornr_send)
    csv_entry2 = int(sensornr_receive)
    for x in range(3):
        if (port_nr_1>x): csv_entry1 += port_and_sensors_list[x][1]
        if (port_nr_2>x): csv_entry2 += port_and_sensors_list[x][1]
    while True:
        ser_send.write(sensornr_send.encode('utf-8'))
        ser_receive.write(sensornr_receive.encode('utf-8'))
        time.sleep(0.1)
        if ser_send.in_waiting:   # wait for data from the arduino
            data_send = ser_send.readline().decode('utf').rstrip('\n')
            distance_send = data_send.split(":")
            dist_send = float(distance_send[1].strip()) * time_to_cm
            row1[csv_entry1] = dist_send
            csv_writer.writerow(row1)
        if ser_receive.in_waiting:   # wait for data from the arduino
            data_receive = ser_receive.readline().decode('utf').rstrip('\n')
            distance_receive = data_receive.split(":")
            dist_receive = float(distance_receive[1].strip()) * time_to_cm
            row2[csv_entry2] = dist_receive
            csv_writer.writerow(row2)
        yield dist_send, dist_receive
# finds the sensors numbers of beacons that are facing each other
def find_sensor(ser_send, ser_receive, port_nr_1, port_nr_2):
    pairs = []
    for index in range(1, max_sensors):
        sensornr_send = str(index)
        sensornr_receive = str(index)
        communicator = communicate_and_process(ser_send, ser_receive, sensornr_send, sensornr_receive, port_nr_1, port_nr_2)
        dist1, dist2 = next(communicator)
        for other_pair in pairs:
            if 10 <= dist1 - other_pair[0] <= 20: return index,other_pair[2]
            if 10 <= other_pair[1] - dist2 <= 20: return other_pair[2],index
        pairs.append((dist1, dist2, index))
    print ("No valid pairs found")
    return None, None
# measure the distance between beacon 1 and 2 and set location beacon 2
def initalize_beacon2(beacon2_stop):
    sensornrB1, sensornrB2 = find_sensor(ser1,ser2,0,1)
    distance_history = []  # Stores all values of dist1
    communicator = communicate_and_process(ser1, ser2, sensornrB1, sensornrB2,0,1)
    while not beacon2_stop.is_set():
        dist1, dist2 = next(communicator)
        distance_history.append(dist1)
        distance_history.append(dist2)
        print(f"Data 1->2: {int(dist1)} ")
        print(f"Data 2->1: {int(dist2)} ")
    distance_history = distance_history[-10:]       # you could do an outlier check to improve accuracy
    print(f"Last distance beacon 1 -> 2: {dist1}, 2 -> 1: {dist2}")
    known_locations[1] = [0,sum(distance_history) / len(distance_history),0]
    print(f"Location beacon 2: {known_locations[1]}")
# measure the distance between beacon 3 and other beacons beacon and sets location beacon 3
def initalize_beacon3(beacon3_stop):
    sensornrB13, sensornrB31 = find_sensor(ser1,ser3,0,2)
    sensornrB23, sensornrB32 = find_sensor(ser2,ser3,1,2)
    distance_history13 = [] 
    distance_history23 = [] 
    communicator1 = communicate_and_process(ser1, ser3, sensornrB13, sensornrB31,0,2)
    communicator2 = communicate_and_process(ser2, ser3, sensornrB23, sensornrB32,1,2)
    while not beacon3_stop.is_set():
        dist13, dist31 = next(communicator1)
        distance_history13.append(dist13)
        distance_history13.append(dist31)
        dist23, dist32 = next(communicator2)
        distance_history23.append(dist23)
        distance_history23.append(dist32)
        print(f"Data 1->3: {int(dist13)}, 3->1: {int(dist31)}")
        print(f"Data 2->3: {int(dist23)}, 3->2: {int(dist32)}")
    print(f"Last distance beacon 1 -> 3: {dist13}, 3->1: {dist31}")
    avg_13 = sum(distance_history13) / len(distance_history13) # here outliers could be spotted
    print(f"Average distance beacon 1 and 3: {avg_13}")
    print(f"Distance beacon 2 -> 3: {dist23}, 3->2: {dist32}")
    avg_23 = sum(distance_history23) / len(distance_history23) # outliers here aswell
    print(f"Average distance beacon 2 and 3: {avg_23}")
    beacon3_x = ((avg_13 ** 2) - (avg_23 ** 2) + (known_locations[1][1] ** 2)) / (2*known_locations[1][1])
    beacon3_y = math.sqrt((avg_13 ** 2) - (beacon3_x ** 2))
    known_locations[2] = [beacon3_y,beacon3_x,0]
    print(f"Final locations, beacon 2: {known_locations[1]}, beacon 3 {known_locations[2]}")
    #This part is probably temporary just to get an estimate in case the connection between tag and beacon isn't formed
    # Calculate centroid (middle point) of three points
    centroid = np.mean(known_locations, axis=0)
    # Calculate distances between centroid and each beacon
    distances_to_centroid = [np.linalg.norm(centroid - beacon) for beacon in known_locations]
    global avg_distances
    avg_distances = distances_to_centroid
# sends the information of which sensors should trigger to the arduinos to find the tag location
def find_tag(tag_stop):
    t1 = "1"
    t2 = "2"
    t3 = "3"
    b1 = "2"
    b2 = "2"
    b3 = "2"
    communicator1 = communicate_and_process(ser1, ser4, b1, t1,0,3)
    communicator2 = communicate_and_process(ser2, ser4, b2, t2,1,3)
    communicator3 = communicate_and_process(ser3, ser4, b3, t3,2,3)
    while not tag_stop.is_set():
        dist1t, distt1 = next(communicator1)
        dist2t, distt2 = next(communicator2)
        dist3t, distt3 = next(communicator3)
        filter_inputs(dist1t, distt1,dist2t, distt2,dist3t, distt3)
        new_point = trilateration()
        possible = check_possible_point(new_point)
        if (possible): data_queue.put(new_point)
# starts all threads
def start_all_threads():
    stop_event = threading.Event()  # the signal that the threads shoulds stop
    beacon2_stop = threading.Event()# the signal that beacon2 is in position
    beacon3_stop = threading.Event()# the signal that beacon2 is in position
    tag_stop = threading.Event()    # the signal that the tag location is no longer required
    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, stop_event,beacon2_stop,beacon3_stop,tag_stop))
    keyboard_listener.start()
    time.sleep(2)
    initalize_beacon2(beacon2_stop) # find the locations of the beacons
    initalize_beacon3(beacon3_stop)
    #find_tag(tag_stop)
    tag_and_beacon_signals = threading.Thread(target=find_tag, args=(tag_stop,))
    tag_and_beacon_signals.start()  # send all the signals to the ports 
    ani = animation.FuncAnimation(fig, update_plot, frames=None, interval=100)
    plt.show()                      # see the location of the tag
    tag_and_beacon_signals.stop()   # end all the threads
    keyboard_listener.stop()
# calls all the required funtions
def main():
    initialize_global_variables()
    header = generate_csv_header(port_and_sensors_list)
    file_path = create_csv_file(header)
    with open(file_path, "a", newline='') as data_file:
        global csv_writer
        csv_writer = csv.writer(data_file)
        start_all_threads()
    start_all_threads()
# makes sure  that he main gets called
if __name__ == "__main__":
    main()