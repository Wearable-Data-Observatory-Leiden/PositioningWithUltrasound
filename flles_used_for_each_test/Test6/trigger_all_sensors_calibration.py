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
from tabulate import tabulate
# all the global variables get created and initialized
def initialize_global_variables():
    global all_points_over_time, fig, ax, sc, number_of_points_shown, port_and_sensors_list
    global known_locations, avg_distances, start_time
    global port0,port1,port2,port3, beacon_connection, tag_connection, csv_writer
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
    
    ax.set_xlim([0, 300])           # Set axis limits in cm
    ax.set_ylim([0, 300])
    ax.set_zlim([-200, 0])
    
    beacon1 = np.array([0, 0, 0])   # beacon locations
    beacon2 = np.array([202, 0, 0])
    beacon3 = np.array([160, 160, 0])    
    known_locations = np.array([beacon1, beacon2, beacon3])

    avg_distances = np.array([0, 0, 0]) # distances for triangulation

    beacon_connection = np.array([2,False,2,False,2,False])
    tag_connection = np.array([3,False,5,False,6,False])
    
    board_type = "uno"              # since we are working with the same port and sensors
    port0 = '/dev/ttyACM0'
    port1 = '/dev/ttyACM1'
    port2 = '/dev/ttyACM2'
    port3 = '/dev/ttyACM3'
    sketch_path = "/home/stanshands/Arduino/Ysensors/2sensors/2sensors.ino"
    sketch_path2 = "/home/stanshands/Arduino/Ysensors/6sensors/6sensors.ino"
    port_and_sensors_list = [('/dev/ttyACM0', 2),('/dev/ttyACM1', 2),('/dev/ttyACM2', 2),('/dev/ttyACM3', 6)]

    upload_arduino_sketch(sketch_path, board_type, port0)
    upload_arduino_sketch(sketch_path, board_type, port1)
    upload_arduino_sketch(sketch_path, board_type, port2)
    upload_arduino_sketch(sketch_path2, board_type, port3)

    ser1 = serial.Serial(port0, baudrate = 115200, timeout=1)    # set up to recieve data from arduino
    ser2 = serial.Serial(port1, baudrate = 115200, timeout=1)
    ser3 = serial.Serial(port2, baudrate = 115200, timeout=1)
    ser4 = serial.Serial(port3, baudrate = 115200, timeout=1)

    time_to_cm =  343 / 10000       # conversion rate in case of normal temperature

    data_queue = queue.Queue()

    csv_writer = None
    start_time = time.time()
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
 
    if K1[2] < 0:
        return K1
    elif K2[2] < 0:
        return K2
    else:
        #print("Error: Trilateration failed. Target point cannot be determined.")
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
    file_path = os.path.join(script_directory, "test_data-12-04.csv")
    with open(file_path, "w", newline='') as data_file:
        csv_writer = csv.writer(data_file)
        csv_writer.writerow(header)
    return file_path
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
        if letter == "q":
            tag_stop.set()
    except AttributeError:
        pass
# update the plot when a new point gets added
def update_plot(frame):
    while not data_queue.empty():
        item = data_queue.get()
        if isinstance(item, np.ndarray) and item.shape == (3,):
            new_point = tuple(item)
            #print(f"New point: {new_point}.")
        else:
            #print(f"Invalid item in the queue: {item}")
            continue
        all_points_over_time[0].append(new_point)
        current_points = all_points_over_time[0][-number_of_points_shown:]
        x, y, z = zip(*current_points)
        sc._offsets3d = (x, y, z)
        sc_static = ax.scatter([], [], [], color='blue')  # The beacons
        sc_static._offsets3d = tuple(map(tuple, known_locations.T))
        fig.canvas.flush_events()
# what sensors should be triggered and sends that to each port
def trigger_all_ports(beacon_sensor, tag_sensor):
    row1 = [''] * 24
    row2 = [''] * 24   
    row3 = [''] * 24
    row4 = [''] * 24    
    csv_entry1 = 0 + int(beacon_sensor)
    csv_entry2 = 2 + int(beacon_sensor)
    csv_entry3 = 4 + int(beacon_sensor)
    csv_entry4 = 6 + int(tag_sensor)
    ser1.write(beacon_sensor.encode('utf-8'))
    ser2.write(beacon_sensor.encode('utf-8'))
    ser3.write(beacon_sensor.encode('utf-8'))
    ser4.write(tag_sensor.encode('utf-8'))
    time.sleep(0.1)
    try:
        if ser1.in_waiting:   # wait for data from the arduino
            info_b1 = ser1.readline().decode('utf').rstrip('\n')
            distance_b1 = info_b1.split(":")
            dist_b1 = float(distance_b1[1].strip()) * time_to_cm
            row1[csv_entry1] = dist_b1               
            timestamp = time.time() - start_time 
            row1[0] = timestamp
            csv_writer.writerow(row1)
        if ser2.in_waiting:   # wait for data from the arduino
            info_b2 = ser2.readline().decode('utf').rstrip('\n')
            distance_b2 = info_b2.split(":")
            dist_b2 = float(distance_b2[1].strip()) * time_to_cm
            row2[csv_entry2] = dist_b2     
            csv_writer.writerow(row2)
        if ser3.in_waiting:   # wait for data from the arduino
            info_b3 = ser3.readline().decode('utf').rstrip('\n')
            distance_b3 = info_b3.split(":")
            dist_b3 = float(distance_b3[1].strip()) * time_to_cm
            row3[csv_entry3] = dist_b3             
            csv_writer.writerow(row3)
        if ser4.in_waiting:   # wait for data from the arduino
            info_b4 = ser4.readline().decode('utf').rstrip('\n')
            distance_b4 = info_b4.split(":")
            dist_b4 = float(distance_b4[1].strip()) * time_to_cm
            row4[csv_entry4] = dist_b4          
            csv_writer.writerow(row4)
    except UnboundLocalError as error:
        print(f"Error: {error}")
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
        try:
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
        except UnboundLocalError as error:
            print(f"Error: {error}, ser_send: {ser_send}")
            yield 0, 0  
# measure the distance between beacon 1 and 2 and set location beacon 2
def initalize_beacon2(beacon2_stop):
    sensornrB1 = "1"
    sensornrB2 = "2"
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
    sensornrB13 = "2"
    sensornrB31 = "1"
    sensornrB23 = "1"
    sensornrB32 = "2"
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
# measures all the timing differences between sensor pairs
def initalize_sensors():
    beacon_sensor = 2               # set up all the variables so that it starts at 1,1,1
    tag_sensor = 4
    beacon = 3
    combinations_checked = set()
    # clear the que, not sure why this is needed
    while not data_queue.empty():
        letter = data_queue.get()   # make sure the queue is empty
    while True:
        if (beacon_sensor == 1):    # alternate between beacon sensor
            beacon_sensor = 2
        else:
            beacon_sensor = 1
            if beacon == 3:         # begin again with beacon 1 and a new sensor
                beacon = 1
                tag_sensor += 3 if tag_sensor < 4 else -3
            else:
                beacon += 1
                tag_sensor += 2 if tag_sensor < 5 else -4
        combination = (beacon, beacon_sensor, tag_sensor)
        if combination in combinations_checked:
            break                   # Stop if combination has already been checked
        combinations_checked.add(combination)
        while data_queue.empty():   # Trigger all the ports in the same way as def find_tag
            trigger_all_ports(str(beacon_sensor),str(tag_sensor))
            print(f"Beacon: {beacon}, B sensor: {beacon_sensor}, tag sensor: {tag_sensor}")
        letter = data_queue.get()   # make sure the queue is empty
    print(f"Amound: {len(combinations_checked)} All combinations checked")
# sends the information of which sensors should trigger to the arduinos to find the tag location
def find_tag(tag_stop):
    while not tag_stop.is_set():
        trigger_all_ports("1","1")
        trigger_all_ports("2","2")
        trigger_all_ports("1","3")
        trigger_all_ports("2","4") 
        trigger_all_ports("1","5")
        trigger_all_ports("2","6")           
# starts all threads
def start_all_threads():
    stop_event = threading.Event()  # the signal that the threads shoulds stop
    beacon2_stop = threading.Event()# the signal that beacon2 is in position
    beacon3_stop = threading.Event()# the signal that beacon3 is in position
    tag_stop = threading.Event()    # the signal that the tag location is no longer required
    keyboard_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, stop_event,beacon2_stop,beacon3_stop,tag_stop))
    keyboard_listener.start()
    time.sleep(2)
    initalize_beacon2(beacon2_stop) # find the locations of the beacons
    initalize_beacon3(beacon3_stop)
    initalize_sensors() # calibrate the sensors
    find_tag(tag_stop)
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