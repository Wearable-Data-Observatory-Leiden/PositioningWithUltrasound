import serial
import serial.tools.list_ports
import csv
import subprocess
import time
import os
import math
import threading
from pynput import mouse
from datetime import datetime
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
    global all_points_over_time, fig, ax, sc, number_of_points_shown
    global known_locations, avg_distances, all_distances, threshold, start_time
    global board_type, port0, port_and_sensors_list, sketch_path
    global data_queue
    
    all_points_over_time = [[]]     # this is where all the points are stored
    number_of_points_shown = 5      # how many points are plotted at the same time
    
    fig = plt.figure()              # Create a 3D plot
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], c='r', marker='o')
    ax.autoscale(enable=False)
    
    ax.set_xlabel('X Label')        # Set labels for each axis
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.set_xlim([0, 100])           # Set axis limits in cm
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])
    
    beacon1 = np.array([0, 0, 0])   # beacon locations
    beacon2 = np.array([35, 0, 0])
    beacon3 = np.array([0, 35, 0])
    
    avg_distances = np.array([0, 0, 0]) # distances for triangulation
    all_distances = np.array([0, 0, 0, 0, 0, 0]) # all sensor measurements
    known_locations = np.array([beacon1, beacon2, beacon3])

    threshold = 0.3                 # how far the measurements are allowed to be apart   
    
    start_time = time.time()        # Used for timestamps

    board_type = "uno"              # since we are working with the same port and sensors
    port0 = '/dev/ttyACM0'
    port_and_sensors_list = [('/dev/ttyACM0', 6)]
    sketch_path = "/home/stanshands/Arduino/Xsensors/6sensors/6sensors.ino"

    data_queue = queue.Queue()
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
 
    if (K1[2]>0):
        return K1
    else:
        return K2
# uploads the correct sketch to the arduino
def upload_arduino_sketch(sketch_path, board_type, port):
    cli_command = "/home/stanshands/bin/arduino-cli"
    fqbn = f"arduino:avr:{board_type}"
    subprocess.run([cli_command, "compile", "--fqbn", fqbn, sketch_path], capture_output=True, text=True)
    subprocess.run([cli_command, "upload", "--fqbn", fqbn, "--port", port, sketch_path])
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
# keyboard listener
def on_key_press(key, stop_event):
    try:
        letter = key.char
        data_queue.put(letter)
        if letter == "s":
            stop_event.set()
    except AttributeError:
        pass
# a map for letters to points
letter_to_point = {
    'a': [10, 10, 10],
    'b': [20, 20, 20],
    'c': [30, 30, 30],
    'd': [40, 40, 40],
    'e': [50, 50, 50],
    'f': [60, 60, 60],
    'g': [70, 70, 70],
    'h': [80, 80, 80],
    'i': [90, 90, 90],
    'j': [100, 100, 100],
    'k': [10, 20, 30],
    'l': [20, 30, 40],
    'm': [30, 40, 50],
    'n': [40, 50, 60],
    'o': [50, 60, 70],
    'p': [60, 70, 80],
    'q': [70, 80, 90],
    'r': [80, 90, 100],
    's': [90, 100, 10],
    't': [100, 10, 20],
    'u': [10, 20, 30],
    'v': [20, 30, 40],
    'w': [30, 40, 50],
    'x': [40, 50, 60],
    'y': [50, 60, 70],
    'z': [60, 70, 80],
    # Add more letters and corresponding points as needed
}
def update_plot(frame):
    while not data_queue.empty():
        item = data_queue.get()

        if isinstance(item, str) and item in letter_to_point:
            # If the item is a letter, convert it to a 3D point
            letter = item
            new_point = letter_to_point[letter]
            print(f"New point for letter {letter}: {new_point}.")
        elif isinstance(item, np.ndarray) and item.shape == (3,):
            # If the item is a NumPy array representing a 3D point, use it directly
            new_point = tuple(item)
            print(f"New point: {new_point}.")
        else:
            print(f"Invalid item in the queue: {item}")
            continue

        all_points_over_time[0].append(new_point)
        current_points = all_points_over_time[0][-number_of_points_shown:]
        x, y, z = zip(*current_points)
        sc._offsets3d = (x, y, z)
        fig.canvas.flush_events()
# starts all threads
def start_all_threads(csv_writer):
    threads = []                    # usefull for grouping all threads
    csv_offset = 0                  # which column to start writing to csv
    stop_event = threading.Event()  # the signal that the threads shoulds stop
    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, stop_event))
    keyboard_listener.start()
    sensor_queues = [queue.Queue() for _ in port_and_sensors_list]  # make a queue for each thread
    for port_info,sensor_queue in zip(port_and_sensors_list, sensor_queues,):# make all threads for the ports
        thread = threading.Thread(target=measurements_for_port, args=(csv_writer,stop_event,port_info,csv_offset,sensor_queue,))
        csv_offset += port_info[1]  # update the offset
        threads.append(thread)
    for thread in threads:          # Start all measurement threads
        thread.start()  
    time.sleep(1)
    ani = animation.FuncAnimation(fig, update_plot, frames=None, interval=100)
    plt.show()
    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        stop_event.set()
        keyboard_listener.stop()
        for thread in threads:
            thread.join()
# recieves all measurements from a port
def measurements_for_port(csv_writer,stop_event,port_info,csv_offset,sensor_queue):
    port = port_info[0]             # select the correct port
    serialInst = serial.Serial()    # set up to recieve data from arduino
    serialInst.baudrate = 9600
    serialInst.port = port
    serialInst.open()
    try:
        while not stop_event.is_set(): # run until stopted                
            try:                    # try to get number from queue 
                sensor_number = sensor_queue.get(block=False)
            except queue.Empty:     
                sensor_number = None
            if sensor_number is not None: # send number to the arduino
                serialInst.write(sensor_number.encode('utf-8'))
            if serialInst.in_waiting:   # wait for data from the arduino
                data = serialInst.readline().decode('utf').rstrip('\n')
                print(f"Received data: {data}")
                process_measurement_data(data, csv_writer,csv_offset)
    except KeyboardInterrupt:       # Handle keyboard interrupt
        pass
    finally:
        serialInst.close()
        print("Closed the serial")
# takes the measurements and fills in the csv file and calculates the new point
def process_measurement_data(data, csv_writer, csv_offset):
    parts = data.split(":")         # split the sensor nr and distance
    if len(parts) != 2:             # exeption handling
        print(f"Unexpected data format: {data}. Skipping this entry.")
        return
    sensornr = int(parts[0].strip())
    measurement = float(parts[1].strip())
    measurement_cm = (measurement * 343 / 10000) + 5 #extra distance because the sensors are some distance away from each other 
    timestamp = time.time() - start_time
    row = [''] * 24                 # for now fill in the other cells with ' '
    row[0] = timestamp              # fill in the time
    row[sensornr+csv_offset] = measurement_cm  # fill in the measurement
    if sensornr % 2 == 1:           # Check if we can calculate a new point
        all_distances[sensornr-1] = measurement_cm
    else:                           # only even numbers so both measurements are done
        other_sensor = sensornr - 1
        if abs(measurement_cm - all_distances[other_sensor-1]) / max(abs(float(measurement_cm)), abs(float(all_distances[other_sensor-1]))) >= threshold:
            print("Measurements too far apart. Skipping this entry.")
        else:
            all_distances[sensornr-1] = measurement_cm
            avg_distances[(sensornr // 2)-1] = (measurement_cm + all_distances[other_sensor-1])//2
            new_point = trilateration()
            data_queue.put(new_point)
            row[7] = new_point      # write the calculated point to the csv file
    csv_writer.writerow(row)        # write to the csv file the correct data
# calls all the required funtions
def main():
    initialize_global_variables()
    upload_arduino_sketch(sketch_path, board_type, port0)
    header = generate_csv_header(port_and_sensors_list)
    file_path = create_csv_file(header)
    with open(file_path, "a", newline='') as data_file:
        csv_writer = csv.writer(data_file)
        start_all_threads(csv_writer)
# makes sure  that he main gets called
if __name__ == "__main__":
    main()