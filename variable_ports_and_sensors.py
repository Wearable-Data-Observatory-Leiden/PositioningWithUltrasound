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
# all the global variables get created and initialized
def initialize_global_variables():
    global all_points_over_time, fig, ax, sc, number_of_points_shown
    global known_locations, avg_distances, all_distances, threshold, start_time
    global board_type, port_and_sensors_list, sketch_path, max_sensors
    
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

    max_sensors = 0                  # maximum amount of senors on 1 arduino

    board_type = "uno"              # since we are working with the same port and sensors
    port_and_sensors_list = []
    sketch_path = "/home/stanshands/Arduino/Ysensors/3sensors/3sensors.ino"
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
# asks the user which beacons are on which ports and how many sensors
def ask_for_ports_and_sensor_count():
    available_ports = list(serial.tools.list_ports.comports())
    portsList = []                  # list to store the available ports
    print("These are the available ports:")
    for i, onePort in enumerate(available_ports, start=0):
        portsList.append(onePort.device)
        print(f"{i}. {onePort}")    # show the available ports
    nr_ports = int(input("How many arduinos do you want to use? (beacons + tag)"))
    for x in range(1, nr_ports+1):  # each beacon and tag gets assigned a port
        if x == nr_ports:
            chosen_nr = int(input(f"What port for the tag?"))
            sensors_beacon = int(input(f"Number of sensors for the tag?"))
        else:
            chosen_nr = int(input(f"What port for beacon {x}?"))
            sensors_beacon = int(input(f"Number of sensors for beacon {x}?"))
        if 0 <= chosen_nr < len(available_ports):
            chosen_port = portsList[chosen_nr]
        port_and_sensors_list.append((chosen_port,sensors_beacon))
    global max_sensors
    for entry in port_and_sensors_list: # upload the .ino code
        chosen_port, sensors = entry
        max_sensors = sensors if sensors > max_sensors else max_sensors
        current_sketch_path = sketch_path.replace('3', str(sensors))
        upload_arduino_sketch(current_sketch_path, board_type, chosen_port)
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
        if letter == "s":
            stop_event.set()
    except AttributeError:
        pass
# starts all threads
def start_all_threads(csv_writer):
    threads = []                    # usefull for grouping all threads
    csv_offset = 0                  # which column to start writing to csv
    stop_event = threading.Event()  # the signal that the threads shoulds stop
    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, stop_event))
    keyboard_listener.start()
    sensor_queues = [queue.Queue() for _ in port_and_sensors_list]  # make a queue for each thread
    for port_info,sensor_queue in zip(port_and_sensors_list, sensor_queues):# make all threads for the ports
        thread = threading.Thread(target=measurements_for_port, args=(csv_writer,stop_event,port_info,csv_offset,sensor_queue,))
        csv_offset += port_info[1]  # update the offset
        threads.append(thread)
    for thread in threads:          # Start all measurement threads
        thread.start()  
    try:
        while not stop_event.is_set():
            for x in range(1,max_sensors+1):
                for sensor_queue in sensor_queues:
                    sensor_queue.put(str(x))
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
    csv_writer.writerow(row)        # write to the csv file the correct data
# calls all the required funtions
def main():
    initialize_global_variables()
    ask_for_ports_and_sensor_count()
    header = generate_csv_header(port_and_sensors_list)
    file_path = create_csv_file(header)
    with open(file_path, "a", newline='') as data_file:
        csv_writer = csv.writer(data_file)
        start_all_threads(csv_writer)
# makes sure  that he main gets called
if __name__ == "__main__":
    main()