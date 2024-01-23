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
import itertools
# all the global variables get created and initialized
def initialize_global_variables():
    global all_points_over_time, fig, ax, sc, number_of_points_shown
    global known_locations, avg_distances, all_distances, threshold, start_time
    global board_type, port0,port1,port2,port3, port_and_sensors_list, sketch_path
    global data_queue, time_to_cm, serialInst1, serialInst2, serialInst3, serialInst4
    
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
    all_distances = np.array([0, 0, 0, 0, 0, 0]) # all sensor measurements
    
    threshold = 0.3                 # how far the measurements are allowed to be apart   
    
    start_time = time.time()        # Used for timestamps

    board_type = "uno"              # since we are working with the same port and sensors
    port0 = '/dev/ttyACM0'
    port1 = '/dev/ttyACM1'
    port2 = '/dev/ttyACM2'
    port3 = '/dev/ttyACM3'
    port_and_sensors_list = [('/dev/ttyACM0', 2), ('/dev/ttyACM1', 2), ('/dev/ttyACM2', 2), ('/dev/ttyACM3', 3)]
    sketch_path = "/home/stanshands/Arduino/Ysensors/2sensors/2sensors.ino"


    serialInst1 = serial.Serial()    # set up to recieve data from arduino
    serialInst1.baudrate = 115200
    serialInst1.port = port0
    serialInst1.open()
    serialInst2 = serial.Serial()    # set up to recieve data from arduino
    serialInst2.baudrate = 115200
    serialInst2.port = port1
    serialInst2.open()
    serialInst3 = serial.Serial()    # set up to recieve data from arduino
    serialInst3.baudrate = 115200
    serialInst3.port = port2
    serialInst3.open()
    serialInst4 = serial.Serial()    # set up to recieve data from arduino
    serialInst4.baudrate = 115200
    serialInst4.port = port3
    serialInst4.open()    

    time_to_cm =  343 / 10000       # conversion rate in case of normal temperature

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
# measure the distance between beacon 1 and 2 and set location beacon 2
def initalize_beacon2(beacon2_stop):
    sensornrB1 = "1"          # these need to be changed
    sensornrB2 = "2"          # to the correct sensor number 
    while not beacon2_stop.is_set():
        serialInst1.write(sensornrB1.encode('utf-8'))
        serialInst2.write(sensornrB2.encode('utf-8'))
        time.sleep(0.1)
        if serialInst1.in_waiting:   # wait for data from the arduino
            data1 = serialInst1.readline().decode('utf').rstrip('\n')
            distance1 = data1.split(":")
            dist1 = float(distance1[1].strip()) * time_to_cm
            print(f"Data port1: {dist1} ")
        if serialInst2.in_waiting:   # wait for data from the arduino
            data2 = serialInst2.readline().decode('utf').rstrip('\n')
            distance2 = data2.split(":")
            dist2 = float(distance2[1].strip()) * time_to_cm
            print(f"Data port2: {dist2} ")
    print(f"Distance beacon 1 and 2: {dist1}")
    print(f"Distance beacon 2 and 1: {dist2}")
    avg_12 = (dist1 + dist2) / 2
    print(f"Average distance beacon 1 and 2: {avg_12}")
    known_locations[1] = [0,avg_12,0]
    print(f"Double check if updated: {known_locations[1]}")
# measure the distance between beacon 3 and other beacons beacon and sets location beacon 3
def initalize_beacon3(beacon3_stop):
    sensornrB13 = "2"          # these need to be changed
    sensornrB31 = "1"          # to the correct sensor number 
    sensornrB23 = "1"
    sensornrB32 = "2"
    while not beacon3_stop.is_set():
        serialInst1.write(sensornrB13.encode('utf-8'))
        serialInst3.write(sensornrB31.encode('utf-8'))
        time.sleep(0.1)
        if serialInst1.in_waiting:   # wait for data from the arduino
            data13 = serialInst1.readline().decode('utf').rstrip('\n')
            distance13 = data13.split(":")
            dist13 = float(distance13[1].strip()) * time_to_cm
            print(f"Data 1->3: {dist13} ")
        if serialInst3.in_waiting:   # wait for data from the arduino
            data31 = serialInst3.readline().decode('utf').rstrip('\n')
            distance31 = data31.split(":")
            dist31 = float(distance31[1].strip()) * time_to_cm
            print(f"Data 3->1: {dist31} ")

        serialInst2.write(sensornrB23.encode('utf-8'))
        serialInst3.write(sensornrB32.encode('utf-8'))
        time.sleep(0.1)
        if serialInst2.in_waiting:   # wait for data from the arduino
            data23 = serialInst2.readline().decode('utf').rstrip('\n')
            distance23 = data23.split(":")
            dist23 = float(distance23[1].strip()) * time_to_cm
            print(f"Data 2->3: {dist23} ")
        if serialInst3.in_waiting:   # wait for data from the arduino
            data32 = serialInst3.readline().decode('utf').rstrip('\n')
            distance32 = data32.split(":")
            dist32 = float(distance32[1].strip()) * time_to_cm
            print(f"Data 3->2: {dist32} ")

    print(f"Distance beacon 1 and 3: {dist13}")
    print(f"Distance beacon 3 and 1: {dist31}")
    avg_13 = (dist13 + dist31) / 2
    print(f"Average distance beacon 1 and 3: {avg_13}")

    print(f"Distance beacon 2 and 3: {dist23}")
    print(f"Distance beacon 3 and 2: {dist32}")
    avg_23 = (dist23 + dist32) / 2
    print(f"Average distance beacon 2 and 3: {avg_23}")

    beacon3_x = ((avg_13 ** 2) - (avg_23 ** 2) + (known_locations[1][1] ** 2)) / (2*known_locations[1][1])
    beacon3_y = math.sqrt((avg_13 ** 2) - (beacon3_x ** 2))

    known_locations[2] = [beacon3_x,beacon3_y,0]
    print(f"Final locations, beacon 2: {known_locations[1]}, beacon 3 {known_locations[2]}")
# sends the information of which sensors should trigger to the arduinos to find the tag location
def find_tag(tag_stop):
    t1 = "1"          # to the correct sensor number 
    t2 = "2"
    t3 = "3"
    b1 = "2"
    b2 = "2"
    b3 = "1"
    # for now lets assume all the sensors nr 2 are the only ones that will connect with the tag
    while not tag_stop.is_set():
        serialInst1.write(b1.encode('utf-8'))
        serialInst4.write(t1.encode('utf-8'))
        time.sleep(0.2)
        if serialInst1.in_waiting:   # wait for data from the arduino
            data1t = serialInst1.readline().decode('utf').rstrip('\n')
            distance1t = data1t.split(":")
            dist1t = float(distance1t[1].strip()) * time_to_cm
            print(f"Data 1->t: {dist1t} ")
        if serialInst4.in_waiting:   # wait for data from the arduino
            datat1 = serialInst4.readline().decode('utf').rstrip('\n')
            distancet1 = datat1.split(":")
            distt1 = float(distancet1[1].strip()) * time_to_cm
            print(f"Data t->1: {distt1} ")
        
        serialInst2.write(b2.encode('utf-8'))
        serialInst4.write(t2.encode('utf-8'))
        time.sleep(0.2)
        if serialInst2.in_waiting:   # wait for data from the arduino
            data2t = serialInst2.readline().decode('utf').rstrip('\n')
            distance2t = data2t.split(":")
            dist2t = float(distance2t[1].strip()) * time_to_cm
            print(f"Data 2->t: {dist2t} ")
        if serialInst4.in_waiting:   # wait for data from the arduino
            datat2 = serialInst4.readline().decode('utf').rstrip('\n')
            distancet2 = datat2.split(":")
            distt2 = float(distancet2[1].strip()) * time_to_cm
            print(f"Data t->2: {distt2} ")


        serialInst3.write(b3.encode('utf-8'))
        serialInst4.write(t3.encode('utf-8'))
        time.sleep(0.2)
        if serialInst3.in_waiting:   # wait for data from the arduino
            data3t = serialInst3.readline().decode('utf').rstrip('\n')
            distance3t = data3t.split(":")
            dist3t = float(distance3t[1].strip()) * time_to_cm
            print(f"Data 3->t: {dist3t} ")
        if serialInst4.in_waiting:   # wait for data from the arduino
            datat3 = serialInst4.readline().decode('utf').rstrip('\n')
            distancet3 = datat3.split(":")
            distt3 = float(distancet3[1].strip()) * time_to_cm
            print(f"Data t->3: {distt3} ")   

        avg_distances = np.array([(dist1t+distt1)/2,(dist2t+distt2)/2,(dist3t+distt3)/2])
        print(avg_distances)
        new_point = trilateration()
        data_queue.put(new_point)
# starts all threads
def start_all_threads():
    threads = []                    # usefull for grouping all threads
    stop_event = threading.Event()  # the signal that the threads shoulds stop
    beacon2_stop = threading.Event()# the signal that beacon2 is in position
    beacon3_stop = threading.Event()# the signal that beacon2 is in position
    tag_stop = threading.Event()    # the signal that the tag location is no longer required
    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, stop_event,beacon2_stop,beacon3_stop,tag_stop))
    keyboard_listener.start()
    sensor_queues = [queue.Queue() for _ in port_and_sensors_list]  # make a queue for each thread
    for port_info,sensor_queue in zip(port_and_sensors_list, sensor_queues,):# make all threads for the ports
        thread = threading.Thread(target=measurements_for_port, args=(stop_event,port_info,))
        threads.append(thread)
    for thread in threads:          # Start all measurement threads
        thread.start()  
    time.sleep(2)
    initalize_beacon2(beacon2_stop) # find the locations of the beacons
    initalize_beacon3(beacon3_stop)
    tag_and_beacon_signals = threading.Thread(target=find_tag, args=(tag_stop,))
    tag_and_beacon_signals.start()  # send all the signals to the ports 
    ani = animation.FuncAnimation(fig, update_plot, frames=None, interval=100)
    plt.show()                      # see the location of the tag
    tag_and_beacon_signals.stop()   # end all the threads
    keyboard_listener.stop()
    for thread in threads:
            thread.join()
# recieves all measurements from a port
def measurements_for_port(stop_event,port_info):
    pass
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
    avg_distances = np.array([1,1,1])
    upload_arduino_sketch(sketch_path, board_type, port0)
    upload_arduino_sketch(sketch_path, board_type, port1)
    upload_arduino_sketch(sketch_path, board_type, port2)
    sketch_path2 = "/home/stanshands/Arduino/Ysensors/3sensors/3sensors.ino"
    upload_arduino_sketch(sketch_path2, board_type, port3)
    start_all_threads()
# makes sure  that he main gets called
if __name__ == "__main__":
    main()