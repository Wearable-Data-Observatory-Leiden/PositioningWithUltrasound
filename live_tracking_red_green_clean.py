import serial
import serial.tools.list_ports
import subprocess
import time
import math
import threading
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
    global board_type, port0,port1,port2,port3, port_and_sensors_list, sketch_path
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
    all_distances = np.array([0, 0, 0, 0, 0, 0]) # all sensor measurements
    
    threshold = 20                 # % of how far the measurements are allowed to be apart   
    
    start_time = time.time()        # Used for timestamps

    board_type = "uno"              # since we are working with the same port and sensors
    port0 = '/dev/ttyACM0'
    port1 = '/dev/ttyACM1'
    port2 = '/dev/ttyACM2'
    port3 = '/dev/ttyACM3'
    port_and_sensors_list = [('/dev/ttyACM0', 2), ('/dev/ttyACM1', 2), ('/dev/ttyACM2', 2), ('/dev/ttyACM3', 3)]
    sketch_path = "/home/stanshands/Arduino/Ysensors/2sensors/2sensors.ino"

    ser1 = serial.Serial(port0, baudrate = 115200, timeout=1)    # set up to recieve data from arduino
    ser2 = serial.Serial(port1, baudrate = 115200, timeout=1)
    ser3 = serial.Serial(port2, baudrate = 115200, timeout=1)
    ser4 = serial.Serial(port3, baudrate = 115200, timeout=1)

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
# gets serial and sensor numbers, then sends and recieves the data from the arduino
def communicate_and_process(ser_send, ser_receive, sensornr_send, sensornr_receive):
    while True:
        ser_send.write(sensornr_send.encode('utf-8'))
        ser_receive.write(sensornr_receive.encode('utf-8'))
        time.sleep(0.1)
        if ser_send.in_waiting:   # wait for data from the arduino
            data_send = ser_send.readline().decode('utf').rstrip('\n')
            distance_send = data_send.split(":")
            dist_send = float(distance_send[1].strip()) * time_to_cm
            print(f"Data send: {dist_send} ")
        if ser_receive.in_waiting:   # wait for data from the arduino
            data_receive = ser_receive.readline().decode('utf').rstrip('\n')
            distance_receive = data_receive.split(":")
            dist_receive = float(distance_receive[1].strip()) * time_to_cm
            print(f"Data receive: {dist_receive} ")
        yield dist_send, dist_receive
# measure the distance between beacon 1 and 2 and set location beacon 2
def initalize_beacon2(beacon2_stop):
    sensornrB1 = "1"
    sensornrB2 = "2"
    communicator = communicate_and_process(ser1, ser2, sensornrB1, sensornrB2)
    while not beacon2_stop.is_set():
        dist1, dist2 = next(communicator)
    print(f"Distance beacon 1 -> 2: {dist1}, 2 -> 1: {dist2}")
    known_locations[1] = [0,(dist1 + dist2) / 2,0]
    print(f"Location beacon 2: {known_locations[1]}")
# measure the distance between beacon 3 and other beacons beacon and sets location beacon 3
def initalize_beacon3(beacon3_stop):
    sensornrB13 = "2"
    sensornrB31 = "1"
    sensornrB23 = "1"
    sensornrB32 = "2"
    communicator1 = communicate_and_process(ser1, ser3, sensornrB13, sensornrB31)
    communicator2 = communicate_and_process(ser2, ser3, sensornrB23, sensornrB32)
    while not beacon3_stop.is_set():
        dist13, dist31 = next(communicator1)
        dist23, dist32 = next(communicator2)
    print(f"Distance beacon 1 -> 3: {dist13}, 3->1: {dist31}")
    avg_13 = (dist13 + dist31) / 2
    print(f"Average distance beacon 1 and 3: {avg_13}")
    print(f"Distance beacon 2 -> 3: {dist23}, 3->2: {dist32}")
    avg_23 = (dist23 + dist32) / 2
    print(f"Average distance beacon 2 and 3: {avg_23}")
    beacon3_x = ((avg_13 ** 2) - (avg_23 ** 2) + (known_locations[1][1] ** 2)) / (2*known_locations[1][1])
    beacon3_y = math.sqrt((avg_13 ** 2) - (beacon3_x ** 2))
    known_locations[2] = [beacon3_x,beacon3_y,0]
    print(f"Final locations, beacon 2: {known_locations[1]}, beacon 3 {known_locations[2]}")
# sends the information of which sensors should trigger to the arduinos to find the tag location
def find_tag(tag_stop):
    t1 = "1"
    t2 = "2"
    t3 = "3"
    b1 = "2"
    b2 = "2"
    b3 = "1"
    communicator1 = communicate_and_process(ser1, ser4, b1, t1)
    communicator2 = communicate_and_process(ser2, ser4, b2, t2)
    communicator3 = communicate_and_process(ser3, ser4, b3, t3)
    while not tag_stop.is_set():
        dist1t, distt1 = next(communicator1)
        dist2t, distt2 = next(communicator2)
        dist3t, distt3 = next(communicator3)
        print(avg_distances)
        new_point = trilateration()
        data_queue.put(new_point)
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
    tag_and_beacon_signals = threading.Thread(target=find_tag, args=(tag_stop,))
    tag_and_beacon_signals.start()  # send all the signals to the ports 
    ani = animation.FuncAnimation(fig, update_plot, frames=None, interval=100)
    plt.show()                      # see the location of the tag
    tag_and_beacon_signals.stop()   # end all the threads
    keyboard_listener.stop()
# calls all the required funtions
def main():
    initialize_global_variables()
    upload_arduino_sketch(sketch_path, board_type, port0)
    upload_arduino_sketch(sketch_path, board_type, port1)
    upload_arduino_sketch(sketch_path, board_type, port2)
    sketch_path2 = "/home/stanshands/Arduino/Ysensors/3sensors/3sensors.ino"
    upload_arduino_sketch(sketch_path2, board_type, port3)

    start_all_threads()
# makes sure  that he main gets called
if __name__ == "__main__":
    main()