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

sketch_path = "/home/stanshands/Arduino/Ysensors/timing_test/timing_test.ino"
board_type = "uno"  
port0 = '/dev/ttyACM0'
port1 = '/dev/ttyACM1'
time_to_cm =  343 / 10000
var1 = "1"


# uploads the correct sketch to the arduino
def upload_arduino_sketch(sketch_path, board_type, port):
    cli_command = "/home/stanshands/bin/arduino-cli"
    fqbn = f"arduino:avr:{board_type}"
    subprocess.run([cli_command, "compile", "--fqbn", fqbn, sketch_path], capture_output=True, text=True)
    subprocess.run([cli_command, "upload", "--fqbn", fqbn, "--port", port, sketch_path])

def send_signal(serial,queue_port):
    while True: # run until stopted                
        try:                    # try to get number from queue 
            sensor_number = queue_port.get(block=False)
            serial.write(sensor_number.encode('utf-8'))
        except queue.Empty:     
            sensor_number = None

# calls all the required funtions
def main():
    upload_arduino_sketch(sketch_path, board_type, port0)
    upload_arduino_sketch(sketch_path, board_type, port1)
    serialInst1 = serial.Serial()    # set up to recieve data from arduino
    serialInst1.baudrate = 2000000
    serialInst1.port = port0
    serialInst1.open()        
    serialInst2 = serial.Serial()    # set up to recieve data from arduino
    serialInst2.baudrate = 2000000
    serialInst2.port = port1
    serialInst2.open()
    queue1 = queue.Queue()
    queue2 = queue.Queue()
    thread1 = threading.Thread(target=send_signal, args=(serialInst1,queue1,))
    thread1.start()
    thread2 = threading.Thread(target=send_signal, args=(serialInst2,queue2,))
    thread2.start()
    time.sleep(2)
    while True:
        queue1.put("1")
        queue2.put("2")
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



# makes sure  that he main gets called
if __name__ == "__main__":
    main()