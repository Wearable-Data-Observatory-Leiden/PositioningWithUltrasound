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

sketch_path = "/home/stanshands/Arduino/oefenen/interval_1_sensor/interval_1_sensor.ino"
board_type = "uno"  
port0 = '/dev/ttyACM0'
port1 = '/dev/ttyACM1'
time_to_cm =  343 / 10000


# uploads the correct sketch to the arduino
def upload_arduino_sketch(sketch_path, board_type, port):
    cli_command = "/home/stanshands/bin/arduino-cli"
    fqbn = f"arduino:avr:{board_type}"
    subprocess.run([cli_command, "compile", "--fqbn", fqbn, sketch_path], capture_output=True, text=True)
    subprocess.run([cli_command, "upload", "--fqbn", fqbn, "--port", port, sketch_path])

# calls all the required funtions
def main():
    upload_arduino_sketch(sketch_path, board_type, port0)
    upload_arduino_sketch(sketch_path, board_type, port1)
    serialInst1 = serial.Serial()    # set up to recieve data from arduino
    serialInst1.baudrate = 115200
    serialInst1.port = port0
    serialInst1.open()        
    serialInst2 = serial.Serial()    # set up to recieve data from arduino
    serialInst2.baudrate = 115200
    serialInst2.port = port1
    serialInst2.open()
    time.sleep(2)
    start = "S"      
    serialInst1.write(start.encode('utf-8'))
    serialInst2.write(start.encode('utf-8'))
    
    while True:
        if serialInst1.in_waiting:   # wait for data from the arduino
            data1 = serialInst1.readline().decode('utf').rstrip('\n')
            print(f"Port1: {data1} ")
        if serialInst2.in_waiting:   # wait for data from the arduino
            data2 = serialInst2.readline().decode('utf').rstrip('\n')
            print(f"Port2: {data2} ")



# makes sure  that he main gets called
if __name__ == "__main__":
    main()