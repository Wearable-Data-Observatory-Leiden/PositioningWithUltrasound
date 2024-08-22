import serial.tools.list_ports
import serial
import time
import os
import csv
import subprocess

# create a csv file and adds the header
def create_csv_file(header):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, "test_data.csv")
    with open(file_path, "w", newline='') as data_file:
        csv_writer = csv.writer(data_file)
        csv_writer.writerow(header)
    return file_path
header = ["time", "sensor1", "sensor2"]
file_path = create_csv_file(header)

# oploads the correct file to the arduino and starts running
def upload_arduino_sketch(sketch_path, board_type, port):
    cli_command = "/home/stanshands/bin/arduino-cli"
    fqbn = f"arduino:avr:{board_type}"
    subprocess.run([cli_command, "compile", "--fqbn", fqbn, sketch_path], capture_output=True, text=True)
    subprocess.run([cli_command, "upload", "--fqbn", fqbn, "--port", port, sketch_path])
port0 = '/dev/ttyACM0'
port1 = '/dev/ttyACM1'
serialInst1 = serial.Serial(port0, baudrate = 115200, timeout=1)
serialInst2 = serial.Serial(port1, baudrate = 115200, timeout=1)
sketch_path = "/home/stanshands/Arduino/test2New/test2New.ino"
board_type = "uno"
upload_arduino_sketch(sketch_path, board_type, port0)
upload_arduino_sketch(sketch_path, board_type, port1)

def test():
    start_time = time.time()
    duration = 61
    end_time = start_time + duration
    while time.time() < end_time:
        row1 = [""] * 3 
        row2 = [""] * 3 
        serialInst1.write("1".encode('utf-8'))
        serialInst2.write("1".encode('utf-8'))
        time.sleep(0.1)
        if serialInst1.in_waiting:   # wait for data from the arduino
            data = serialInst1.readline().decode('utf').rstrip('\n')
            time_elapsed = round(time.time() - start_time, 4)
            if time_elapsed<1: continue
            sensor1_data = data[2:] 
            row1 = [time_elapsed, sensor1_data, '']
            csv_writer.writerow(row1)
        if serialInst2.in_waiting:   # wait for data from the arduino
            data = serialInst2.readline().decode('utf').rstrip('\n')
            time_elapsed = round(time.time() - start_time, 4)
            if time_elapsed<1: continue
            sensor2_data = data[2:] 
            row2 = [time_elapsed, '',sensor2_data]
            csv_writer.writerow(row2)

with open(file_path, "a", newline='') as data_file:
    global csv_writer
    csv_writer = csv.writer(data_file)
    test()