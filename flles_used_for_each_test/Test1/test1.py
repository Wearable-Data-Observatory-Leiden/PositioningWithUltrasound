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
serialInst = serial.Serial(port0, baudrate = 115200, timeout=1)
sketch_path = "/home/stanshands/Arduino/test1New/test1New.ino"
board_type = "uno"
upload_arduino_sketch(sketch_path, board_type, port0)

def test():
    start_time = time.time()
    duration = 61
    end_time = start_time + duration
    while time.time() < end_time:
        row = [""] * 3 
        if serialInst.in_waiting:   # wait for data from the arduino
            data = serialInst.readline().decode('utf').rstrip('\n')
            time_elapsed = round(time.time() - start_time, 4)
            if time_elapsed<1: continue
            if data.startswith('1:'):
                sensor1_data = data[2:] 
                row = [time_elapsed, sensor1_data, '']
            elif data.startswith('2:'):
                sensor2_data = data[2:] 
                row = [time_elapsed, '', sensor2_data]
            csv_writer.writerow(row)

with open(file_path, "a", newline='') as data_file:
    global csv_writer
    csv_writer = csv.writer(data_file)
    test()