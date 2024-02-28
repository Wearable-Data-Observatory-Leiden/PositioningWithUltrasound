import serial
import subprocess
import time

sketch_path = "/home/stanshands/Arduino/presentation/1sensor_input/1sensor_input.ino"
board_type = "uno"  
port0 = '/dev/ttyACM0'
port1 = '/dev/ttyACM1'

# uploads the correct sketch to the arduino
def upload_arduino_sketch(sketch_path, board_type, port):
    cli_command = "/home/stanshands/bin/arduino-cli"
    fqbn = f"arduino:avr:{board_type}"
    subprocess.run([cli_command, "compile", "--fqbn", fqbn, sketch_path], capture_output=True, text=True)
    subprocess.run([cli_command, "upload", "--fqbn", fqbn, "--port", port, sketch_path])

# calls function to upload the sketches, send signal to arduino, recieve measurements from arduino
def main():
    upload_arduino_sketch(sketch_path, board_type, port0)
    upload_arduino_sketch(sketch_path, board_type, port1)
    serialInst1 = serial.Serial(port0, baudrate = 115200, timeout=1)    # set up to recieve data from arduino
    serialInst2 = serial.Serial(port1, baudrate = 115200, timeout=1)
    var1 = "1"         # this is just a signal that will be send to the arduino
    while True:
        serialInst1.write(var1.encode('utf-8'))
        serialInst2.write(var1.encode('utf-8'))
        time.sleep(0.2)
        if serialInst1.in_waiting:   # wait for data from the arduino
            data1 = serialInst1.readline().decode('utf').rstrip('\n')
            print(f"sData port1: {data1} ") # the first letter doesn't get printed for some reason
        if serialInst2.in_waiting:   # wait for data from the arduino
            data2 = serialInst2.readline().decode('utf').rstrip('\n')
            print(f"sData port2:                              {data2} ")

# makes sure  that he main gets called
if __name__ == "__main__":
    main()