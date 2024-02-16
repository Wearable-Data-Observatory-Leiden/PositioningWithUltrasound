import time
import queue
from queue import Queue
from pynput import keyboard

def initialize_sensors(data_queue):
    beacon_sensor = 2
    tag_sensor = 4
    beacon = 3
    combinations_checked = set()
    while True:
        if (beacon_sensor == 1):
            beacon_sensor = 2
        else:
            beacon_sensor = 1
            if beacon == 3:
                beacon = 1
                tag_sensor += 3 if tag_sensor < 3 else -3
            else:
                beacon += 1
                tag_sensor += 2 if tag_sensor < 4 else -4
        combination = (beacon, beacon_sensor, tag_sensor)
        if combination in combinations_checked:
            break  # Stop if combination has already been checked
        combinations_checked.add(combination)
        print(f"Beacon: {beacon}, B sensor: {beacon_sensor}, tag sensor: {tag_sensor}")
        while data_queue.empty():
            print(f"Beacon: {beacon}, B sensor: {beacon_sensor}, tag sensor: {tag_sensor}")
            time.sleep(0.2)
        letter = data_queue.get()
        # Trigger sensor ports here
    print(f"Amound: {len(combinations_checked)} All combinations checked")

def on_key_press(key,data_queue):
    try:
        letter = key.char
        data_queue.put(letter)
    except AttributeError:
        pass

# Example usage
def main():
    data_queue = queue.Queue()
    keyboard_listener = keyboard.Listener(on_press=lambda key: on_key_press(key,data_queue))
    keyboard_listener.start()
    initialize_sensors(data_queue)
    keyboard_listener.stop()

if __name__ == "__main__":
    main()