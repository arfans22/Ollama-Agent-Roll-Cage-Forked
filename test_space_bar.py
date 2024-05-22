import keyboard
import time

def wait_for_space():
    print("Press the space bar to test...")
    while True:
        if keyboard.is_pressed('space'):
            while keyboard.is_pressed('space'):
                time.sleep(0.1)  # Wait for the space bar to be released
            print("Space bar detected!")
            break
        time.sleep(0.1)

if __name__ == "__main__":
    wait_for_space()