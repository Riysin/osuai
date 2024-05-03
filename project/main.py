from data_upload import status, thread_start
from screen import show_screen
import cv2
import time
from mss import mss
import numpy as np


def test():
    print('hi')
        
class OsuEnv():
    def __init__(self):
        self.test = test
        
    def step(self):
        reward = self.calculate_reward()
        done = self.is_done()
        new_state = self.capture_screen()
        info = None
        
        return new_state, done, reward, info
        
    def reset(self):
        return None
        
    def calculate_reward(self):
        reward = 0
        c300 = status['300']
        time.sleep(0.00001)
        if status['300'] > c300:
            reward = 1.0
        else:
            reward = -0.1
        return reward
    
    def capture_screen(self):
        capture_range = {'top': 30, 'left': 30, 'width': 920, 'height': 770}
        sct = mss()
        cv2.namedWindow('Osu! Capture', cv2.WINDOW_NORMAL)
        
        screen = sct.grab(capture_range)
        screen_array = np.array(screen)

        gray_screen = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
        ds_factor = 0.5
        downscaled_screen = cv2.resize(gray_screen, (0, 0), fx=ds_factor, fy=ds_factor)
        
        return downscaled_screen

    def is_done(self):
        if status['state'] == 2:
            return False
        else:
            return True
        
env = OsuEnv()
        
        
    
if __name__ == '__main__':
    thread_start()
    time.sleep(1)
    test()
    show_screen()
    print(env.step())