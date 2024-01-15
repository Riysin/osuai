import cv2
import numpy as np
from mss import mss


def show_screen():
    capture_range = {'top': 30, 'left': 30, 'width': 920, 'height': 770}
    sct = mss()
    cv2.namedWindow('Osu! Capture', cv2.WINDOW_NORMAL)

    while True:
        screen = sct.grab(capture_range)
        screen_array = np.array(screen)

        gray_screen = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
        ds_factor = 0.5
        downscaled_screen = cv2.resize(gray_screen, (0, 0), fx=ds_factor, fy=ds_factor)    
        cv2.imshow('Osu! Capture', downscaled_screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
