"""
Improved Gymnasium-compatible environment for osu! AI training
"""
import time
import numpy as np
import cv2
from mss import mss
import pyautogui
import gymnasium as gym
from gymnasium import spaces
from project.data_upload import status, thread_start


class OsuEnv(gym.Env):
    """
    Custom Environment for osu! that follows gymnasium interface
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(OsuEnv, self).__init__()

        # Screen capture settings
        self.capture_range = {'top': 30, 'left': 30, 'width': 920, 'height': 770}
        self.sct = mss()
        self.ds_factor = 0.5
        self.screen_width = int(920 * self.ds_factor)
        self.screen_height = int(770 * self.ds_factor)

        # Define action space: [mouse_x, mouse_y, click]
        # mouse_x, mouse_y are normalized to [-1, 1]
        # click is binary but represented as continuous [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space: grayscale downscaled screen
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width),
            dtype=np.uint8
        )

        # State tracking
        self.prev_hits = {'300': 0, '100': 0, '50': 0, 'miss': 0}
        self.render_mode = render_mode
        self.episode_steps = 0
        self.max_episode_steps = 10000  # Prevent infinite episodes

        # Start gosumemory data thread
        thread_start()
        time.sleep(1)  # Wait for connection

    def capture_screen(self):
        """Capture and process the game screen"""
        screen = self.sct.grab(self.capture_range)
        screen_array = np.array(screen)

        # Convert to grayscale
        gray_screen = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)

        # Downscale
        downscaled = cv2.resize(
            gray_screen,
            (self.screen_width, self.screen_height),
            interpolation=cv2.INTER_AREA
        )

        return downscaled

    def execute_action(self, action):
        """
        Execute the given action in the game

        Args:
            action: numpy array [mouse_x, mouse_y, click]
                   mouse_x, mouse_y in range [-1, 1]
                   click in range [0, 1]
        """
        # Convert normalized coordinates to screen coordinates
        x = int((action[0] + 1) * 920 / 2) + 30  # Offset by capture_range left
        y = int((action[1] + 1) * 770 / 2) + 30  # Offset by capture_range top

        # Clamp to screen bounds
        x = max(30, min(x, 950))
        y = max(30, min(y, 800))

        # Move mouse instantly
        pyautogui.moveTo(x, y, duration=0)

        # Click if action[2] > 0.5
        if action[2] > 0.5:
            pyautogui.click()

    def calculate_reward(self):
        """
        Calculate reward based on gameplay performance

        Returns:
            float: reward value
        """
        reward = 0.0

        # Reward for different hit types
        if status['300'] > self.prev_hits['300']:
            reward = 1.0  # Perfect hit
        elif status['100'] > self.prev_hits['100']:
            reward = 0.5  # Good hit
        elif status['50'] > self.prev_hits['50']:
            reward = 0.2  # Okay hit
        elif status['miss'] > self.prev_hits['miss']:
            reward = -1.0  # Miss penalty
        else:
            reward = -0.01  # Small penalty for inaction

        # Update tracking
        self.prev_hits = status.copy()

        return reward

    def is_done(self):
        """
        Check if episode should terminate

        Returns:
            bool: True if episode is done
        """
        # state == 2 means playing, anything else means not playing
        if status['state'] != 2:
            return True

        # Also terminate if max steps reached
        if self.episode_steps >= self.max_episode_steps:
            return True

        return False

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode

        Returns:
            observation: initial observation
            info: additional information
        """
        super().reset(seed=seed)

        print("Waiting for game to start (state == 2)...")
        print("Please start a song in osu!")

        # Wait for game to be in playing state
        while status['state'] != 2:
            time.sleep(0.1)

        print("Game started! Beginning episode...")

        # Reset tracking
        self.prev_hits = {
            '300': status['300'],
            '100': status['100'],
            '50': status['50'],
            'miss': status['miss']
        }
        self.episode_steps = 0

        # Get initial observation
        observation = self.capture_screen()
        info = {}

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: action to take

        Returns:
            observation: new state
            reward: reward received
            terminated: whether episode ended
            truncated: whether episode was truncated
            info: additional information
        """
        # Execute action
        self.execute_action(action)

        # Small delay to let game update (adjust as needed)
        time.sleep(0.01)

        # Get new observation
        observation = self.capture_screen()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if done
        terminated = self.is_done()
        truncated = False

        # Increment step counter
        self.episode_steps += 1

        # Additional info
        info = {
            'hits_300': status['300'],
            'hits_100': status['100'],
            'hits_50': status['50'],
            'misses': status['miss'],
            'state': status['state'],
            'episode_steps': self.episode_steps
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == 'human':
            screen = self.capture_screen()
            cv2.imshow('osu! AI View', screen)
            cv2.waitKey(1)

    def close(self):
        """Clean up resources"""
        cv2.destroyAllWindows()


# Test the environment
if __name__ == '__main__':
    print("Testing OsuEnv...")
    env = OsuEnv(render_mode='human')

    try:
        # Reset and wait for game
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        # Run for a few steps with random actions
        for i in range(100):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {i}: Reward={reward:.2f}, Info={info}")

            env.render()

            if terminated or truncated:
                print("Episode finished!")
                break

    finally:
        env.close()
