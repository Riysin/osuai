# Building an AI to Play osu! - Complete Tutorial

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Environment Setup](#environment-setup)
3. [State Representation](#state-representation)
4. [Action Space](#action-space)
5. [Reward Function](#reward-function)
6. [Neural Network Architecture](#neural-network-architecture)
7. [Training Process](#training-process)
8. [Advanced Techniques](#advanced-techniques)

---

## Project Architecture

### High-Level Overview
```
┌─────────────────┐
│   osu! Game     │
└────────┬────────┘
         │
    ┌────┴─────┬──────────────┐
    │          │              │
    ▼          ▼              ▼
Screen     gosumemory    Mouse/Click
Capture    (Game State)   Control
    │          │              │
    └────┬─────┴──────────────┘
         ▼
   ┌─────────────┐
   │ RL Agent    │
   │ (Neural Net)│
   └─────────────┘
```

### Key Components:
1. **Input**: Screen pixels + game state data
2. **Processing**: Convolutional Neural Network
3. **Output**: Mouse position + click timing
4. **Feedback**: Reward based on accuracy (300/100/50/miss)

---

## Environment Setup

### Prerequisites
```bash
# Core ML libraries
pip install torch torchvision  # PyTorch
pip install stable-baselines3  # RL algorithms
pip install gymnasium          # RL environment interface

# Game interaction
pip install pyautogui          # Mouse control
pip install mss opencv-python  # Screen capture (you have these!)
pip install websockets         # gosumemory connection (you have this!)

# Utilities
pip install numpy matplotlib tensorboard
```

### gosumemory Setup
1. Download from https://github.com/l3lackShark/gosumemory
2. Run gosumemory.exe while osu! is open
3. Access WebSocket at `ws://localhost:24050/ws`

---

## State Representation

### Option 1: Raw Pixels (Current Approach)
```python
# Your current implementation
def capture_screen(self):
    capture_range = {'top': 30, 'left': 30, 'width': 920, 'height': 770}
    screen = sct.grab(capture_range)
    screen_array = np.array(screen)
    gray_screen = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
    downscaled_screen = cv2.resize(gray_screen, (0, 0), fx=0.5, fy=0.5)
    return downscaled_screen  # Shape: (385, 460)
```

**Pros**: Simple, captures all visual information
**Cons**: High dimensional, slow to train

### Option 2: Feature Extraction (Recommended)
Extract hit circle positions using computer vision:

```python
def extract_features(self, screen):
    """
    Extract hit circles from screen using computer vision
    Returns: list of (x, y, radius, time_until_hit)
    """
    # 1. Threshold to find bright circles
    _, thresh = cv2.threshold(screen, 200, 255, cv2.THRESH_BINARY)

    # 2. Find contours (circles)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 3. Filter for circular shapes
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:  # Likely a circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append((x, y, radius))

    return circles
```

### Option 3: Hybrid (Best)
Combine raw pixels with extracted features:
- CNN processes downscaled image
- Concatenate with explicit circle positions from gosumemory or CV

---

## Action Space

### Continuous Actions (Recommended for osu!)
```python
action_space = {
    'mouse_x': continuous(-1, 1),      # Normalized screen position
    'mouse_y': continuous(-1, 1),      # Normalized screen position
    'click': discrete(0, 1)            # Click or not
}
```

### Implementation
```python
import pyautogui

def execute_action(self, action):
    """
    action: [mouse_x, mouse_y, click]
    mouse_x, mouse_y in range [-1, 1]
    click in {0, 1}
    """
    # Convert normalized coords to screen coords
    screen_width = 920
    screen_height = 770
    x = int((action[0] + 1) * screen_width / 2) + 30  # Offset
    y = int((action[1] + 1) * screen_height / 2) + 30

    # Move mouse
    pyautogui.moveTo(x, y, duration=0)  # Instant movement

    # Click if action[2] > 0.5 (for continuous) or == 1 (for discrete)
    if action[2] > 0.5:
        pyautogui.click()
```

### Alternative: Delta Movement
```python
# Instead of absolute position, move relative to current position
delta_x = action[0] * max_speed
delta_y = action[1] * max_speed
pyautogui.moveRel(delta_x, delta_y)
```

---

## Reward Function

### Basic Reward (Your Current Approach - Needs Improvement)
```python
def calculate_reward(self):
    # Your current implementation
    reward = 0
    c300 = status['300']
    time.sleep(0.00001)
    if status['300'] > c300:
        reward = 1.0
    else:
        reward = -0.1
    return reward
```

**Problem**: This only checks if 300 count increased, doesn't account for 100/50/miss

### Improved Reward Function
```python
class OsuEnv():
    def __init__(self):
        self.prev_hits = {'300': 0, '100': 0, '50': 0, 'miss': 0}

    def calculate_reward(self):
        reward = 0

        # Check what changed
        if status['300'] > self.prev_hits['300']:
            reward = 1.0      # Perfect hit
        elif status['100'] > self.prev_hits['100']:
            reward = 0.5      # Good hit
        elif status['50'] > self.prev_hits['50']:
            reward = 0.1      # Okay hit
        elif status['miss'] > self.prev_hits['miss']:
            reward = -1.0     # Penalty for miss
        else:
            reward = -0.01    # Small penalty for doing nothing

        # Update previous hits
        self.prev_hits = status.copy()

        return reward
```

### Advanced Reward Shaping
```python
def calculate_reward(self):
    reward = 0

    # Reward based on accuracy
    hit_rewards = {
        '300': 1.0,
        '100': 0.5,
        '50': 0.2,
        'miss': -1.0
    }

    # Check for new hits
    for hit_type, reward_value in hit_rewards.items():
        if status[hit_type] > self.prev_hits[hit_type]:
            reward += reward_value

    # Bonus for combo
    if status.get('combo', 0) > 10:
        reward += 0.1  # Bonus for maintaining combo

    # Penalty for being far from targets (requires CV detection)
    # distance_penalty = -self.distance_to_nearest_circle() * 0.01
    # reward += distance_penalty

    self.prev_hits = status.copy()
    return reward
```

---

## Neural Network Architecture

### Option 1: CNN for Raw Pixels
```python
import torch
import torch.nn as nn

class OsuCNN(nn.Module):
    def __init__(self, input_shape=(1, 385, 460), action_dim=3):
        super(OsuCNN, self).__init__()

        # Convolutional layers to process screen
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_output(input_shape)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.conv(dummy)
            return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
```

### Option 2: Feature-Based Network
```python
class OsuFeatureNet(nn.Module):
    def __init__(self, num_circles=10, action_dim=3):
        super(OsuFeatureNet, self).__init__()

        # Input: [x1, y1, r1, t1, x2, y2, r2, t2, ..., cursor_x, cursor_y]
        input_dim = num_circles * 4 + 2

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)
```

---

## Training Process

### Step 1: Create Gymnasium-Compatible Environment
```python
import gymnasium as gym
from gymnasium import spaces

class OsuEnv(gym.Env):
    def __init__(self):
        super(OsuEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(385, 460),
            dtype=np.uint8
        )

        self.prev_hits = {'300': 0, '100': 0, '50': 0, 'miss': 0}

    def step(self, action):
        # Execute action
        self.execute_action(action)

        # Get new state
        observation = self.capture_screen()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if episode is done
        terminated = self.is_done()
        truncated = False

        # Additional info
        info = {
            'hits_300': status['300'],
            'hits_100': status['100'],
            'hits_50': status['50'],
            'misses': status['miss']
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Wait for user to start a new song or restart
        print("Waiting for game to start...")
        while status['state'] != 2:  # 2 = playing
            time.sleep(0.1)

        self.prev_hits = {'300': 0, '100': 0, '50': 0, 'miss': 0}
        observation = self.capture_screen()
        info = {}
        return observation, info
```

### Step 2: Train with Stable-Baselines3 (Easiest)
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = OsuEnv()
env = DummyVecEnv([lambda: env])

# Create PPO agent
model = PPO(
    "CnnPolicy",  # Use CNN policy for image input
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    tensorboard_log="./tensorboard_logs/"
)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("osu_ai_model")
```

### Step 3: Implement Custom Training Loop (Advanced)
```python
import torch
import torch.optim as optim

# Initialize
env = OsuEnv()
model = OsuCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0

    while True:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

        # Get action from model
        with torch.no_grad():
            action = model(state_tensor).squeeze().numpy()

        # Execute action
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        if done or truncated:
            break

        state = next_state

    print(f"Episode {episode}: Reward = {episode_reward}")
```

---

## Advanced Techniques

### 1. Curriculum Learning
Start with easy songs, gradually increase difficulty:
```python
difficulty_levels = ['Easy', 'Normal', 'Hard', 'Insane', 'Expert']
current_level = 0

if average_accuracy > 0.8 and episodes > 100:
    current_level = min(current_level + 1, len(difficulty_levels) - 1)
```

### 2. Demonstration Learning (Imitation Learning)
Record human gameplay and train the AI to imitate:
```python
# Record expert demonstrations
demonstrations = []
while playing:
    state = capture_screen()
    action = get_mouse_position_and_click()
    demonstrations.append((state, action))

# Train with behavioral cloning
for state, action in demonstrations:
    predicted_action = model(state)
    loss = mse_loss(predicted_action, action)
    loss.backward()
    optimizer.step()
```

### 3. Frame Stacking
Use multiple consecutive frames to capture motion:
```python
from collections import deque

class FrameStack:
    def __init__(self, n_frames=4):
        self.frames = deque(maxlen=n_frames)

    def add(self, frame):
        self.frames.append(frame)

    def get_state(self):
        return np.concatenate(self.frames, axis=0)
```

### 4. Attention Mechanism
Focus on relevant parts of the screen:
```python
class AttentionOsuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ...  # CNN layers
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.fc = ...    # FC layers
```

### 5. Use gosumemory for Precise Timing
```python
def get_hit_objects_from_memory():
    """
    gosumemory can provide upcoming hit objects
    This gives you perfect information about what's coming
    """
    # Access gosumemory API for hit object data
    # You'll need to explore their API documentation
    pass
```

---

## Common Challenges & Solutions

### Challenge 1: Slow Training
**Solution**:
- Use smaller screen resolution
- Train on shorter/easier songs first
- Use GPU acceleration
- Implement frame skipping (only process every nth frame)

### Challenge 2: AI Clicks Too Early/Late
**Solution**:
- Add audio offset compensation
- Use timing information from gosumemory
- Train separate network for timing vs position

### Challenge 3: Overfitting to One Song
**Solution**:
- Train on diverse song set
- Use data augmentation (speed up/slow down songs)
- Regularization (dropout, weight decay)

### Challenge 4: Mouse Movement Too Jerky
**Solution**:
- Add action smoothing
- Use trajectory prediction
- Implement velocity-based control instead of position

---

## Next Steps for Your Project

Based on your current code, here's what to implement next:

1. **Fix the environment** (main.py):
   - Implement proper `reset()` method
   - Fix `calculate_reward()` to track all hit types
   - Add `execute_action()` method using pyautogui

2. **Add action execution**:
   - Install pyautogui: `pip install pyautogui`
   - Implement mouse control

3. **Make it Gymnasium-compatible**:
   - Add proper spaces
   - Return (observation, reward, terminated, truncated, info)

4. **Choose an RL algorithm**:
   - Start with PPO from stable-baselines3 (easiest)
   - Or implement DQN/A3C yourself

5. **Train on easy songs first**:
   - Find 1-2 star difficulty songs
   - Short songs (1-2 minutes)
   - Consistent patterns

6. **Monitor training**:
   - Use tensorboard to track rewards
   - Log accuracy metrics (300/100/50/miss counts)

---

## Resources

- **gosumemory**: https://github.com/l3lackShark/gosumemory
- **Stable-Baselines3 docs**: https://stable-baselines3.readthedocs.io/
- **PyTorch tutorials**: https://pytorch.org/tutorials/
- **OpenAI Gym tutorial**: https://gymnasium.farama.org/
- **Similar projects**: Search GitHub for "osu reinforcement learning"

---

## Example: Complete Training Script

See `train.py` (to be created) for a complete working example that ties everything together.

Good luck with your osu! AI project! 🎮🤖
