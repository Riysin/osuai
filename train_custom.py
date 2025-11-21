"""
Custom training loop with PyTorch
For advanced users who want more control over the training process
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from osu_env_improved import OsuEnv


class OsuCNN(nn.Module):
    """
    Convolutional Neural Network for processing osu! screen
    """
    def __init__(self, input_shape=(385, 460), action_dim=3):
        super(OsuCNN, self).__init__()

        # Convolutional layers
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

        # Actor (policy) head - outputs actions
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

        # Critic (value) head - estimates state value
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_output(self, shape):
        """Calculate the output size of conv layers"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            output = self.conv(dummy)
            return int(np.prod(output.size()))

    def forward(self, x):
        """
        Forward pass
        Returns: (action, value)
        """
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Convolutional features
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        # Get action and value
        action = self.actor(features)
        value = self.critic(features)

        return action, value


class ReplayBuffer:
    """Simple replay buffer for storing experiences"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


def train_custom():
    """Custom training loop"""

    # Hyperparameters
    learning_rate = 3e-4
    gamma = 0.99
    batch_size = 32
    buffer_size = 10000
    num_episodes = 1000

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment and model
    env = OsuEnv()
    model = OsuCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)

    # Loss functions
    mse_loss = nn.MSELoss()

    print("\n" + "="*50)
    print("CUSTOM TRAINING")
    print("="*50)
    print("Start playing osu! songs when prompted")
    print("="*50 + "\n")

    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get action from model
            with torch.no_grad():
                action, _ = model(state_tensor)
                action = action.cpu().numpy()[0]

            # Add exploration noise
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, -1, 1)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1

            # Train if we have enough samples
            if len(replay_buffer) > batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Convert to tensors
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Compute predicted actions and values
                pred_actions, values = model(states)

                # Compute target values
                with torch.no_grad():
                    _, next_values = model(next_states)
                    target_values = rewards + gamma * next_values * (1 - dones)

                # Compute losses
                value_loss = mse_loss(values, target_values)
                policy_loss = mse_loss(pred_actions, actions)

                total_loss = value_loss + policy_loss

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if done:
                break

            state = next_state

        # Print episode stats
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_steps}")
        print(f"  300s: {info['hits_300']}, 100s: {info['hits_100']}, "
              f"50s: {info['hits_50']}, Misses: {info['misses']}")
        print()

        # Save model periodically
        if (episode + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/osu_custom_ep{episode+1}.pt")
            print(f"Model saved at episode {episode + 1}")

    # Save final model
    torch.save(model.state_dict(), "models/osu_custom_final.pt")
    print("Training complete!")
    env.close()


if __name__ == '__main__':
    train_custom()
