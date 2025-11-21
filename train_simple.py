"""
Simple training script using Stable-Baselines3
This is the easiest way to get started with training your osu! AI
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from osu_env_improved import OsuEnv


def make_env():
    """Create and return the environment"""
    return OsuEnv()


def train():
    """Train the osu! AI using PPO algorithm"""

    # Create directories for saving
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create environment
    print("Creating environment...")
    env = DummyVecEnv([make_env])

    # Create callbacks for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path="./models/",
        name_prefix="osu_ppo"
    )

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "CnnPolicy",           # Use CNN to process screen images
        env,
        verbose=1,
        learning_rate=3e-4,    # Learning rate
        n_steps=2048,          # Steps per update
        batch_size=64,         # Batch size
        n_epochs=10,           # Training epochs per update
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # GAE lambda
        clip_range=0.2,        # PPO clip range
        tensorboard_log="./logs/",
        device="auto"          # Use GPU if available
    )

    # Print model info
    print("\nModel architecture:")
    print(model.policy)

    # Train the model
    print("\n" + "="*50)
    print("TRAINING INSTRUCTIONS:")
    print("="*50)
    print("1. Make sure gosumemory is running")
    print("2. Open osu!")
    print("3. When prompted, start playing a song")
    print("4. The AI will take control of your mouse")
    print("5. Let it play through multiple songs")
    print("6. Press Ctrl+C to stop training")
    print("="*50 + "\n")

    try:
        model.learn(
            total_timesteps=1000000,  # Train for 1M steps
            callback=checkpoint_callback,
            progress_bar=True
        )

        # Save final model
        print("\nSaving final model...")
        model.save("models/osu_ppo_final")
        print("Training complete!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current model...")
        model.save("models/osu_ppo_interrupted")
        print("Model saved!")

    finally:
        env.close()


def test_model(model_path="models/osu_ppo_final.zip"):
    """Test a trained model"""

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create environment
    env = OsuEnv(render_mode='human')

    print("\nTesting model...")
    print("Start a song in osu!")

    try:
        obs, info = env.reset()
        episode_reward = 0

        while True:
            # Get action from model (deterministic=True for testing)
            action, _states = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            env.render()

            if terminated or truncated:
                print(f"\nEpisode finished!")
                print(f"Total reward: {episode_reward}")
                print(f"Stats: {info}")
                break

    finally:
        env.close()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else "models/osu_ppo_final.zip"
        test_model(model_path)
    else:
        # Training mode
        train()
