# Quick Start Guide - osu! AI

## Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and Run gosumemory
1. Download from: https://github.com/l3lackShark/gosumemory/releases
2. Extract and run `gosumemory.exe`
3. Open osu! - gosumemory will connect automatically
4. Verify connection at http://localhost:24050/

### 3. Test Your Setup
```bash
# Test the environment
python osu_env_improved.py
```

This will:
- Wait for you to start a song
- Take random actions
- Display what the AI sees

---

## Training Options

### Option 1: Simple Training (Recommended for Beginners)
Uses Stable-Baselines3 PPO - just run and let it train!

```bash
python train_simple.py
```

**What to expect:**
- Training will start when you play a song
- The AI will control your mouse
- Progress is saved every 10k steps to `models/`
- Use TensorBoard to monitor: `tensorboard --logdir=./logs/`

**To test your trained model:**
```bash
python train_simple.py test models/osu_ppo_final.zip
```

### Option 2: Custom Training (Advanced)
More control over the training process using raw PyTorch

```bash
python train_custom.py
```

---

## Training Tips

### For Best Results:

1. **Start with Easy Songs**
   - 1-2 star difficulty
   - Short duration (1-2 minutes)
   - Consistent patterns

2. **Training Strategy**
   - Train on the same song 10-20 times first
   - Then introduce variety
   - Gradually increase difficulty

3. **Monitoring Progress**
   ```bash
   # In another terminal
   tensorboard --logdir=./logs/
   ```
   Then visit http://localhost:6006/

4. **Expected Timeline**
   - First few episodes: Random clicking
   - After 50-100 episodes: Starts tracking circles
   - After 500-1000 episodes: Can hit some notes
   - After 5000+ episodes: Decent accuracy on trained songs

---

## Troubleshooting

### "Waiting for game to start..."
- Make sure gosumemory is running
- Open osu! and start playing a song
- Check gosumemory status at http://localhost:24050/

### Mouse movement is jerky
- This is normal initially
- The AI learns smoother movement over time
- You can add action smoothing in the environment

### Training is very slow
- Use shorter songs
- Reduce screen resolution in `OsuEnv`
- Use GPU if available (automatically detected)
- Train on easier patterns first

### AI keeps missing
- This is expected at first
- Ensure reward function is working (check logs)
- Try reducing difficulty
- May need more training time

---

## Project Structure

```
osuai/
├── TUTORIAL.md              # Comprehensive guide
├── QUICKSTART.md           # This file
├── requirements.txt        # Dependencies
├── osu_env_improved.py     # Main environment
├── train_simple.py         # Easy training with SB3
├── train_custom.py         # Advanced PyTorch training
├── project/                # Your original code
│   ├── main.py
│   ├── data_get.py
│   ├── data_upload.py
│   └── screen.py
└── models/                 # Saved models (created during training)
```

---

## Next Steps After Training

1. **Evaluate Performance**
   - Test on different songs
   - Track accuracy metrics
   - Compare human vs AI scores

2. **Improve the AI**
   - Add frame stacking (temporal information)
   - Implement attention mechanisms
   - Use imitation learning from human gameplay
   - Add audio processing

3. **Advanced Features**
   - Curriculum learning (progressive difficulty)
   - Multi-task learning (different game modes)
   - Transfer learning between songs

---

## Common Issues

### ImportError for pyautogui
```bash
pip install pyautogui
# On Linux you may also need:
sudo apt-get install python3-tk python3-dev
```

### CUDA out of memory
Reduce batch size in the training script:
```python
batch_size=32  # Change to 16 or 8
```

### AI doesn't improve
- Check reward function is working
- Ensure actions are being executed
- Verify gosumemory is providing data
- Try simpler songs

---

## Resources

- Full tutorial: See `TUTORIAL.md`
- gosumemory: https://github.com/l3lackShark/gosumemory
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- RL Glossary: https://spinningup.openai.com/

---

**Ready to start?**

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run gosumemory
# (Download and run from their releases)

# 3. Train!
python train_simple.py
```

Good luck! 🎮🤖
