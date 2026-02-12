# AlphaZero Gomoku (Threat-Enhanced Version)

This project is a modified implementation of AlphaZero for Gomoku (9×9),  
built on top of:

https://github.com/junxiaosong/AlphaZero_Gomoku

The main purpose of this version is to experiment with adding simple handcrafted tactical features to the input representation, while keeping the original AlphaZero self-play framework.

---

## What Was Added

Compared to the original implementation:

- Replaced the backbone with a deeper ResNet structure
- Expanded input channels from basic board planes to 8 channels
- Added threat feature maps (three-in-a-row and four-in-a-row detection)
- Adjusted the reinforcement training pipeline for stronger evaluation

---

## State Representation (8 Channels)

The network input is:

8 × 9 × 9

Channels:

0. Current player's stones  
1. Opponent's stones  
2. Last move  
3. Player indicator  
4. Current player's three-in-a-row threat map  
5. Opponent's three-in-a-row threat map  
6. Current player's four-in-a-row threat map  
7. Opponent's four-in-a-row threat map  

Threat maps are computed from the current board by checking tactical patterns along horizontal, vertical, and diagonal directions.

---

## Network Architecture

Input: 8 × 9 × 9  

Backbone:

- Initial Conv layer (8 → 64)
- 10 Residual Blocks (64 channels)

Policy Head:

- 1×1 Conv (64 → 2)
- Fully connected layer → 81 moves
- Log-softmax output

Value Head:

- 1×1 Conv (64 → 1)
- Fully connected → 256
- Final scalar output with tanh

Loss:

MSE(value) + CrossEntropy(policy)

---

## Training Pipeline

Training follows the AlphaZero self-play loop:

1. Self-play using MCTS + neural network
2. Store game data into replay buffer
3. Data augmentation (rotation + reflection)
4. Mini-batch training
5. KL-divergence monitoring
6. Update best model if performance improves

### 1. Train Base Model

First train the initial model:

```bash
python train.py
```

### 2. Further Reinforcement Training

Continue improving the model with reinforcement self-play:

```bash
python reinforcement_learning.py
```

### 3. Monitor Model Strength

Evaluate model performance:

```bash
python test.py
```

### 4. Human vs AI

Play against the trained model:

```bash
python human_play.py
```

---

## Dependencies

Tested with:

- Python 3.8
- PyTorch 1.10+
- NumPy 1.21+
- tqdm

You can install basic requirements with:

```bash
pip install torch numpy tqdm
```

GPU is optional but recommended for faster training.

---

## Practical Advantages Observed

Adding simple tactical feature planes can help:

- Improve early training efficiency
- Stabilize mid-game tactical behavior
- Reduce reliance on extremely deep MCTS search

The goal is not to replace self-play learning, but to introduce lightweight inductive bias.

---

## Self-Play Demonstration

Below is a self-play game from the currently strongest trained model.

![test](https://github.com/user-attachments/assets/14b5e6b4-807c-4e7b-a99f-9050a02457dc)

---

## Notes

This implementation is meant for experimentation on single-machine environments.  
It does not include distributed training or large-scale optimization.

The base framework and core idea come from:

https://github.com/junxiaosong/AlphaZero_Gomoku

This repository simply extends it with additional feature channels and architectural changes.
