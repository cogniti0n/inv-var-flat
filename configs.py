import torch

CONFIGS = {
    "hidden_dim": 50,
    "epochs": 50,
    "exploration_epoch": 10,
    "learning_rate": 0.1,
    "batch_size": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}