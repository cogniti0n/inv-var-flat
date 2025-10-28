import torch

CONFIGS = {
    "hidden_dim": 50,
    "epochs": 50,
    "cutoff_epoch": 40,
    "learning_rate": 0.1,
    "batch_size": 64,
    "num_components": 500,
    "loss_sample_batch_size": 5000,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}