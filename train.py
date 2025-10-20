import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

from configs import CONFIGS
print("Using device: ", CONFIGS["device"])

if not os.path.exists('data'):
    os.makedirs('data')

from models.mlp import MLP

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=CONFIGS["batch_size"],
    shuffle=True
)

model = MLP(input_dim=28*28, hidden_dim=50, output_dim=10).to(CONFIGS["device"])
optimizer = optim.SGD(model.parameters(), lr=CONFIGS["learning_rate"])
criterion = nn.CrossEntropyLoss()

weight_trajectory = []

print("Starting training...")
for epoch in range(1, CONFIGS["epochs"] + 1):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(CONFIGS["device"]), target.to(CONFIGS["device"])

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch >= CONFIGS["exploration_epoch"]:
            
            weights = model.fc2.weight.data.clone().detach().cpu().numpy()
            weight_trajectory.append(weights.flatten())

    print(f'Epoch {epoch}/{CONFIGS["epochs"]}, Loss: {loss.item():.4f}')


print("\nsaving weight trajectory")
weight_trajectory_np = np.array(weight_trajectory)  
np.save('data/weight_trajectory.npy', weight_trajectory_np)

torch.save(model.state_dict(), 'data/final_model.pth')

print(f"weight trajectory shape: {weight_trajectory_np.shape}")
