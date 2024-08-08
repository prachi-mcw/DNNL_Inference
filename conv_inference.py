import torch
import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv1(x)

def read_mnist_binary(filename, num_images=1, image_size=24):
    with open(filename, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(-1, 1, image_size, image_size)
    return torch.from_numpy(images[:num_images]).float() / 255.0

# Load the model with the learned weights

model = SimpleCNN()
try:
    state_dict = torch.load('mnist_cnn_weights.pth')
    # Filter out unexpected keys
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    print("Loaded pre-trained weights (conv layer only)")
except FileNotFoundError:
    print("No pre-trained weights found, using initialized weights")

model.eval()

# Load images from binary file
input_images = read_mnist_binary('mnist_image.bin', num_images=1, image_size=24)

def get_output_for_mnist_image(image):
   
    with torch.no_grad():
        output_conv = model(image)
    
    return output_conv

# Perform inference on the loaded image
output_conv = get_output_for_mnist_image(input_images[0].unsqueeze(0))

# Convert the output to numpy array
output_conv_np = output_conv.detach().numpy()

# Save the output to a .bin file
output_conv_np.tofile('conv.bin')

print("Python output after Conv saved to conv.bin")

