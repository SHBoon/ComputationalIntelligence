import numpy as np
import scipy.io as spio
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
import torch.optim as optim

# Filtering functions
def butter_bandpass(lowcut, highcut, fs, order):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a

def apply_bandpass_filter(data, lowcut=2, highcut=10000, fs=25000, order = 2):
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   y = filtfilt(b, a, data)
   return y

mat = spio.loadmat('Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)

data = mat['d']
indexes = mat['Index']
classes = mat['Class']

Fs = 25000  # Sampling frequency (Hz)
Ts = 1 / Fs  # Sampling period (s)
N = len(data)  # Number of samples

# Apply bandpass filter to the data
filtered_data = apply_bandpass_filter(data, lowcut=2, highcut=10000, fs=Fs, order=3)

# Training data preparation
X_train = []
y_train = []

spike_length = int(0.002 / Ts)
pre = 2 * spike_length // 3
post = spike_length - pre

for i, idx in enumerate(indexes):
    if idx - pre >= 0 and idx + post < len(filtered_data):
        snippet = filtered_data[idx-pre : idx+post]
        X_train.append(snippet)
        y_train.append(classes[i] - 1)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
X_train = X_train / (np.std(X_train, axis=1, keepdims=True) + 1e-8)

X_train = X_train.reshape((-1, X_train.shape[1], 1))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Dataset and DataLoader
dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Define 1D CNN model
class SpikeCNN(nn.Module):
   def __init__(self, input_length, num_classes=5):
      super(SpikeCNN, self).__init__()
      self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
      self.pool1 = nn.MaxPool1d(2)
      self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
      self.pool2 = nn.MaxPool1d(2)
      self.conv3 = nn.Conv1d(32, 64, kernel_size=3)
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(self._get_conv_output(input_length), 64)
      self.fc2 = nn.Linear(64, num_classes)
      
   def _get_conv_output(self, input_length):
      x = torch.zeros(1, 1, input_length)
      x = self.pool1(self.conv1(x))
      x = self.pool2(self.conv2(x))
      x = self.conv3(x)
      return int(torch.numel(x))
   
   def forward(self, x):
      x = x.permute(0, 2, 1)  # reshape (batch, channels, length)
      x = torch.relu(self.conv1(x))
      x = self.pool1(x)
      x = torch.relu(self.conv2(x))
      x = self.pool2(x)
      x = torch.relu(self.conv3(x))
      x = self.flatten(x)
      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x

# Instantiate model
model = SpikeCNN(input_length=X_train.shape[1], num_classes=5)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")