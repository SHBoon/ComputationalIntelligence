import numpy as np
import torch
import scipy.io as spio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import fftconvolve, find_peaks
from torch.utils.data import TensorDataset, DataLoader
import pywt
import os

class SpikeCNN(nn.Module):
   def __init__(self, num_classes=5, window_size=100):
      super(SpikeCNN, self).__init__()

      self.conv1 = nn.Conv1d(
         in_channels=2,
         out_channels=16,
         kernel_size=5,
         padding=2
      )
      self.pool1 = nn.MaxPool1d(kernel_size=2)

      self.conv2 = nn.Conv1d(
         in_channels=16,
         out_channels=32,
         kernel_size=5,
         padding=2
      )
      self.pool2 = nn.MaxPool1d(kernel_size=2)

      self.flattened = 32 * (window_size // 4)

      self.fc1 = nn.Linear(self.flattened, 64)
      self.fc2 = nn.Linear(64, num_classes)

   def forward(self, x):
      x = self.pool1(F.relu(self.conv1(x)))
      x = self.pool2(F.relu(self.conv2(x)))
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

from scipy.signal import find_peaks

def detect_peaks(y, dataset = 'd1'):
   noise = np.median(np.abs(y)) / 0.6745
   threshold = 10 * noise
   peaks, _ = find_peaks(y, height=threshold, distance=20)
   return peaks

def extract_wavelet_coeffs(window, wavelet='db4', level=3):
   coeffs = pywt.wavedec(window, wavelet, level=level)
   # Flatten all coefficients into one vector
   return np.concatenate(coeffs)

def add_noise_for_dataset(data, version, Fs):
   noise_levels = {
      "d1": 0.0,
      "d2": 0.1,
      "d3": 0.3,
      "d4": 0.7,
      "d5": 1.0,
      "d6": 1.5,
   }

   level = noise_levels.get(version.lower(), 0.0)

   if level == 0:
      return data  # no noise for D1

   # Gaussian noise
   gaussian = np.random.normal(0, np.std(data) * level, size=data.shape)

   # Band‑limited coloured noise
   freqs = np.fft.rfftfreq(len(data), 1/Fs)
   spectrum = np.exp(-((freqs - 2000)**2) / (2 * (1500**2)))
   phases = np.exp(1j * np.random.uniform(0, 2*np.pi, len(freqs)))
   coloured = np.fft.irfft(spectrum * phases)
   coloured = coloured / np.std(coloured)

   noisy = data + gaussian + 0.5 * level * coloured
   return noisy

datasets = ('d1', 'd2', 'd3', 'd4', 'd5', 'd6')

for i, ds in enumerate(datasets, start=1):
   print(f"Processing dataset {ds.upper()} ({i}/{len(datasets)})")
   dataset = ds

   training_mat = spio.loadmat('Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)

   # Import dataset
   mat = spio.loadmat(f'Coursework/Coursework C/Coursework_C_Datasets/{dataset.upper()}.mat', squeeze_me=True)

   Fs = 25000  # Sampling frequency (Hz)

   training_data = add_noise_for_dataset(training_mat['d'], dataset, Fs)
   indexes = training_mat['Index']
   classes = training_mat['Class']

   data = mat['d']

   # Sort spikes with their class
   classes = classes[np.argsort(indexes)]
   indexes = np.sort(indexes)

   Fs = 25000  # Sampling frequency (Hz)
   Ts = 1 / Fs  # Sampling period (s)
   N = len(data)  # Number of samples

   # Matched filtering using pre-computed templates
   templates_raw = np.load('Coursework/Coursework C/templates.npy', allow_pickle=True)

   filtered_outputs = []

   # Convolve templates with recording
   for template in templates_raw:
      template = template - np.mean(template)
      template = template / np.linalg.norm(template)
      matched_filtered = fftconvolve(data, template[::-1], mode='same')
      matched_filtered = matched_filtered / np.max(np.abs(matched_filtered))
      filtered_outputs.append(matched_filtered)

   filtered_outputs = np.array(filtered_outputs)

   # Detect peaks in each filtered output
   peak_lists = [detect_peaks(y, ds) for y in filtered_outputs]

   # Merge peaks from all templates and sort
   merged = []

   all_peaks = [(p, t) for t, peaks in enumerate(peak_lists) for p in peaks]
   all_peaks.sort()

   i = 0
   while i < len(all_peaks):
      group = [all_peaks[i]]
      j = i + 1

      # Group peaks across templates
      while j < len(all_peaks) and abs(all_peaks[j][0] - all_peaks[i][0]) < 5:
         group.append(all_peaks[j])
         j += 1

      # Choose strongest template’s peak
      best_peak = max(group, key=lambda x: filtered_outputs[x[1], x[0]])  
      merged.append(best_peak[0])

      i = j

   spike_times = np.array(merged)

   # Refractory period filtering
   cleaned_spikes = []
   last_spike = -1e9
   for p in spike_times:
      if p - last_spike > 30:
         cleaned_spikes.append(p)
         last_spike = p
   spike_times = np.array(cleaned_spikes)

   # Extract spike windows
   window_pre = 50
   window_post = 50
   window_size = window_pre + window_post

   spike_windows = []

   for t in spike_times:
      # Reject spikes with too little amplitude
      if np.max(np.abs(data[t - window_pre:t + window_post])) < 4 * (np.median(np.abs(data)) / 0.6745):
         continue
      if t - window_pre >= 0 and t + window_post < N:
         window = data[t - window_pre:t + window_post]
         spike_windows.append(window)

   # Align spikes
   spike_windows_aligned = []
   for window in spike_windows:
      peak_idx = np.argmax(np.abs(window))
      shift = window_pre - peak_idx
      aligned_window = np.roll(window, shift)
      spike_windows_aligned.append(aligned_window)

   spike_windows_aligned = np.array(spike_windows_aligned)

   # Normalise spikes
   spike_windows_aligned = np.array(spike_windows_aligned)
   spike_windows_normalised = spike_windows_aligned - np.mean(spike_windows_aligned, axis=1, keepdims=True)
   spike_windows_normalised = spike_windows_normalised / np.max(np.abs(spike_windows_normalised), axis=1, keepdims=True)

   # Wavelet transform
   wavelet_features = []
   for w in spike_windows_normalised:
      coeffs = extract_wavelet_coeffs(w)
      if len(coeffs) < window_size:
         coeffs = np.pad(coeffs, (0, window_size - len(coeffs)))
      else:
         coeffs = coeffs[:window_size]
      wavelet_features.append(coeffs)

   wavelet_features = np.array(wavelet_features)

   combined = np.stack((spike_windows_normalised, wavelet_features), axis=1)

   # Prepare for CNN
   spike_windows_tensor = torch.tensor(combined, dtype=torch.float32)

   # 1D CNN Classifier
   # Build training data
   train_features = []
   train_labels = []

   for idx, cls in zip(indexes, classes):
      if idx - window_pre >= 0 and idx + window_post < N:
         w = data[idx - window_pre : idx + window_post]

         # Align
         peak_idx = np.argmax(np.abs(w))
         shift = window_pre - peak_idx
         aligned = np.roll(w, shift)

         # Normalise waveform
         aligned_norm = aligned - np.mean(aligned)
         aligned_norm = aligned_norm / (np.max(np.abs(aligned_norm)) + 1e-8)

         # Wavelet coefficients
         coeffs = extract_wavelet_coeffs(aligned_norm)
         if len(coeffs) < window_size:
            coeffs = np.pad(coeffs, (0, window_size - len(coeffs)))
         else:
            coeffs = coeffs[:window_size]

         # Combine into 2‑channel input
         two_channel = np.stack((aligned_norm, coeffs), axis=0)
         train_features.append(two_channel)
         train_labels.append(cls)

   train_x = torch.tensor(np.array(train_features), dtype=torch.float32)
   train_y = torch.tensor(np.array(train_labels) - 1, dtype=torch.long)

   # Training
   model = SpikeCNN(num_classes=5, window_size=100)
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

   # Batch training
   batch_size = 32
   dataset = TensorDataset(train_x, train_y)
   loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

   num_epochs = 200

   for epoch in range(num_epochs):
      model.train()
      epoch_loss = 0.0

      for batch_x, batch_y in loader:
         optimizer.zero_grad()

         outputs = model(batch_x)
         loss = criterion(outputs, batch_y)
         loss.backward()
         optimizer.step()

         epoch_loss += loss.item()

      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

   # Inference
   model.eval()

   with torch.no_grad():
      logits = model(spike_windows_tensor)
      probs = F.softmax(logits, dim=1)
      confidence, raw_pred = torch.max(probs, dim=1)
      # Apply confidence threshold: classify as noise (0) if confidence < 0.7
      raw_pred = raw_pred + 1
      pred_classes = torch.where(confidence > 0.7, raw_pred, torch.tensor(0))
      # pred_classes = torch.argmax(logits, dim=1).cpu().numpy() + 1  # Convert to 1–5

   Index_vec = spike_times
   Class_vec = pred_classes

   # Save results# Save Index and Class predictions to a .mat file named for the dataset (e.g. D1.mat)
   results_dir = 'Coursework/Coursework C/Coursework_C_Results'
   os.makedirs(results_dir, exist_ok=True)

   dataset_name = ds.upper()  # 'D1', 'D2', ...
   save_path = os.path.join(results_dir, f'{dataset_name}.mat')

   # Convert Class_vec (torch tensor or tensor-like) and Index_vec to numpy arrays
   if isinstance(Class_vec, torch.Tensor):
      class_np = Class_vec.cpu().numpy()
   else:
      class_np = np.asarray(Class_vec)

   index_np = np.asarray(Index_vec)

   # Ensure integer dtypes for class/index
   try:
      class_np = class_np.astype(np.int32)
   except Exception:
      pass
   try:
      index_np = index_np.astype(np.int64)
   except Exception:
      pass

   spio.savemat(save_path, {'Index': index_np, 'Class': class_np})
   print(f"Saved predictions to: {save_path}")