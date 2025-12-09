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
   def __init__(self, num_classes=5, window_size=150):
      super(SpikeCNN, self).__init__()

      self.conv1 = nn.Conv1d(
         in_channels=1,
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
   
def detect_peaks(y, dataset = 'D1'):
   threshold_coeffs = {
      "D1": 8,
      "D2": 6,
      "D3": 5,
      "D4": 4,
      "D5": 3,
      "D6": 2.5,
   }

   noise = np.median(np.abs(y)) / 0.6745
   y_abs = np.abs(y)
   threshold = threshold_coeffs.get(dataset, 10) * noise
   peaks, _ = find_peaks(y_abs, height=threshold, distance=2)
   return peaks

datasets = ('D1', )

for i, ds in enumerate(datasets, start=1):
   print(f"Processing dataset {ds} ({i}/{len(datasets)})")

   # Training 
   training_dataset = spio.loadmat(f'Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)
   training_data = training_dataset['d']
   indexes = training_dataset['Index']
   classes = training_dataset['Class']

   # Parameters
   Fs = 25000  # Sampling frequency (Hz)
   Ts = 1 / Fs  # Sampling period (s)
   N = len(training_data)  # Number of samples
   window_pre = 75  # Samples before peak
   window_post = 75  # Samples after peak
   window_size = window_pre + window_post  # Total window size

   # Sort spikes with their class
   classes = classes[np.argsort(indexes)]
   indexes = np.sort(indexes)

   # Extract spikes from training data using indexes
   spike_windows = []
   for idx in indexes:
      if idx - window_pre >= 0 and idx + window_post < N:
         w = training_data[idx - window_pre : idx + window_post]
         spike_windows.append(w)

   spike_windows = np.array(spike_windows)

   # Normalise spike waveforms
   spike_windows_normalised = []
   for w in spike_windows:
      w_norm = w - np.mean(w)
      w_norm = w_norm / (np.max(np.abs(w_norm)) + 1e-8)
      spike_windows_normalised.append(w_norm)

   spike_windows_normalised = np.array(spike_windows_normalised)

   # Use only normalised spike waveforms (1‑channel)
   spike_windows_tensor = torch.tensor(spike_windows_normalised[:, None, :], dtype=torch.float32)


   # 1D CNN Classifier
   # Build training data
   train_features = []
   train_labels = []

   for idx, cls in zip(indexes, classes):
      if idx - window_pre >= 0 and idx + window_post < N:
         w = training_data[idx - window_pre : idx + window_post]

         # Align
         peak_idx = np.argmax(np.abs(w))
         shift = window_pre - peak_idx
         aligned = np.roll(w, shift)

         # Normalise waveform
         aligned_norm = aligned - np.mean(aligned)
         aligned_norm = aligned_norm / (np.max(np.abs(aligned_norm)) + 1e-8)

         # 1‑channel input (waveform only)
         train_features.append(aligned_norm[None, :])
         train_labels.append(cls)

   train_x = torch.tensor(np.array(train_features), dtype=torch.float32)
   train_y = torch.tensor(np.array(train_labels) - 1, dtype=torch.long)

   # Training
   model = SpikeCNN(num_classes=5, window_size=150)
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

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


   # Dataset analysis

   # Import dataset for analysis
   data = spio.loadmat(f'Coursework/Coursework C/Coursework_C_Datasets/{ds}.mat', squeeze_me=True)

   data = data['d']

   num_classes = 5  # D1 has 5 neurons
   templates = []

   for cls in range(1, num_classes+1):
      class_idxs = indexes[classes == cls]
      spikes = []

      for idx in class_idxs:
         if idx - window_pre >= 0 and idx + window_post < len(data):
               w = data[idx - window_pre : idx + window_post]
               # Align spike on maximum absolute value
               peak_idx = np.argmax(np.abs(w))
               shift = window_pre - peak_idx
               aligned = np.roll(w, shift)
               # Normalize waveform
               aligned = aligned - np.mean(aligned)
               aligned = aligned / (np.max(np.abs(aligned)) + 1e-12)
               spikes.append(aligned)

      spikes = np.array(spikes)
      # Average over all spikes of this class
      template = np.mean(spikes, axis=0)
      templates.append(template)

   templates = np.array(templates)

   filtered_outputs = []

   # Convolve templates with recording
   for template in templates:
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
      while j < len(all_peaks) and abs(all_peaks[j][0] - all_peaks[i][0]) < 2:
         group.append(all_peaks[j])
         j += 1

      # Choose strongest template’s peak
      best_peak = max(group, key=lambda x: filtered_outputs[x[1], x[0]])

      # Cross‑template consistency check
      scores = np.array([filtered_outputs[t, best_peak[0]] for t in range(len(templates))])
      if np.max(scores) < 0.4 or np.sum(scores > 0.25) < 2:
         i = j
         continue

      merged.append(best_peak[0])

      i = j

   spike_times = np.array(merged)

   spike_windows = []

   for t in spike_times:
      # Reject spikes with too little amplitude
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

   # Prepare for CNN
   spike_windows_tensor = torch.tensor(spike_windows_normalised[:, None, :], dtype=torch.float32)

   # Inference
   model.eval()

   confidence_threshold = 0.7  # CNN confidence threshold

   with torch.no_grad():
      logits = model(spike_windows_tensor)
      probs = F.softmax(logits, dim=1)
      conf, preds = torch.max(probs, dim=1)

   # Apply threshold: keep only spikes where maximum class probability >= threshold
   mask = conf >= confidence_threshold

   # Filter spike times and classes
   Index_vec = spike_times[mask.cpu().numpy()]
   # Convert to class labels 1–5
   Class_vec = (preds[mask].cpu().numpy() + 1)  

   # Save results
   results_dir = 'Coursework/Coursework C/Coursework_C_Results'
   os.makedirs(results_dir, exist_ok=True)

   dataset_name = ds # 'D1', 'D2', ...
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

   if ds == 'D1':
      # Evaluation for D1
      D1_indexes = np.asarray(training_dataset['Index']).ravel()
      D1_classes = np.asarray(training_dataset['Class']).ravel()

      D1_predictions = spio.loadmat(save_path, squeeze_me=True)
      predicted_indexes = np.asarray(D1_predictions['Index']).ravel()
      predicted_classes = np.asarray(D1_predictions['Class']).ravel()

      print("D1 Assessment Results")
      # Ground truth
      GT_idx = D1_indexes
      GT_cls = D1_classes

      # Predictions
      PR_idx = predicted_indexes
      PR_cls = predicted_classes

      tolerance = 50

      TP = 0
      FP = 0
      FN = 0

      # Per-class counts
      TP_class = {c: 0 for c in range(1, 6)}
      FP_class = {c: 0 for c in range(1, 6)}
      FN_class = {c: 0 for c in range(1, 6)}

      used_gt = set()

      for p_idx, p_cls in zip(PR_idx, PR_cls):

          # Compute abs difference to all GT spikes
          diffs = np.abs(GT_idx - p_idx)
          min_diff = np.min(diffs)
          min_loc = np.argmin(diffs)

          # If no GT spike within tolerance → FP
          if min_diff > tolerance:
              FP += 1
              continue

          # GT spike already matched by another prediction → FP
          if min_loc in used_gt:
              FP += 1
              continue

          # Check class match
          if p_cls == GT_cls[min_loc]:
              TP += 1
              TP_class[p_cls] += 1
              used_gt.add(min_loc)
          else:
              # Correct location but wrong class
              FP += 1
              if int(p_cls) in FP_class:
                  FP_class[int(p_cls)] += 1

      for i, cls in enumerate(GT_cls):
          if i not in used_gt:
              FN += 1
              FN_class[cls] += 1

      print("Detection Performance:")
      print("TP:", TP)
      print("FP:", FP)
      print("FN:", FN)
      print("Detection Recall:", TP / (TP + FN))
      print("Detection Precision:", TP / (TP + FP))
      print("\nClassification Performance per Class:")
      for c in range(1, 6):
          tp_c = TP_class[c]
          fp_c = FP_class[c]
          fn_c = FN_class[c]
          recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
          precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
          print(f"Class {c}: TP={tp_c}, FP={fp_c}, FN={fn_c}, Recall={recall_c:.4f}, Precision={precision_c:.4f}")