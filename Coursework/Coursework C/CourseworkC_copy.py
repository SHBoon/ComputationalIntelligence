import numpy as np
import torch
import scipy.io as spio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import fftconvolve, find_peaks
import pywt
from torch.utils.data import TensorDataset, DataLoader
import os

# ==========================
# Global Threshold Lookups
# ==========================

THRESHOLDS_PEAK = {
    "D1": 6.0,
    "D2": 4.0,
    "D3": 2.5,
    "D4": 1.5,
    "D5": 1.0,
    "D6": 0.7,
}

THRESHOLDS_MF = {
    "D1": 0.03,
    "D2": 0.04,
    "D3": 0.05,
    "D4": 0.06,
    "D5": 0.07,
    "D6": 0.08,
}

THRESHOLDS_CNN = {
    "base": {
        "D1": 0.65,
        "D2": 0.60,
        "D3": 0.55,
        "D4": 0.50,
        "D5": 0.50,
        "D6": 0.50,
    },
    "scale": {
        "D1": 0.15,
        "D2": 0.15,
        "D3": 0.10,
        "D4": 0.08,
        "D5": 0.08,
        "D6": 0.05,
    }
}

# ================================
# USER-ADJUSTABLE THRESHOLD SECTION
# ================================

# --- Begin user-adjustable section ---

# Per-class thresholds (optional, for advanced tuning):
THRESHOLDS_CNN_CLASS = {
    "base": {
        "D1": {1: 0.65, 2: 0.65, 3: 0.70, 4: 0.40, 5: 0.65},
        "D2": {1: 0.60, 2: 0.60, 3: 0.80, 4: 0.65, 5: 0.55},
        "D3": {1: 0.50, 2: 0.55, 3: 0.90, 4: 0.60, 5: 0.30},
        "D4": {1: 0.45, 2: 0.40, 3: 0.90, 4: 0.50, 5: 0.30},
        "D5": {1: 0.40, 2: 0.30, 3: 0.95, 4: 0.45, 5: 0.15},
        "D6": {1: 0.45, 2: 0.35, 3: 0.995, 4: 0.50, 5: 0.20},
    }
}

THRESHOLDS_MF_CLASS = {
    "D1": {1: 0.03, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03},
    "D2": {1: 0.04, 2: 0.04, 3: 0.04, 4: 0.04, 5: 0.04},
    "D3": {1: 0.05, 2: 0.05, 3: 0.08, 4: 0.05, 5: 0.05},
    "D4": {1: 0.06, 2: 0.06, 3: 0.11, 4: 0.06, 5: 0.06},
    "D5": {1: 0.07, 2: 0.07, 3: 0.12, 4: 0.07, 5: 0.07},
    "D6": {1: 0.08, 2: 0.08, 3: 0.14, 4: 0.08, 5: 0.08},
}

THRESHOLDS_PEAK_CLASS = {
    "D1": {1: 6.0, 2: 6.0, 3: 6.0, 4: 6.0, 5: 6.0},
    "D2": {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0},
    "D3": {1: 4.5, 2: 4.5, 3: 4.5, 4: 4.5, 5: 4.5},
    "D4": {1: 3.5, 2: 3.5, 3: 3.5, 4: 3.5, 5: 3.5},
    "D5": {1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5},
    "D6": {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0},
}

# --- End user-adjustable section ---

class SpikeCNN(nn.Module):
   def __init__(self, num_classes=5, window_size=150):
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
   
def detect_peaks(y, dataset='D1', class_id=1):
   noise = np.median(np.abs(y)) / 0.6745
   y_abs = np.abs(y)
   threshold = THRESHOLDS_PEAK_CLASS[dataset][class_id] * noise
   peak_distance = {"D1":30,"D2":30,"D3":30,"D4":30,"D5":30,"D6":30}
   peaks, _ = find_peaks(y_abs, height=threshold, distance=peak_distance.get(dataset,20))
   return peaks

def add_noise_for_dataset(data, version, Fs):
   noise_levels = {
      "D1": 0.0,
      "D2": 0.1,
      "D3": 0.3,
      "D4": 0.7,
      "D5": 1.0,
      "D6": 1.5,
   }

   level = noise_levels.get(version, 0.0)

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

datasets = ('D1', 'D2', 'D3', 'D4', 'D5', 'D6')

for i, ds in enumerate(datasets, start=1):
   print(f"Processing dataset {ds} ({i}/{len(datasets)})")

   # Parameters
   Fs = 25000  # Sampling frequency (Hz)
   Ts = 1 / Fs  # Sampling period (s)
   window_pre = 100  # Samples before peak
   window_post = 100  # Samples after peak
   window_size = window_pre + window_post  # Total window size

   # Training 
   training_dataset = spio.loadmat(f'Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)
   training_data = training_dataset['d']
   # Apply noise augmentation to training data
   # Use the dataset version for consistent augmentation (D1..D6)
   training_data = add_noise_for_dataset(training_data, ds, Fs)
   indexes = training_dataset['Index']
   classes = training_dataset['Class']

   N = len(training_data)

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

         # --- Noise augmentation per spike ---
         aug = aligned_norm.copy()

         # Gaussian noise
         noise_std = 0.05
         aug += np.random.normal(0, noise_std, size=aug.shape)

         # Random amplitude scaling
         scale = 1.0 + 0.1 * (np.random.rand() - 0.5)
         aug *= scale

         # Random small time shift (±3 samples)
         shift_amount = np.random.randint(-3, 4)
         aug = np.roll(aug, shift_amount)

         # Renormalise
         aug = aug - np.mean(aug)
         aug = aug / (np.max(np.abs(aug)) + 1e-8)

         # Wavelet channel
         cA, cD = pywt.dwt(aug, 'db4')
         wavelet_rec = pywt.idwt(cA, cD, 'db4')
         wavelet_rec = wavelet_rec[:len(aug)]
         wavelet_rec = wavelet_rec - np.mean(wavelet_rec)
         wavelet_rec = wavelet_rec / (np.max(np.abs(wavelet_rec)) + 1e-8)

         two_channel = np.stack([aug, wavelet_rec], axis=0)
         train_features.append(two_channel)
         train_labels.append(cls)

   train_x = torch.tensor(np.array(train_features), dtype=torch.float32)
   train_y = torch.tensor(np.array(train_labels) - 1, dtype=torch.long)

   # Training
   model = SpikeCNN(num_classes=5, window_size=200)
   class_counts = np.bincount(train_y.numpy())
   class_weights = 1.0 / (class_counts + 1e-6)
   class_weights = class_weights / np.sum(class_weights)
   criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
   optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

   # Batch training
   batch_size = 32
   dataset = TensorDataset(train_x, train_y)
   loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

   num_epochs = 100

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

   num_classes = 5 
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

      # Sort spikes by peak amplitude
      peak_amplitudes = np.max(np.abs(spikes), axis=1)
      top_indices = np.argsort(peak_amplitudes)[-30:]
      template = np.mean(spikes[top_indices], axis=0)
      template = np.convolve(template, np.ones(5)/5, mode='same')
      templates.append(template)

   templates = np.array(templates)

   # Compute per-template amplitudes for dynamic thresholds
   template_amplitudes = np.max(np.abs(templates), axis=1)

   # Normalise amplitudes to 0–1 range
   amp_norm = (template_amplitudes - np.min(template_amplitudes)) / (np.max(template_amplitudes) - np.min(template_amplitudes) + 1e-12)

   # Map amplitudes to MF thresholds (stronger spikes → higher threshold)
   mf_dynamic = np.array([
       THRESHOLDS_MF_CLASS[ds][cls] + 0.18 * amp_norm[cls-1]
       for cls in range(1, 6)
   ])

   filtered_outputs = []

   # Convolve templates with recording
   for template in templates:
      template = template - np.mean(template)
      template = template / (np.max(np.abs(template)) + 1e-8)
      matched_filtered = fftconvolve(data, template[::-1], mode='same')
      matched_filtered = matched_filtered / np.max(np.abs(matched_filtered))
      filtered_outputs.append(matched_filtered)

   filtered_outputs = np.array(filtered_outputs)

   # Detect peaks in each filtered output
   peak_lists = [detect_peaks(filtered_outputs[t], ds, t+1) for t in range(len(filtered_outputs))]


   print("Peak counts per template:")
   for t, peaks in enumerate(peak_lists):
      print(f"Template {t+1}: {len(peaks)} peaks")

   # Merge peaks from all templates and sort
   merged = []

   all_peaks = [(p, t) for t, peaks in enumerate(peak_lists) for p in peaks]
   all_peaks.sort()

   i = 0
   while i < len(all_peaks):
      group = [all_peaks[i]]
      j = i + 1

      # Group peaks across templates
      while j < len(all_peaks) and abs(all_peaks[j][0] - all_peaks[i][0]) < 45:
         group.append(all_peaks[j])
         j += 1

      # Choose strongest template’s peak
      best_peak = max(group, key=lambda x: filtered_outputs[x[1], x[0]])

      # Cross‑template consistency check
      scores = np.array([filtered_outputs[t, best_peak[0]] for t in range(len(templates))])

      # Dynamic MF threshold based on template amplitude
      threshold_value = mf_dynamic[best_peak[1]]
      if np.max(scores) < threshold_value:
          i = j
          continue

      merged.append(best_peak[0])

      i = j

   spike_times = np.array(merged)

   spike_windows = []
   valid_spike_times = []

   for t in spike_times:
    if t - window_pre >= 0 and t + window_post < N:
      window = data[t - window_pre:t + window_post]
      spike_windows.append(window)
      valid_spike_times.append(t)

   spike_times = np.array(valid_spike_times)

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

   # Print number of detected spikes
   print(f"Detected {len(spike_windows_normalised)} spikes in dataset {ds}")

   # Prepare for CNN
   wavelet_channels = []
   for w in spike_windows_normalised:
       cA, cD = pywt.dwt(w, 'db4')
       wavelet_rec = pywt.idwt(cA, cD, 'db4')
       wavelet_rec = wavelet_rec[:len(w)]
       wavelet_rec = wavelet_rec - np.mean(wavelet_rec)
       wavelet_rec = wavelet_rec / (np.max(np.abs(wavelet_rec)) + 1e-8)
       wavelet_channels.append(wavelet_rec)

   wavelet_channels = np.array(wavelet_channels)
   combined_inputs = np.stack([spike_windows_normalised, wavelet_channels], axis=1)
   spike_windows_tensor = torch.tensor(combined_inputs, dtype=torch.float32)

   # Inference
   model.eval()

   with torch.no_grad():
      logits = model(spike_windows_tensor)
      probs = F.softmax(logits, dim=1)
      conf, preds = torch.max(probs, dim=1)

   cnn_thresholds_np = np.array([
       THRESHOLDS_CNN_CLASS["base"][ds][cls] +
       THRESHOLDS_CNN["scale"][ds] * amp_norm[cls-1]
       for cls in range(1, 6)
   ])
   cnn_thresholds = torch.tensor(cnn_thresholds_np, dtype=torch.float32)
   mask = conf >= cnn_thresholds[preds]

   print("CNN raw predictions:", np.bincount(preds.cpu().numpy() + 1))

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