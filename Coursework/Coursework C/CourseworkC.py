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

# Per-class thresholds (optional, for advanced tuning):
THRESHOLDS_CNN_CLASS = {
    "base": {
        "D1": {1: 0.65, 2: 0.65, 3: 0.70, 4: 0.40, 5: 0.65},
        "D2": {1: 0.60, 2: 0.60, 3: 0.80, 4: 0.65, 5: 0.55},
        "D3": {1: 0.50, 2: 0.55, 3: 0.90, 4: 0.60, 5: 0.30},
        "D4": {1: 0.45, 2: 0.40, 3: 0.90, 4: 0.50, 5: 0.30},
        "D5": {1: 0.40, 2: 0.30, 3: 0.95, 4: 0.45, 5: 0.15},
        "D6": {1: 0.45, 2: 0.35, 3: 0.90, 4: 0.50, 5: 0.20},
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
    "D4": {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0},
    "D5": {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0, 5: 3.0},
    "D6": {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0},
}

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

# From your noise analysis (rounded)
BASELINE_STD = 0.664  # ≈ std of D1 baseline noise
NOISE_STD_ABS = {
    "D1": 0.0,     # we'll special-case this
    "D2": 1.187,   # std(D2 - D1)
    "D3": 1.2525,  # std(D3 - D1)
    "D4": 1.6846,
    "D5": 2.9619,
    "D6": 4.3965,
}

# Ratios relative to D1 baseline
NOISE_STD_RATIO = {
    k: (v / BASELINE_STD if k != "D1" else 0.0)
    for k, v in NOISE_STD_ABS.items()
}

# Desired fraction of power in each band (0–300, 300–3k, 3–10k),
# roughly matching your measured percentages.
BAND_WEIGHTS = {
    "D2": (0.45, 0.40, 0.15),
    "D3": (0.38, 0.38, 0.24),
    "D4": (0.26, 0.33, 0.41),
    "D5": (0.12, 0.25, 0.63),
    "D6": (0.08, 0.24, 0.68),
}


def add_noise_for_dataset(data, version, Fs):
   """
   Data-driven noise model:
   - amplitude scaled using NOISE_STD_RATIO (from your analysis)
   - frequency shape set by BAND_WEIGHTS (low/mid/high bands)
   - slightly heavy-tailed noise for D2/D3 via Laplace mixing
   """
   if version == "D1":
      return data  # no extra noise for clean D1

   # 1) Decide overall target noise std relative to the signal
   # Your spike snippets are usually normalised to ~1 peak,
   # so we use a base sigma and scale by the dataset ratio.
   base_sigma = 0.10  # tune this global knob if needed
   ratio = NOISE_STD_RATIO.get(version, 1.0)
   target_std = base_sigma * ratio

   N = len(data)

   # 2) Start with white Gaussian noise (unit variance)
   white = np.random.normal(0.0, 1.0, size=N)

   # 3) Shape it in frequency to match band weights
   freqs = np.fft.rfftfreq(N, 1.0 / Fs)
   H = np.zeros_like(freqs)

   # Band edges
   f1, f2, f3, f4 = 0.0, 300.0, 3000.0, 10000.0
   w_low, w_mid, w_high = BAND_WEIGHTS.get(version, (0.4, 0.4, 0.2))

   low_mask = (freqs >= f1) & (freqs < f2)
   mid_mask = (freqs >= f2) & (freqs < f3)
   high_mask = (freqs >= f3) & (freqs <= f4)

   H[low_mask] = w_low
   H[mid_mask] = w_mid
   H[high_mask] = w_high

   # Avoid all-zero H if version not found
   if np.all(H == 0):
      H += 1.0

   # Random phases, magnitude = H
   phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=len(freqs)))
   spectrum = H * phases

   coloured = np.fft.irfft(spectrum, n=N)
   coloured = coloured / (np.std(coloured) + 1e-12)

   # 4) Combine white + coloured
   noise = 0.5 * white + 0.5 * coloured

   # 5) D2/D3: add a tiny heavy-tailed component (Laplace) to bump kurtosis
   if version in ("D2", "D3"):
      lap = np.random.laplace(loc=0.0, scale=1.0, size=N)
      lap = lap / (np.std(lap) + 1e-12)
      noise = 0.8 * noise + 0.2 * lap

   # 6) Scale to target std and add
   noise = noise / (np.std(noise) + 1e-12) * target_std
   return data + noise

def build_templates_from_clean(data_clean, GT_idx, GT_cls,
                               window_pre=100, window_post=100):
    N = len(data_clean)
    num_classes = 5
    templates = []

    for cls in range(1, num_classes + 1):
        class_idxs = GT_idx[GT_cls == cls]
        spikes = []

        for idx in class_idxs:
            if idx - window_pre >= 0 and idx + window_post < N:
                w = data_clean[idx - window_pre: idx + window_post]

                peak_idx = np.argmax(np.abs(w))
                shift = window_pre - peak_idx
                aligned = np.roll(w, shift)

                aligned = aligned - np.mean(aligned)
                aligned = aligned / (np.max(np.abs(aligned)) + 1e-12)
                spikes.append(aligned)

        spikes = np.array(spikes)
        if spikes.size == 0:
            templates.append(np.zeros(window_pre + window_post))
            continue

        peak_amplitudes = np.max(np.abs(spikes), axis=1)
        top_indices = np.argsort(peak_amplitudes)[-30:]
        template = np.mean(spikes[top_indices], axis=0)

        template = np.convolve(template, np.ones(5) / 5, mode='same')
        templates.append(template)

    templates = np.array(templates)

    template_amplitudes = np.max(np.abs(templates), axis=1)
    amp_norm = (template_amplitudes - np.min(template_amplitudes)) / (
        np.max(template_amplitudes) - np.min(template_amplitudes) + 1e-12
    )

    return templates, amp_norm


def build_d1_training_set(data_clean, indexes, classes,
                          Fs=25000, window_pre=100, window_post=100):
    """
    Build training features/labels from *clean* D1, using a
    two-channel (waveform + wavelet) representation and per-spike
    augmentation with realistic dataset-level noise.
    """
    N = len(data_clean)
    train_features = []
    train_labels = []

    for idx, cls in zip(indexes, classes):
        if idx - window_pre < 0 or idx + window_post >= N:
            continue

        w = data_clean[idx - window_pre: idx + window_post]

        # Align on peak
        peak_idx = np.argmax(np.abs(w))
        shift = window_pre - peak_idx
        aligned = np.roll(w, shift)

        # Normalise waveform
        aligned_norm = aligned - np.mean(aligned)
        aligned_norm = aligned_norm / (np.max(np.abs(aligned_norm)) + 1e-8)

        # --- Per-spike augmentation ---
        # Start from the aligned, normalised waveform
        aug = aligned_norm.copy()

        # With high probability, apply a realistic dataset-level noise profile
        if np.random.rand() < 0.7:
            # Choose a random "dataset" noise level (exclude D1 = clean)
            ds_choice = np.random.choice(["D2", "D3", "D4", "D5"])
            aug_noisy = add_noise_for_dataset(aug.copy(), ds_choice, Fs)
            aug = aug_noisy
        else:
            # Gaussian noise
            noise_std = 0.05
            aug += np.random.normal(0, noise_std, size=aug.shape)

        # Random amplitude scaling
        scale = 1.0 + 0.1 * (np.random.rand() - 0.5)
        aug *= scale

        # Random small time shift (±3 samples)
        shift_amount = np.random.randint(-3, 4)
        aug = np.roll(aug, shift_amount)

        # Renormalise after all augmentations
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
    return train_x, train_y


def train_cnn_on_d1(train_x, train_y, window_size=200, num_epochs=100):
    """
    Train SpikeCNN on the D1 training set.
    """
    model = SpikeCNN(num_classes=5, window_size=window_size)

    class_counts = np.bincount(train_y.numpy())
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / np.sum(class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

        print(f"[Train] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model


def run_full_pipeline_on_data(data, ds_label, model, GT_idx, GT_cls, templates, amp_norm, Fs=25000, window_pre=100, window_post=100):
   """
   Run the full MF + CNN pipeline on 'data' for a given dataset label,
   using shared templates and thresholds, and evaluate against the
   D1 ground truth (GT_idx, GT_cls). Returns the detected spike
   indexes and classes plus a metrics dictionary.
   """
   N = len(data)

   mf_dynamic = np.array([
      THRESHOLDS_MF_CLASS[ds_label][cls] + 0.18 * amp_norm[cls - 1]
      for cls in range(1, 6)
   ])

   # -----------------------------
   # 1) Matched filtering
   # -----------------------------
   filtered_outputs = []

   for template in templates:
      template = template - np.mean(template)
      template = template / (np.max(np.abs(template)) + 1e-8)
      matched_filtered = fftconvolve(data, template[::-1], mode='same')
      matched_filtered = matched_filtered / (np.max(np.abs(matched_filtered)) + 1e-12)
      filtered_outputs.append(matched_filtered)

   filtered_outputs = np.array(filtered_outputs)

   # Peak detection per template
   peak_lists = [
      detect_peaks(filtered_outputs[t], ds_label, t + 1)
      for t in range(len(filtered_outputs))
   ]

   print(f"[{ds_label}] Peak counts per template:")
   for t, peaks in enumerate(peak_lists):
      print(f"  Template {t+1}: {len(peaks)} peaks")

   # -----------------------------
   # 2) Merge peaks & MF threshold
   # -----------------------------
   all_peaks = [(p, t) for t, peaks in enumerate(peak_lists) for p in peaks]
   all_peaks.sort()

   merged = []
   i = 0

   while i < len(all_peaks):
      group = [all_peaks[i]]
      j = i + 1

      # Group peaks within ±45 samples
      while j < len(all_peaks) and abs(all_peaks[j][0] - all_peaks[i][0]) < 45:
         group.append(all_peaks[j])
         j += 1

      # Choose template with strongest MF response
      best_peak = max(group, key=lambda x: filtered_outputs[x[1], x[0]])
      scores = np.array([filtered_outputs[t, best_peak[0]] for t in range(len(templates))])

      # Dynamic MF threshold based on template amplitude
      threshold_value = mf_dynamic[best_peak[1]]
      if np.max(scores) >= threshold_value:
         merged.append(best_peak[0])

      i = j

   spike_times = np.array(merged)

   # -----------------------------
   # 3) Extract spike windows & CNN features
   # -----------------------------
   spike_windows = []
   valid_spike_times = []

   for t in spike_times:
      if t - window_pre >= 0 and t + window_post < N:
         window = data[t - window_pre:t + window_post]
         spike_windows.append(window)
         valid_spike_times.append(t)

   spike_times = np.array(valid_spike_times)

   # Align on peak
   spike_windows_aligned = []
   for window in spike_windows:
      peak_idx = np.argmax(np.abs(window))
      shift = window_pre - peak_idx
      aligned_window = np.roll(window, shift)
      spike_windows_aligned.append(aligned_window)

   spike_windows_aligned = np.array(spike_windows_aligned)

   # Normalise spikes
   spike_windows_normalised = spike_windows_aligned - np.mean(
      spike_windows_aligned, axis=1, keepdims=True
   )
   spike_windows_normalised = spike_windows_normalised / np.max(
      np.abs(spike_windows_normalised), axis=1, keepdims=True
   )

   print(f"[{ds_label}] Detected {len(spike_windows_normalised)} spikes after MF stage")

   # Wavelet second channel
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

   # -----------------------------
   # 4) CNN inference + confidence gating
   # -----------------------------
   model.eval()
   with torch.no_grad():
      logits = model(spike_windows_tensor)
      probs = F.softmax(logits, dim=1)
      conf, preds = torch.max(probs, dim=1)

   print(f"[{ds_label}] CNN raw predictions (1..5):",
         np.bincount(preds.cpu().numpy() + 1, minlength=6)[1:])

   # Per-class CNN thresholds for this noise profile
   cnn_thresholds_np = np.array([
      THRESHOLDS_CNN_CLASS["base"][ds_label][cls] +
      THRESHOLDS_CNN["scale"][ds_label] * amp_norm[cls - 1]
      for cls in range(1, 6)
   ])
   cnn_thresholds = torch.tensor(cnn_thresholds_np, dtype=torch.float32)

   mask = conf >= cnn_thresholds[preds]

   Index_vec = spike_times[mask.cpu().numpy()]
   Class_vec = (preds[mask].cpu().numpy() + 1)

   print(f"[{ds_label}] Spike counts per class (after gating):")
   for c in range(1, 6):
      count_c = int(np.sum(Class_vec == c))
      print(f"  Class {c}: {count_c} spikes")

   # -----------------------------
   # 5) Evaluation vs D1 GT (for analysis)
   # -----------------------------
   if ds_label == 'D1':
      GT_idx = GT_idx.ravel()
      GT_cls = GT_cls.ravel()

      PR_idx = np.asarray(Index_vec).ravel()
      PR_cls = np.asarray(Class_vec).ravel()

      tolerance = 50
      TP = 0
      FP = 0
      FN = 0

      TP_class = {c: 0 for c in range(1, 6)}
      FP_class = {c: 0 for c in range(1, 6)}
      FN_class = {c: 0 for c in range(1, 6)}

      used_gt = set()

      for p_idx, p_cls in zip(PR_idx, PR_cls):
         diffs = np.abs(GT_idx - p_idx)
         min_diff = np.min(diffs)
         min_loc = np.argmin(diffs)

         if min_diff > tolerance:
            FP += 1
            continue

         if min_loc in used_gt:
            FP += 1
            continue

         if p_cls == GT_cls[min_loc]:
            TP += 1
            TP_class[int(p_cls)] += 1
            used_gt.add(min_loc)
         else:
            FP += 1
            FP_class[int(p_cls)] += 1

      for i, cls in enumerate(GT_cls):
         if i not in used_gt:
            FN += 1
            FN_class[int(cls)] += 1

      det_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
      det_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

      print(f"\n[{ds_label}] Detection Performance (vs D1 GT):")
      print("  TP:", TP)
      print("  FP:", FP)
      print("  FN:", FN)
      print("  Detection Recall:", det_recall)
      print("  Detection Precision:", det_precision)

      print(f"\n[{ds_label}] Classification Performance per Class:")
      for c in range(1, 6):
         tp_c = TP_class[c]
         fp_c = FP_class[c]
         fn_c = FN_class[c]
         recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
         precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
         print(
            f"  Class {c}: TP={tp_c}, FP={fp_c}, FN={fn_c}, "
            f"Recall={recall_c:.4f}, Precision={precision_c:.4f}"
         )

      metrics = {
         "TP": TP,
         "FP": FP,
         "FN": FN,
         "det_recall": det_recall,
         "det_precision": det_precision,
         "TP_class": TP_class,
         "FP_class": FP_class,
         "FN_class": FN_class,
      }

      return Index_vec, Class_vec, metrics
   
   else:
      return Index_vec, Class_vec

datasets = ('D1', 'D2', 'D3', 'D4', 'D5', 'D6')

# Shared parameters
Fs = 25000  # Sampling frequency (Hz)
window_pre = 100
window_post = 100
window_size = window_pre + window_post

# --------------------------------------------------------------
# 0) Load clean D1 + ground truth for training/templates
# --------------------------------------------------------------
training_dataset = spio.loadmat(
    'Coursework/Coursework C/Coursework_C_Datasets/D1.mat',
    squeeze_me=True
)
data_clean = training_dataset['d']
GT_idx = np.asarray(training_dataset['Index']).ravel()
GT_cls = np.asarray(training_dataset['Class']).ravel()

# Sort GT spikes by index
order = np.argsort(GT_idx)
GT_idx = GT_idx[order]
GT_cls = GT_cls[order]

# --------------------------------------------------------------
# 1) Build training set on *clean* D1 and train shared CNN
# --------------------------------------------------------------
train_x, train_y = build_d1_training_set(
    data_clean, GT_idx, GT_cls,
    Fs=Fs, window_pre=window_pre, window_post=window_post
)

print(f"Training set: {train_x.shape[0]} spikes, "
      f"{train_x.shape[2]} samples per channel")

model = train_cnn_on_d1(
    train_x, train_y,
    window_size=window_size,
    num_epochs=100
)

# --------------------------------------------------------------
# 2) Build templates and amp_norm from clean D1
# --------------------------------------------------------------
templates, amp_norm = build_templates_from_clean(
    data_clean, GT_idx, GT_cls,
    window_pre=window_pre, window_post=window_post
)

# --------------------------------------------------------------
# 3) Process each dataset D1..D6 and save results
# --------------------------------------------------------------
results_dir = 'Coursework/Coursework C/Coursework_C_Results'
os.makedirs(results_dir, exist_ok=True)

for i, ds in enumerate(datasets, start=1):
   print("\n" + "=" * 70)
   print(f"Processing dataset {ds} ({i}/{len(datasets)})")
   print("=" * 70)

   data_mat = spio.loadmat(
      f'Coursework/Coursework C/Coursework_C_Datasets/{ds}.mat',
      squeeze_me=True
   )
   data = data_mat['d']

   if ds == 'D1':
      Index_vec, Class_vec, metrics = run_full_pipeline_on_data(
         data,
         ds_label=ds,
         model=model,
         GT_idx=GT_idx,
         GT_cls=GT_cls,
         templates=templates,
         amp_norm=amp_norm,
         Fs=Fs,
         window_pre=window_pre,
         window_post=window_post
      )
   else:
      Index_vec, Class_vec = run_full_pipeline_on_data(
         data,
         ds_label=ds,
         model=model,
         GT_idx=GT_idx,
         GT_cls=GT_cls,
         templates=templates,
         amp_norm=amp_norm,
         Fs=Fs,
         window_pre=window_pre,
         window_post=window_post
      )

   # Convert to numpy arrays and ensure integer dtypes
   class_np = np.asarray(Class_vec)
   index_np = np.asarray(Index_vec)
   try:
      class_np = class_np.astype(np.int32)
   except Exception:
      pass
   try:
      index_np = index_np.astype(np.int64)
   except Exception:
      pass

   # Print how many spikes per class are being saved
   print(f"Spike counts per class for dataset {ds}:")
   for c in range(1, 6):
      count_c = int(np.sum(class_np == c))
      print(f"  Class {c}: {count_c} spikes")

   # Save .mat for this dataset
   save_path = os.path.join(results_dir, f'{ds}.mat')
   spio.savemat(save_path, {'Index': index_np, 'Class': class_np})
   print(f"Saved predictions to: {save_path}")