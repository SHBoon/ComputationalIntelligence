import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, fftconvolve, find_peaks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SpikeCNN(nn.Module):
   """CNN with two heads:
   - classification head: outputs logits for 5 classes (CrossEntropyLoss)
   - overlap head: outputs single logit for whether the snippet contains a secondary spike (BCEWithLogitsLoss)
   """
   def __init__(self, input_length=100, num_classes=5):
      super().__init__()

      self.features = nn.Sequential(
         nn.Conv1d(1, 16, kernel_size=5, padding=2),
         nn.ReLU(),
         nn.MaxPool1d(2),

         nn.Conv1d(16, 32, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.MaxPool1d(2)
      )

      with torch.no_grad():
         dummy = torch.zeros(1, 1, input_length)
         flat_size = self.features(dummy).numel()

      # classification head
      self.classifier = nn.Sequential(
         nn.Flatten(),
         nn.Linear(flat_size, 64),
         nn.ReLU(),
         nn.Linear(64, num_classes)
      )

      # overlap detection head (binary)
      self.overlap_head = nn.Sequential(
         nn.Flatten(),
         nn.Linear(flat_size, 32),
         nn.ReLU(),
         nn.Linear(32, 1)
      )

   def forward(self, x):
      feats = self.features(x)
      class_logits = self.classifier(feats)
      overlap_logit = self.overlap_head(feats).squeeze(-1)  # (N,) logits
      return class_logits, overlap_logit

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

# Template prep
def prepare_template(template):
   template = np.asarray(template, dtype=float)
   template = template - np.mean(template)
   norm = np.linalg.norm(template)
   if norm == 0:
      return template
   return template / norm

# Matched filter responses
def matched_filter_responses(data, templates):
   responses = []
   for h in templates:
      resp = fftconvolve(data, h[::-1], mode='same')
      responses.append(resp)
   return responses

# Peak detection in matched filter responses
def detect_peaks_resp(resp, Fs, threshold_mul=5.0, min_dist_ms=0.5):
   med = np.median(resp)
   mad = np.median(np.abs(resp - med))
   thr = med + threshold_mul * mad
   min_dist = int(np.round(min_dist_ms * Fs / 1000.0))
   peaks, props = find_peaks(resp, height=thr, distance=min_dist)
   return peaks, thr

# Template matching and spike assignment
def assign_detections(responses, det_times_per_template, templates, signal_data, pre_samples, post_samples):
   candidate_list = []
   for template_index, peak_locations in enumerate(det_times_per_template):
      response = responses[template_index]
      for peak_sample in peak_locations:
         score_at_peak = response[peak_sample]
         candidate_list.append((peak_sample, template_index, score_at_peak))

   candidate_list.sort(key=lambda x: x[2], reverse=True)

   used_region = np.zeros(len(signal_data), dtype=bool)
   detected_spikes = []

   for peak_sample, template_index, peak_score in candidate_list:
      # --- confidence filter: discard detections with low matched-filter score ---
      SCORE_THRESH = 0.25  # adjust as needed
      if peak_score < SCORE_THRESH:
          continue
      start_sample = peak_sample - pre_samples
      end_sample = peak_sample + post_samples

      if start_sample < 0 or end_sample >= len(signal_data):
         continue

      if used_region[start_sample:end_sample].any():
         continue

      spike_snippet = signal_data[start_sample:end_sample]

      template_waveform = templates[template_index]
      if len(template_waveform) != len(spike_snippet):
         window_length = len(spike_snippet)
         if len(template_waveform) > window_length:
            crop_start = (len(template_waveform) - window_length) // 2
            template_adjusted = template_waveform[crop_start:crop_start + window_length]
         else:
            pad_amount = window_length - len(template_waveform)
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            template_adjusted = np.pad(template_waveform, (pad_left, pad_right))
      else:
         template_adjusted = template_waveform

      spike_amplitude = np.dot(spike_snippet - np.mean(spike_snippet), template_adjusted)

      detected_spikes.append({
         'time': peak_sample,
         'template': template_index,
         'amp': spike_amplitude,
         'snippet': spike_snippet
      })

      used_region[start_sample:end_sample] = True

   return detected_spikes

mat = spio.loadmat('Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)

data = mat['d']
indexes = mat['Index']
classes = mat['Class']

# Sort spikes with their class
classes = classes[np.argsort(indexes)]
indexes = np.sort(indexes)

Fs = 25000  # Sampling frequency (Hz)
Ts = 1 / Fs  # Sampling period (s)
N = len(data)  # Number of samples

# Apply bandpass filter to the data
filtered_data = apply_bandpass_filter(data, lowcut=2, highcut=10000, fs=Fs, order=2)

spikes = []
spike_length = int(0.006 / Ts)  # 150 samples at 25 kHz
pre = spike_length//2
post = spike_length - pre

for i, spike_class in zip(indexes, classes):
   # Peak detection to find spikes
   spikes.append([spike_class,filtered_data[i - pre: i + post]])

# Sort spikes into separate lists per class
spike1 = []
spike2 = []
spike3 = []
spike4 = []
spike5 = []

for s in spikes:
   if s[0] == 1:
      spike1.append(s[1])
   elif s[0] == 2:
      spike2.append(s[1])
   elif s[0] == 3:
      spike3.append(s[1])
   elif s[0] == 4:
      spike4.append(s[1])
   elif s[0] == 5:
      spike5.append(s[1])

# Convert lists to proper 2D numpy arrays
spike1 = np.array(spike1)
spike2 = np.array(spike2)
spike3 = np.array(spike3)
spike4 = np.array(spike4)
spike5 = np.array(spike5)

template1 = np.mean(spike1, axis=0)
template2 = np.mean(spike2, axis=0)
template3 = np.mean(spike3, axis=0)
template4 = np.mean(spike4, axis=0)
template5 = np.mean(spike5, axis=0)

templates_raw = [template1, template2, template3, template4, template5]

# Load spike templates
# templates_raw = np.load('Coursework/Coursework C/templates.npy', allow_pickle=True)

templates = [prepare_template(t) for t in templates_raw]

# Compute matched filter responses
responses = matched_filter_responses(filtered_data, templates)

# Detect peaks in each matched filter response (use a lower threshold to be sensitive)
det_times_per_template = []
thresholds = []
# use a lower threshold to capture more candidates (we'll filter later with CNN)
for resp in responses:
   peaks, thr = detect_peaks_resp(resp, Fs, threshold_mul=1.2, min_dist_ms=0.06)
   det_times_per_template.append(peaks)
   thresholds.append(thr)

# Initial template matching (matched filter) to get candidate detections
detections = assign_detections(responses, det_times_per_template, templates, filtered_data, pre, post)
print(f"Initial matched-filter detections: {len(detections)} (sensitive threshold)")

# --- Build dataset for CNN training (using ground truth spikes only) ---

class TrueSpikeDataset(Dataset):
    def __init__(self, filtered_data, indexes, classes, pre, post, overlap_window=100):
        self.X = []
        self.y = []
        self.ov = []

        for t, c in zip(indexes, classes):
            t = int(t)
            start = t - pre
            end = t + post
            if start < 0 or end >= len(filtered_data):
                continue

            snippet = filtered_data[start:end].copy()
            snippet = snippet - np.mean(snippet)
            snippet = snippet / (np.std(snippet) + 1e-8)

            label = int(c) - 1  # convert classes 1–5 → 0–4

            overlap = any(abs(int(t2) - t) <= overlap_window and int(t2) != t for t2 in indexes)

            self.X.append(snippet.astype(np.float32))
            self.y.append(label)
            self.ov.append(1.0 if overlap else 0.0)

        self.X = torch.tensor(self.X).unsqueeze(1)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.ov = torch.tensor(self.ov, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.ov[i]

# Instantiate training dataset from ground truth spikes
true_dataset = TrueSpikeDataset(filtered_data, indexes, classes, pre, post, overlap_window=150)
true_loader = DataLoader(true_dataset, batch_size=64, shuffle=True)
print(f"Training CNN on {len(true_dataset)} true spikes.")

# Train CNN on true spikes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpikeCNN(input_length=spike_length, num_classes=5).to(device)
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
print('Training CNN classifier on ground truth spikes...')

for epoch in range(num_epochs):
    model.train()
    total = 0.0
    for xb, yb, ovb in true_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        ovb = ovb.to(device)

        opt.zero_grad()
        logits, ologits = model(xb)
        loss = ce_loss(logits, yb) + 0.3 * bce_loss(ologits, ovb)
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss {total:.4f}")

print('CNN training finished (ground truth).')

# Helper: classify a snippet and return (class, confidence, overlap_prob)
def classify_snippet(snippet):
   sn = snippet.copy()
   sn = (sn - np.mean(sn)) / (np.std(sn) + 1e-8)
   x = torch.tensor(sn, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
   model.eval()
   with torch.no_grad():
      logits, ologits = model(x)
      probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
      cls = int(np.argmax(probs))
      conf = float(np.max(probs))
      ovp = float(torch.sigmoid(ologits).cpu().numpy().squeeze())
   return cls, conf, ovp

# Apply CNN to all MF detections and keep confidence info
for d in detections:
   t = int(d['time'])
   start = t - pre
   end = t + post
   if start < 0 or end >= len(filtered_data):
      d['cnn_class'] = None
      d['cnn_conf'] = 0.0
      d['cnn_ovp'] = 1.0
      continue
   snippet = filtered_data[start:end]
   cls, conf, ovp = classify_snippet(snippet)
   d['cnn_class'] = cls
   d['cnn_conf'] = conf
   d['cnn_ovp'] = ovp

# Filter detections by CNN confidence and overlap probability
CONF_THRESH = 0.65
OVERLAP_REJECT = 0.9
kept = []
discarded = 0
for d in detections:
   if d['cnn_conf'] >= CONF_THRESH and d['cnn_ovp'] <= OVERLAP_REJECT:
      kept.append(d)
   else:
      discarded += 1
print(f"Detections kept after CNN filtering: {len(kept)} (discarded {discarded})")
# Use kept detections for further unmixing
detections = kept

# --- Matching-pursuit unmixing for detections flagged as overlapping or low-confidence ---
# re-use matching_pursuit_unmix defined earlier if present

   # define a simple version (same as previously inserted)
def shift_and_crop_template(template, shift, length):
   tpl = template.copy()
   if len(tpl) != length:
      if len(tpl) > length:
         crop_start = (len(tpl) - length) // 2
         tpl = tpl[crop_start:crop_start+length]
      else:
         pad = length - len(tpl)
         left = pad // 2
         right = pad - left
         tpl = np.pad(tpl, (left, right))
   if shift == 0:
      return tpl
   shifted = np.roll(tpl, shift)
   if shift > 0:
      shifted[:shift] = 0.0
   else:
      shifted[shift:] = 0.0
   return shifted

def match_templates(snippet, templates, shift_range=12):
   best = {'tpl_idx': None, 'shift': 0, 'amp': 0.0, 'score': -np.inf}
   L = len(snippet)
   for i, tpl in enumerate(templates):
      for s in range(-shift_range, shift_range+1):
         t_shifted = shift_and_crop_template(tpl, s, L)
         amp = np.dot(snippet, t_shifted)
         if amp > best['score']:
            best = {'tpl_idx': i, 'shift': s, 'amp': amp, 'score': amp, 'tpl_shifted': t_shifted}
   return best

def matching_pursuit_unmix(snippet, templates, max_components=2, shift_range=12, amp_ratio_thresh=0.2):
   comps = []
   residual = snippet.copy()
   best1 = match_templates(residual, templates, shift_range=shift_range)
   if best1['tpl_idx'] is None:
      return comps
   amp1 = best1['amp']
   tpl1 = best1['tpl_shifted']
   comp1 = amp1 * tpl1
   residual = residual - comp1
   comps.append({'tpl_idx': best1['tpl_idx'], 'shift': best1['shift'], 'amp': amp1, 'waveform': comp1})
   best2 = match_templates(residual, templates, shift_range=shift_range)
   if best2['tpl_idx'] is None:
      return comps
   amp2 = best2['amp']
   if abs(amp2) >= amp_ratio_thresh * max(1e-12, abs(amp1)):
      tpl2 = best2['tpl_shifted']
      comp2 = amp2 * tpl2
      residual2 = residual - comp2
      if np.sum(residual**2) - np.sum(residual2**2) > 1e-6:
         comps.append({'tpl_idx': best2['tpl_idx'], 'shift': best2['shift'], 'amp': amp2, 'waveform': comp2})
   return comps

# Perform unmixing on detections that have high overlap probability
final_spikes = []
for d in detections:
   snippet = d['snippet']
   if d['cnn_ovp'] > 0.1:
      comps = matching_pursuit_unmix(snippet, templates, max_components=2, shift_range=12, amp_ratio_thresh=0.2)
      if len(comps) == 0:
         # fallback: use CNN's single prediction
         final_spikes.append({'time': d['time'], 'class': d['cnn_class'], 'conf': d['cnn_conf']})
      elif len(comps) == 1:
         cls, conf, ovp = classify_snippet(comps[0]['waveform'])
         final_spikes.append({'time': d['time'] + comps[0]['shift'], 'class': cls, 'conf': conf})
      else:
         # two components: estimate times and classes
         for c in comps:
            cls, conf, ovp = classify_snippet(c['waveform'])
            est_time = int(d['time'] + c['shift'])
            final_spikes.append({'time': est_time, 'class': cls, 'conf': conf})
   else:
      # not overlapping: keep CNN result
      final_spikes.append({'time': d['time'], 'class': d['cnn_class'], 'conf': d['cnn_conf']})

print(f"Final spike count after CNN refinement & unmixing: {len(final_spikes)}")

# Final reconciliation: remove duplicates and sort
final_spikes_sorted = sorted(final_spikes, key=lambda x: x['time'])
# merge spikes of same class very close together (<3 samples)
merged = []
for s in final_spikes_sorted:
   if len(merged) == 0:
      merged.append(s)
      continue
   last = merged[-1]
   if s['class'] == last['class'] and abs(s['time'] - last['time']) <= 3:
      # keep the one with higher confidence
      if s['conf'] > last['conf']:
         merged[-1] = s
   else:
      merged.append(s)

print(f"Final spikes after merge: {len(merged)}")

# Replace detections with merged final spikes for downstream evaluation
# Build a new detections-like list for compatibility
detections = []
for s in merged:
   detections.append({'time': int(s['time']), 'template': s['class'], 'amp': None, 'snippet': filtered_data[int(s['time'])-pre:int(s['time'])+post] if int(s['time'])-pre>=0 and int(s['time'])+post < len(filtered_data) else np.zeros(spike_length)})

# Re-run evaluation (TP/FP/FN) using final detections
# Build list of (index_time, true_class)
gt_list = list(zip(indexes, classes))
TP = 0
FP = 0
FN = 0
tolerance = 50
det_used = [False] * len(detections)
gt_used = [False] * len(gt_list)
for i, (gt_time, gt_class) in enumerate(gt_list):
   best_err = float('inf')
   best_j = None
   for j, d in enumerate(detections):
      if det_used[j]:
         continue
      err = abs(d['time'] - gt_time)
      if err <= tolerance and err < best_err:
         best_err = err
         best_j = j
   if best_j is not None:
      gt_used[i] = True
      det_used[best_j] = True
      TP += 1
FP = det_used.count(False)
FN = gt_used.count(False)
print('Post-refinement TP:', TP)
print('Post-refinement FP:', FP)
print('Post-refinement FN:', FN)
print(f"Detection precision (final): {TP / (TP + FP + 1e-9):.3f}")
print(f"Detection recall (final): {TP / (TP + FN + 1e-9):.3f}")

# End of pipeline