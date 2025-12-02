import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, fftconvolve, find_peaks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

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

def classify_detection(snippet):
   # Normalize snippet
   snippet = snippet.copy()
   snippet = snippet - np.mean(snippet)
   snippet = snippet / (np.std(snippet)+1e-8)

   # Convert to torch tensor
   x = torch.tensor(snippet, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

   with torch.no_grad():
      class_logits, overlap_logit = model(x)

      # Predicted class
      pred_class = torch.argmax(class_logits, dim=1).item()

      # Class confidence (softmax probability of predicted class)
      probs = torch.softmax(class_logits, dim=1)
      confidence = probs.max().item()

      # Overlap probability
      overlap_prob = torch.sigmoid(overlap_logit).item()

   return pred_class, overlap_prob, confidence

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

# Detect peaks in each matched filter response
det_times_per_template = []
thresholds = []

for resp in responses:
   peaks, thr = detect_peaks_resp(resp, Fs, threshold_mul=2.5, min_dist_ms=0.1)
   det_times_per_template.append(peaks)
   thresholds.append(thr)

# Template matching and spike assignment
detections = assign_detections(responses, det_times_per_template, templates, filtered_data, pre, post)

# Refine matched filtering results with a CNN
learning_rate = 0.01
num_epochs = 400
batch_size = 64

class SpikeDataset(Dataset):
   """Dataset returning (snippet, class_label, overlap_flag)
   overlap_flag is 1 if another detection/event exists within `overlap_window` samples of the center.
   When overlap augmentation is used, the dominant-spike-wins label is retained and overlap_flag is set to 1.
   """
   def __init__(self, data, spike_items, spike_classes=None, overlap_prob=0.2, overlap_window=150):
      self.X = []
      self.y = []
      self.ov = []

      self.data = data
      self.pre = pre
      self.post = post
      self.overlap_prob = overlap_prob
      self.overlap_window = overlap_window

      # helper to check nearby detections
      def has_nearby(item_time, items, window):
         for other in items:
            if other is item_time:
               continue
            t = int(other['time']) if isinstance(other, dict) else int(other)
            if abs(t - int(item_time)) <= window:
               return True
         return False

      # If spike_items is a list of detection dicts (built by assign_detections)
      if isinstance(spike_items, list) and len(spike_items) > 0 and isinstance(spike_items[0], dict):
         for det in spike_items:
            t = int(det['time'])
            # extract centered snippet at detection time
            start = t - self.pre
            end = t + self.post
            if start < 0 or end >= len(data):
               continue

            # base snippet
            snippet = data[start:end]
            snippet = (snippet - np.mean(snippet)) / (np.std(snippet) + 1e-8)

            label = int(det.get('template', 0))
            label = max(0, min(4, label))

            overlap_flag = 1 if has_nearby(t, spike_items, self.overlap_window) else 0

            # --- overlap augmentation (dominant spike wins) ---
            if np.random.rand() < overlap_prob:
                # choose a random second detection
                det2 = spike_items[np.random.randint(0, len(spike_items))]
                t2 = int(det2['time'])
                start2 = t2 - self.pre
                end2 = t2 + self.post
                if start2 >= 0 and end2 < len(self.data):
                    snippet2 = self.data[start2:end2]
                    snippet2 = (snippet2 - np.mean(snippet2)) / (np.std(snippet2) + 1e-8)

                    # random temporal misalignment
                    offset = np.random.randint(-10, 11)
                    combined = snippet.copy()

                    if offset >= 0:
                        combined[offset:] += snippet2[:len(combined) - offset]
                    else:
                        k = abs(offset)
                        combined[:len(combined) - k] += snippet2[k:]

                    # dominant template determines label
                    amp1 = np.max(np.abs(snippet))
                    amp2 = np.max(np.abs(snippet2))
                    if amp2 > amp1:
                        label = int(det2.get('template', 0))
                        label = max(0, min(4, label))

                    snippet = combined
                    overlap_flag = 1

            self.X.append(snippet.astype(np.float32))
            self.y.append(label)
            self.ov.append(overlap_flag)

      else:
         # fallback: spike_items is expected to be an array-like of spike times and spike_classes provided
         for i, t in enumerate(spike_items):
            start = int(t) - self.pre
            end   = int(t) + self.post

            if start < 0 or end >= len(data):
               continue

            snippet = data[start:end]
            snippet = (snippet - np.mean(snippet)) / (np.std(snippet)+1e-8)

            # single integer label from spike_classes (1..5) -> convert to 0..4
            lbl = 0
            if spike_classes is not None:
               lbl = int(spike_classes[i]) - 1
               lbl = max(0, min(4, lbl))

            # overlap label determined by presence of other indexes nearby
            overlap_flag = 0
            if spike_classes is not None:
               # check neighbouring indexes
               for j, tj in enumerate(spike_items):
                  if j == i: continue
                  if abs(int(tj) - int(t)) <= self.overlap_window:
                     overlap_flag = 1
                     break

            self.X.append(snippet.astype(np.float32))
            self.y.append(lbl)
            self.ov.append(overlap_flag)

      # Convert to tensors: X shaped (N,1,L), y shaped (N,) long, ov shaped (N,) float
      self.X = torch.tensor(self.X).unsqueeze(1)
      self.y = torch.tensor(self.y, dtype=torch.long)
      self.ov = torch.tensor(self.ov, dtype=torch.float32)

   def __len__(self):
      return len(self.X)

   def __getitem__(self, idx):
      return self.X[idx], self.y[idx], self.ov[idx]

# Train the CNN on snippets centered at matched-filter detections (aligned to MF peaks)
# use `detections` (list of dicts) created by assign_detections so train/infer domains match
dataset = SpikeDataset(filtered_data, detections, overlap_prob=0.2, overlap_window=150)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training data preparation
snippets = []
target_classes = []

for i, idx in enumerate(indexes):
   if idx - pre >= 0 and idx + post < len(filtered_data):
      snippet = filtered_data[idx-pre : idx+post]
      snippets.append(snippet)
      target_classes.append(classes[i])

model = SpikeCNN(input_length=spike_length, num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
   total_loss = 0
   for batch in train_loader:
      batch_x = batch[0].to(device)
      batch_y = batch[1].to(device)
      batch_ov = batch[2].to(device)

      optimiser.zero_grad()
      logits, ov_logits = model(batch_x)

      loss_class = criterion(logits, batch_y)
      loss_ov = bce_loss(ov_logits, batch_ov)

      # weight the overlap loss so it doesn't dominate
      loss = loss_class + 0.5 * loss_ov
      loss.backward()
      optimiser.step()

      total_loss += loss.item()

   print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

print("CNN training complete.")

# Add CNN predictions to detections
for d in detections:
   center = d['time']
   snippet = filtered_data[center-pre : center+post]
   if len(snippet) < spike_length:
      pad = spike_length - len(snippet)
      snippet = np.pad(snippet, (0, pad))
   cls, ovp, conf = classify_detection(snippet)
   if conf >= 0.6:
      d['cnn_class'] = cls
      d['cnn_overlap_prob'] = ovp

cnn_filtered_detections = []
for d in detections:
   enter = d['time']
   snippet = filtered_data[center-pre : center+post]
   if len(snippet) < spike_length:
      pad = spike_length - len(snippet)
      snippet = np.pad(snippet, (0, pad))
   cls, ovp, conf = classify_detection(d['snippet'])
   if conf >= 0.99:
      d['cnn_class'] = cls
      d['cnn_overlap_prob'] = ovp
      cnn_filtered_detections.append(d)

detections = cnn_filtered_detections

close_pairs = 0
pairs = []

tolerance = 100

for i in range(len(indexes)):
    for j in range(i+1, len(indexes)):  
        if indexes[j] - indexes[i] > tolerance:
            break  # no need to check further (sorted)
        close_pairs += 1
        pairs.append((indexes[i], indexes[j]))

print(f"Number of index pairs within {tolerance} samples: {close_pairs}")
print()

# Compare indexed spikes with detected spikes with 50 index tolerance with class labels
tolerance = 50
matched_spikes = []
unmatched_spikes = []
unmatched_snippets = []

for idx in indexes:
   for d in detections:
      if abs(d['time'] - idx) <= tolerance:
         if d['template'] + 1 == classes[np.where(indexes == idx)[0][0]]:
            matched_spikes.append((idx, d['template'] + 1))  # +1 to match class labels
         else:
            unmatched_spikes.append((idx, d['template'] + 1))

# Print number of detected spikes
print(f"Data contains {len(indexes)} indexed spikes.".format(len(indexes)))
print(f"Detected {len(detections)} spikes after template matching.")
print()

detected_index_spikes = 0

detected_index_spikes = 0
miss_errors = []

for idx in indexes:
   matched = False
   closest_error = float('inf')

   for d in detections:
      err = abs(d['time'] - idx)
      if err < closest_error:
         closest_error = err

      if err <= tolerance:
         matched = True
         detected_index_spikes += 1
         break

   if not matched:
      miss_errors.append(closest_error)

if len(miss_errors) > 0:
   avg_miss_error = np.mean(miss_errors)
else:
   avg_miss_error = 0

print(f"Matched filter detected indexed spikes (within {tolerance} samples): {detected_index_spikes}")
print(f"Recall: {detected_index_spikes/len(indexes):.3f}")
print(f"Average miss error (samples) for unmatched spikes: {avg_miss_error:.1f}")
print()

matched_spikes = []
unmatched_spikes = []
unmatched_snippets = []

for idx in indexes:
   for d in detections:
      if abs(d['time'] - idx) <= tolerance:
         if d['cnn_class'] + 1 == classes[np.where(indexes == idx)[0][0]]:
            matched_spikes.append((idx, d['cnn_class'] + 1))  # +1 to match class labels
         else:
            unmatched_spikes.append((idx, d['cnn_class'] + 1))

print(f"Number of CNN matched spikes (within {tolerance} samples): {len(matched_spikes)}")
print(f"Accuracy: {len(matched_spikes)/len(indexes)}")
print(f"Number of CNN unmatched spikes: {len(unmatched_spikes)}")
print()

# # Show unmatched spikes
# for idx, pred_class in unmatched_spikes:
#    # find corresponding detection snippet
#    for d in detections:
#       if abs(d['time'] - idx) <= tolerance:
#          unmatched_snippets.append(d['snippet'])
#          break

# for snippet in unmatched_snippets:
#     plt.plot(snippet, alpha=0.5)

# plt.title("Unmatched Spike Snippets")
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")
# plt.show()

# # Show signal with spike points detected
# plt.figure(figsize=(12,5))
# plt.plot(filtered_data, label='Filtered signal')
# colors = ['r','g','b','m','c']
# for d in detections:
#     plt.scatter(d['time'], filtered_data[d['time']], color=colors[d['template']], s=10)
# plt.title("Matched Filter Detections")
# plt.legend()
# plt.show()

# # Show templates found from indexes
# plt.figure(figsize=(10,6))
# for i, t in enumerate(templates_raw):
#     plt.plot(t, label=f'Template {i+1}')
# plt.title("Template Spikes (Averaged)")
# plt.legend()
# plt.show()

# # Show spike snippets assigned to each template
# for template_index in range(len(templates)):
#    plt.figure(figsize=(10,6))
#    assigned_snippets = [d['snippet'] for d in detections if d['template'] == template_index]
#    for snippet in assigned_snippets[:50]:
#       plt.plot(snippet, alpha=0.5)
#    plt.title(f"Snippets assigned to Template {template_index+1}")
#    plt.show()