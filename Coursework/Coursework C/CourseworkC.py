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
   def __init__(self, input_length=100):
      super().__init__()

      self.features = nn.Sequential(
         nn.Conv1d(1, 16, kernel_size=5, padding=2),
         nn.ReLU(),
         nn.MaxPool1d(2),        # → length / 2

         nn.Conv1d(16, 32, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.MaxPool1d(2)         # → length / 4
      )

      # compute flattened size automatically
      with torch.no_grad():
         dummy = torch.zeros(1, 1, input_length)
         flat_size = self.features(dummy).numel()

      self.classifier = nn.Sequential(
         nn.Flatten(),
         nn.Linear(flat_size, 64),
         nn.ReLU(),
         nn.Linear(64, 5)
      )

   def forward(self, x):
      x = self.features(x)
      return self.classifier(x)

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
def prepare_template(w):
   # w = apply_bandpass_filter(w, lowcut=1, highcut=300, fs=25000, order=2)
   w = np.asarray(w, dtype=float)
   w = w - np.mean(w)
   norm = np.linalg.norm(w)
   if norm == 0:
      return w
   return w / norm

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

# Sort spikes with their class into a list
classes = classes[np.argsort(indexes)]
indexes = np.sort(indexes)

Fs = 25000  # Sampling frequency (Hz)
Ts = 1 / Fs  # Sampling period (s)
N = len(data)  # Number of samples

# Apply bandpass filter to the data
filtered_data = apply_bandpass_filter(data, lowcut=2, highcut=3000, fs=Fs, order=2)

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
   peaks, thr = detect_peaks_resp(resp, Fs, threshold_mul=5, min_dist_ms=0.5)
   det_times_per_template.append(peaks)
   thresholds.append(thr)

# Template matching and spike assignment
detections = assign_detections(responses, det_times_per_template, templates, filtered_data, pre, post)

# Refine matched filtering results with a CNN
learning_rate = 0.2
num_epochs = 200
batch_size = 64

class SpikeDataset(Dataset):
   def __init__(self, data, spike_times, spike_classes):
      self.X = []
      self.y = []

      for i, t in enumerate(spike_times):
         start = t - pre
         end   = t + post

         if start < 0 or end >= len(data):
            continue

         snippet = data[start:end]
         snippet = snippet - np.mean(snippet)     # normalise
         snippet = snippet / (np.std(snippet)+1e-8)

         self.X.append(snippet.astype(np.float32))
         self.y.append(int(spike_classes[i]) - 1)  # convert to 0–4

      self.X = torch.tensor(self.X).unsqueeze(1)   # (N,1,50)
      self.y = torch.tensor(self.y)

   def __len__(self):
      return len(self.X)

   def __getitem__(self, idx):
      return self.X[idx], self.y[idx]

# Build dataset + loader
dataset = SpikeDataset(filtered_data, indexes, classes)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Training data preparation
snippets = []
target_classes = []

for i, idx in enumerate(indexes):
   if idx - pre >= 0 and idx + post < len(filtered_data):
      snippet = filtered_data[idx-pre : idx+post]
      snippets.append(snippet)
      target_classes.append(classes[i])

model = SpikeCNN(input_length=spike_length).to(device)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
   total_loss = 0
   for batch_x, batch_y in train_loader:
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      optimiser.zero_grad()
      logits = model(batch_x)
      loss = criterion(logits, batch_y)
      loss.backward()
      optimiser.step()

      total_loss += loss.item()

   print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

print("CNN training complete.")

def classify_detection(snippet):
   snippet = snippet - np.mean(snippet)
   snippet = snippet / (np.std(snippet)+1e-8)
   snippet = torch.tensor(snippet, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
   with torch.no_grad():
      logits = model(snippet)
      pred = torch.argmax(logits).item()
   return pred  # 0–4

# Add CNN predictions to detections
for d in detections:
   snippet = d['snippet']
   if len(snippet) < spike_length:
      pad = spike_length - len(snippet)
      snippet = np.pad(snippet, (0, pad))
   d['cnn_class'] = classify_detection(snippet)

# Compare indexed spikes with detected spikes with 50 index tolerance with class labels
tolerance = 100
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

# Print number of detected spikes
print(f"Data contains {len(indexes)} indexed spikes.".format(len(indexes)))
print(f"Detected {len(detections)} spikes after template matching.")

print(f"Number of matched spikes within {tolerance} samples tolerance: {len(matched_spikes)}")
print(f"Accuracy: {len(matched_spikes)/len(indexes)}")
print(f"Number of unmatched spikes: {len(unmatched_spikes)}")

# Show unmatched spikes
for idx, pred_class in unmatched_spikes:
   # find corresponding detection snippet
   for d in detections:
      if abs(d['time'] - idx) <= tolerance:
         unmatched_snippets.append(d['snippet'])
         break

for snippet in unmatched_snippets:
    plt.plot(snippet, alpha=0.5)

plt.title("Unmatched Spike Snippets")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# plt.figure(figsize=(12,5))
# plt.plot(filtered_data, label='Filtered signal')
# colors = ['r','g','b','m','c']
# for d in detections:
#     plt.scatter(d['time'], filtered_data[d['time']], color=colors[d['template']], s=10)
# plt.title("Matched Filter Detections")
# plt.legend()
# plt.show()

plt.figure(figsize=(10,6))
for i, t in enumerate(templates_raw):
    plt.plot(t, label=f'Template {i+1}')
plt.title("Template Spikes (Averaged)")
plt.legend()
plt.show()

# # show spike snippets assigned to each template
# for template_index in range(len(templates)):
#    plt.figure(figsize=(10,6))
#    assigned_snippets = [d['snippet'] for d in detections if d['template'] == template_index]
#    for snippet in assigned_snippets[:50]:
#       plt.plot(snippet, alpha=0.5)
#    plt.title(f"Snippets assigned to Template {template_index+1}")
#    plt.show()