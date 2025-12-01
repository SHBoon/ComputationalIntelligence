import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, fftconvolve, find_peaks

# Filtering functions
def butter_bandpass(lowcut, highcut, fs, order):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a

def apply_bandpass_filter(data, lowcut=1, highcut=300, fs=25000, order = 2):
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   y = filtfilt(b, a, data)
   return y

# Template preparation
def prepare_template(w):
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



mat = spio.loadmat('Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)

data = mat['d']
indexes = mat['Index']
classes = mat['Class']

Fs = 25000  # Sampling frequency (Hz)
Ts = 1 / Fs  # Sampling period (s)
N = len(data)  # Number of samples
t = np.arange(N) * Ts  # Time axis in seconds

# Apply bandpass filter to the data
filtered_data = apply_bandpass_filter(data, lowcut=30, highcut=4000, fs=Fs, order=2)

# Sort spikes with their class into a list
classes = classes[np.argsort(indexes)]
indexes = np.sort(indexes)

print("Number of spikes:", len(indexes))
print("Number of classes of spikes:", len(classes))

spikes = []
spike_length = int(0.002 / Ts)  # 2 ms spike length in samples

for i, spike_class in zip(indexes, classes):
   # Peak detection to find spikes
   spikes.append([spike_class,filtered_data[i-(spike_length//3):i+(2*spike_length//3)]])

# Show 5 spikes from each class
plt.figure(figsize=(10, 10))
for c in np.unique(classes):
   plt.subplot(len(np.unique(classes)), 1, c)
   class_spikes = [s[1] for s in spikes if s[0] == c]
   for s in class_spikes[:50]:
      plt.plot(np.arange(len(s)) * Ts * 1000, s)  # Time in ms
   plt.title("Class {}".format(c))
   plt.xlabel("Time (ms)")
   plt.ylabel("Amplitude")
   plt.grid(True)


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

templates = [prepare_template(t) for t in templates_raw]

# Save templates to file
np.save('Coursework/Coursework C/templates.npy', templates)

plt.figure(figsize=(10,6))
plt.plot(templates[0], label="Class 1")
plt.plot(templates[1], label="Class 2")
plt.plot(templates[2], label="Class 3")
plt.plot(templates[3], label="Class 4")
plt.plot(templates[4], label="Class 5")
plt.legend()
plt.title("Average Template Spikes")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


# # Plot data against time
# plt.figure(figsize=(10, 5))
# plt.plot(t, filtered_data)  # Plot the first detected spike as an example
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Spike Waveform (Class {})".format(spikes[spike][0]))
# plt.grid(True)

# # Prepare signal for FFT: remove DC and apply a Hann window
# N = len(filtered_data)
# data_d = filtered_data - np.mean(filtered_data)
# window = np.hanning(N)
# data_win = data_d * window

# # Compute real FFT and frequency axis
# Y = np.fft.rfft(data_win)
# freqs = np.fft.rfftfreq(N, d=1.0/Fs)

# # Correct amplitude for windowing and FFT length
# scale = 1.0 / (np.sum(window) / N)
# amp = np.abs(Y) * scale / N

# # Plot amplitude spectrum (linear)
# plt.figure(figsize=(10, 5))
# plt.plot(freqs, amp)
# plt.xlim(0, Fs/2)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
# plt.title("Amplitude Spectrum (FFT)")
# plt.grid(True)

# # Plot amplitude spectrum in dB
# eps = 1e-12
# plt.figure(figsize=(10, 5))
# plt.plot(freqs, 20 * np.log10(amp + eps))
# plt.xlim(0, Fs/2)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (dB)")
# plt.title("Amplitude Spectrum (dB)")
# plt.grid(True)

plt.show()