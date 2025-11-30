import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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

mat = spio.loadmat('Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)

data = mat['d']
indexes = mat['Index']
classes = mat['Class']

Fs = 25000  # Sampling frequency (Hz)
Ts = 1 / Fs  # Sampling period (s)
N = len(data)  # Number of samples
t = np.arange(N) * Ts  # Time axis in seconds

filtered_data = apply_bandpass_filter(data, lowcut=300, highcut=5000, fs=Fs, order=2)

# Sort spikes with their class into a list
classes = classes[np.argsort(indexes)]
indexes = np.sort(indexes)

print("Number of detected spikes:", len(indexes))
print("Number of classes of detected spikes:", len(classes))

spikes = []
spike_length = int(0.006 / Ts)  # 4 ms spike length in samples

for i, spike_class in zip(indexes, classes):
   # Peak detection to find spikes
   spikes.append([spike_class,filtered_data[i-(spike_length//3):i+(2*spike_length//3)]])

spike = 9

# Show 5 spakes from each class
plt.figure(figsize=(10, 5))
for c in np.unique(classes):
   plt.subplot(len(np.unique(classes)), 1, c)
   class_spikes = [s[1] for s in spikes if s[0] == c]
   for s in class_spikes[:5]:
       plt.plot(np.arange(len(s)) * Ts * 1000, s)  # Time in ms
   plt.title("Class {}".format(c))
   plt.xlabel("Time (ms)")
   plt.ylabel("Amplitude")
   plt.grid(True)

# Plot data against time
plt.figure(figsize=(10, 5))
plt.plot(t, filtered_data)  # Plot the first detected spike as an example
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Spike Waveform (Class {})".format(spikes[spike][0]))
plt.grid(True)

# Prepare signal for FFT: remove DC and apply a Hann window
N = len(filtered_data)
data_d = filtered_data - np.mean(filtered_data)
window = np.hanning(N)
data_win = data_d * window

# Compute real FFT and frequency axis
Y = np.fft.rfft(data_win)
freqs = np.fft.rfftfreq(N, d=1.0/Fs)

# Correct amplitude for windowing and FFT length
scale = 1.0 / (np.sum(window) / N)
amp = np.abs(Y) * scale / N

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