import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a

def apply_bandpass_filter(data, lowcut=300, highcut=5000, fs=25000, order = 2):
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   y = filtfilt(b, a, data)
   return y

mat = spio.loadmat('Coursework/Coursework C/Coursework_C_Datasets/D1.mat', squeeze_me=True)

data = mat['d']

indexes = mat['Index']

classes = mat['Class']

Fs = 25000  # Sampling frequency (Hz)
Ts = 1 / Fs  # Sampling period (s)

filtered_data = apply_bandpass_filter(data, lowcut=1, highcut=3000, fs=Fs, order=2)

N = len(filtered_data)  # Number of samples
t = np.arange(N) * Ts  # Time axis in seconds

spikes = []

for i in indexes:
   # Peak detection to find spikes

   pass