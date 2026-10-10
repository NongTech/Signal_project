import IPython.display as ipd
import librosa
from scipy import signal
from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import librosa
import io

#load file
y, sr = librosa.load("test1.wav", sr=None)

plt.figure(figsize=(20, 5))
plt.subplot(2,2,1)
plt.plot(y)
plt.title("Original Audio")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0, stationary=True)
plt.subplot(2,2,2)
plt.plot(reduced_noise)
plt.title("Reduced Noise Audio")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

reduced_noise2 = nr.reduce_noise(y = y, sr=sr, thresh_n_mult_nonstationary=2,stationary=False)
plt.subplot(2,2,3)
plt.plot(reduced_noise2)
plt.title("Reduced Noise Audio2")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

#sf.write("reduced_noise.wav", reduced_noise, sr)
#sf.write("reduced_noise2.wav", reduced_noise2, sr)