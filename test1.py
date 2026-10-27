import librosa #โหลดไฟล์เสียง
import soundfile as sf #
import matplotlib.pyplot as plt #วาดกราฟ
import noisereduce as nr
import numpy as np

y, sr = librosa.load("test1.wav", sr=None)
# plt.subplot(2,1,1)
# plt.plot(y)
# plt.title("Original Audio")

# ประมวลผลตัดเสียงรบกวน
reduce_noise = nr.reduce_noise(y=y,sr=sr,n_std_thresh_stationary=1.5,stationary=True)*2
D_original = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
D_reduced = librosa.amplitude_to_db(np.abs(librosa.stft(reduce_noise)), ref=np.max)

plt.figure(figsize=(12, 8))

# Top: Original Spectrogram
plt.subplot(2, 1, 1)
librosa.display.specshow(D_original, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("Original Audio Spectrogram")

# Bottom: Reduced Noise Spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(D_reduced, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("Reduce_noise Audio Spectrogram")

plt.tight_layout()
plt.show()
# sf.write("reduce Audio",reduce_noise,sr)