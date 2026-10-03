import numpy as np #สมการกราฟ
import librosa #โหลดไฟล์เสียง
import soundfile as sf #
import scipy.fftpack as fft #ฟังก์ชันแปลงสัญญาณ
from scipy import signal  #สัญญาณเสียง
import matplotlib.pyplot as plt #วาดกราฟ

def single_mic_noise_reduction(audio_signal, sample_rate, noise_estimation_time=0.5):
    """
    ลดเสียงรบกวนสำหรับไมโครโฟนตัวเดียวด้วยวิธี Spectral Subtraction
    """
    # 1. แปลงสัญญาณจาก Time Domain เป็น Frequency Domain (STFT)
    f, t, Zxx = signal.stft(audio_signal, fs=sample_rate, nperseg=1024)
    
    # 2. ประมาณค่าสเปกตรัมของเสียงรบกวนจากช่วงแรก ( noise_estimation_time )
    noise_frames = int(noise_estimation_time * sample_rate / (1024 / 2))
    noise_spectrum = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1, keepdims=True)
    
    # 3. แยก Magnitude และ Phase ออกจากกัน
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # 4. หักล้างสเปกตรัมเสียงรบกวนออกจากสัญญาณหลัก (ตั้งค่าไม่ให้ติดลบ)
    # ตัวคูณ (e.g. 1.5) คือ Over-subtraction factor ช่วยให้ตัดเสียงรบกวนได้เกลี้ยงขึ้น
    cleaned_magnitude = np.maximum(magnitude - 1.5 * noise_spectrum, 0)
    
    # 5. รวม Magnitude ใหม่เข้ากับ Phase เดิม แล้วแปลงกลับเป็น Time Domain (ISTFT)
    Zxx_cleaned = cleaned_magnitude * np.exp(1j * phase)
    _, cleaned_signal = signal.istft(Zxx_cleaned, fs=sample_rate)
    
    return cleaned_signal

#load file
y, sr = librosa.load("test1.wav", sr=None)

# ประมวลผลตัดเสียงรบกวน
cleaned_audio = single_mic_noise_reduction(y, sr)

sf.write('clean.wav', cleaned_audio, sr)

plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, len(y)/sr, num=len(y)), y, label="Original Single Mic (Voice + Noise)", alpha=0.6)
plt.plot(np.linspace(0, len(cleaned_audio)/sr, num=len(cleaned_audio)), cleaned_audio, label="Cleaned Signal (Spectral Subtraction)", color='green')
plt.title("Single Microphone Noise Reduction")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.show()