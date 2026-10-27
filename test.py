import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 1. โหลดไฟล์เสียง
file_path = "test1.wav"
y, sr = librosa.load(file_path, sr=None)

# ตั้งค่าพารามิเตอร์สำหรับการแปลงสัญญาณ (เหมือนค่าเริ่มต้นของ noisereduce)
n_fft = 2048
hop_length = 512
win_length = 2048

# 2. แปลงสัญญาณเสียงเป็น Spectrogram (STFT)
sig_stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
# แปลงขนาด (Magnitude) เป็นหน่วยเดซิเบล (dB) สำหรับคำนวณเกณฑ์เสียง
sig_stft_db = librosa.amplitude_to_db(np.abs(sig_stft), ref=1.0)
max_db=np.max(sig_stft_db)

# 3. คำนวณ Noise Threshold 
# หาค่าเฉลี่ย (Mean) และส่วนเบี่ยงเบนมาตรฐาน (Std) ของแต่ละย่านความถี่จากสัญญาณ
mean_freq_noise = np.mean(sig_stft_db, axis=1)
std_freq_noise = np.std(sig_stft_db, axis=1)

# กำหนดระดับความเข้มในการตัด (เทียบเท่า n_std_thresh_stationary=1.5)
n_std_thresh_stationary = 1.5
noise_thresh = mean_freq_noise + (std_freq_noise * n_std_thresh_stationary)

# ขยายขนาด Threshold ให้เท่ากับความยาวของ Spectrogram (เพื่อให้เปรียบเทียบได้ทุกช่วงเวลา)
db_thresh = np.repeat(np.reshape(noise_thresh, (-1, 1)), sig_stft_db.shape[1], axis=1)

# 4. สร้าง Spectral Mask
prop_decrease = 1.0  # สัดส่วนการลดเสียง (1.0 = ตัดเสียงรบกวนออก 100%)
sig_mask = (sig_stft_db > db_thresh).astype(float)
# ปรับ Mask เผื่อกรณีไม่ได้ต้องการตัดเสียงรบกวนออกทั้งหมด
sig_mask = (sig_mask * prop_decrease) + (1.0 - prop_decrease)

# 5. คูณ Mask กลับเข้าไปที่ STFT ต้นฉบับ (Denoised Spectrogram)
sig_stft_denoised = sig_stft * sig_mask

# 6. แปลงสัญญาณกลับเป็นเสียง (Inverse STFT)
y_denoised = librosa.istft(sig_stft_denoised, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
# คุณสามารถนำ y_denoised ไปเซฟเป็นไฟล์เสียงใหม่ได้ด้วย soundfile.write()

# ==========================================
# ส่วนของการแสดงผลกราฟแบบ 3 แถว 1 คอลัมน์
# ==========================================
plt.figure(figsize=(6, 10))

plt.subplot(4, 1, 1)
plt.plot(y)
plt.title("Original Sound Wave")
# (1) Spectrogram ต้นฉบับ
plt.subplot(4, 1, 2)
librosa.display.specshow(sig_stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", cmap="afmhot",vmin=max_db - 60,vmax=max_db)
# plt.colorbar(format="%+2.0f dB")
plt.ylim(0, 10000)
plt.title("Original Spectrogram")

# (2) Spectral Mask (สีขาวดำ)
plt.subplot(4, 1, 3)
librosa.display.specshow(sig_mask, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", cmap="gray")
plt.ylim(0, 10000)
plt.title("Spectral Mask")

# (3) Spectrogram ที่ลดเสียงรบกวนแล้ว
sig_stft_denoised_db = librosa.amplitude_to_db(np.abs(sig_stft_denoised), ref=1.0)
plt.subplot(4, 1, 4)
librosa.display.specshow(sig_stft_denoised_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", cmap="afmhot",vmin=max_db - 60,vmax=max_db)
# plt.colorbar(format="%+2.0f dB")
plt.title("Denoised Spectrogram (Masked)")
plt.ylim(0, 10000)
plt.tight_layout()
plt.show()