# import numpy as np
# import os
# from scipy.io import wavfile
# from scipy.signal import lfilter, spectrogram

# # --- User settings ---
# input_path = "C:/Users/DJ/Downloads/FIR Filter/FIR Filter/Original.wav"          # replace with your audio file path
# output_path = "output_filtered.wav"
# frame_size = 2048
# hop_size = frame_size // 2
# eps = 1e-10

# if not os.path.exists(input_path):
#     raise FileNotFoundError(f"Input audio not found: {input_path}")

# # --- Load audio ---
# sr, audio = wavfile.read(input_path)
# if audio.dtype.kind == "i":  # integer PCM -> convert to float32 in [-1,1]
#     max_int = np.iinfo(audio.dtype).max
#     audio = audio.astype(np.float32) / max_int
# else:
#     audio = audio.astype(np.float32)

# # If stereo, convert to mono (average channels)
# if audio.ndim > 1:
#     audio_mono = audio.mean(axis=1)
# else:
#     audio_mono = audio

# # --- Compute spectral flatness per frame to detect noisy vs tonal (clear) frames ---
# def spectral_flatness(frame):
#     X = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
#     geom = np.exp(np.mean(np.log(X + eps)))
#     arith = np.mean(X + eps)
#     return geom / arith

# n_frames = 1 + (len(audio_mono) - frame_size) // hop_size
# if n_frames < 1:
#     n_frames = 1
#     pad = frame_size - len(audio_mono)
#     audio_mono = np.concatenate([audio_mono, np.zeros(pad)])

# flatness = np.zeros(n_frames)
# for i in range(n_frames):
#     start = i * hop_size
#     frame = audio_mono[start:start + frame_size]
#     flatness[i] = spectral_flatness(frame)

# # Threshold: below -> tonal/clear, above -> noisy
# flat_thresh = 0.35
# is_noisy = flatness > flat_thresh

# # --- Design two FIR filters: mild (for clear) and aggressive (for noisy) ---
# # Normalized cutoff (0.0..0.5)
# mild_cutoff = 0.20
# agg_cutoff = 0.08
# filt_len = 101  # fairly long for good attenuation

# from scipy.signal import firwin
# mild_fir = firwin(filt_len, cutoff=mild_cutoff, window='hann')
# agg_fir = firwin(filt_len, cutoff=agg_cutoff, window='hann')

# # --- Pre-filter whole signal with both filters (to avoid per-frame convolution cost) ---
# mild_full = lfilter(mild_fir, 1.0, audio_mono)
# agg_full = lfilter(agg_fir, 1.0, audio_mono)

# # --- Reconstruct output by selecting per-frame (with overlap-add) ---
# out = np.zeros(len(audio_mono) + filt_len)  # allow for filter delay/tails
# win = np.hanning(frame_size)
# norm = np.zeros_like(out)

# for i in range(n_frames):
#     start = i * hop_size
#     frame_m = mild_full[start:start + frame_size] * win
#     frame_a = agg_full[start:start + frame_size] * win
#     chosen = frame_a if is_noisy[i] else frame_m
#     out[start:start + frame_size] += chosen
#     norm[start:start + frame_size] += win

# # avoid division by zero
# nz = norm > 0
# out[nz] /= norm[nz]
# out = out[:len(audio_mono)]

# # --- Restore to original channels and dtype, save ---
# if audio.ndim > 1:
#     # duplicate mono back to stereo (simple approach)
#     out_final = np.vstack([out, out]).T
# else:
#     out_final = out

# # convert float [-1,1] to int16 for writing
# max_int16 = np.iinfo(np.int16).max
# wavfile.write(output_path, sr, (np.clip(out_final, -1, 1) * max_int16).astype(np.int16))

# # --- Optional: compute and show spectrograms for quick analysis (matplotlib used later in file) ---
# f_orig, t_orig, Sxx_orig = spectrogram(audio_mono, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_size)
# f_out, t_out, Sxx_out = spectrogram(out, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_size)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.pcolormesh(t_orig, f_orig, 10 * np.log10(Sxx_orig + eps), shading='gouraud')
# plt.title("Original Spectrogram")
# plt.ylabel("Frequency [Hz]")
# plt.colorbar(label="dB")

# plt.subplot(2, 1, 2)
# plt.pcolormesh(t_out, f_out, 10 * np.log10(Sxx_out + eps), shading='gouraud')
# plt.title("Filtered Spectrogram (frame-wise noisy -> aggressive, clear -> mild)")
# plt.xlabel("Time [s]")
# plt.ylabel("Frequency [Hz]")
# plt.colorbar(label="dB")
# plt.tight_layout()
# plt.show()

# print(f"Processed '{input_path}' -> '{output_path}'. Frames classified as noisy: {is_noisy.sum()}/{len(is_noisy)}")
#######################################################################
# import matplotlib.pyplot as plt
# from scipy.signal import firwin, freqz
# # Filter parameters
# cutoff_frequency = 0.2 # Normalized cutoff frequency (0.0 to 0.5)
# filter_length = 31 # Number of filter taps (odd for symmetry)
# # Design the FIR filter using Hanning window method
# filter_coefficients = firwin(filter_length, cutoff=cutoff_frequency, window='hann')
# # Plot the impulse response of the filter
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 1, 1)
# plt.stem(filter_coefficients)
# plt.title("Impulse Response")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")
# # Plot the frequency response of the filter
# plt.subplot(2, 1, 2)
# frequencies, response = freqz(filter_coefficients)
# plt.plot(frequencies, 20 * np.log10(np.abs(response)))
# plt.title("Frequency Response")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#############################
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write

from scipy.signal import butter, lfilter, firwin, freqz





def apply_IIR_filter(signal, cutoff, fs, order=6):
    b, a = butter(order, cutoff / (fs/2), btype="low")
    filtered = lfilter(b, a, signal)
    return filtered


def apply_FIR_filter(signal, cutoff, fs, numtaps=101):
    fir_coeff = firwin(numtaps, cutoff / (fs/2), window="hann")
    filtered = np.convolve(signal, fir_coeff, mode='same')
    return filtered


def correlation(clean, test):
    min_len = min(len(clean), len(test))
    return np.corrcoef(clean[:min_len], test[:min_len])[0,1]



clean_fs, clean = wavfile.read("C:/Users/DJ/Downloads/FIR Filter/FIR Filter/Original.wav")
noise1_fs, noise1 = wavfile.read("C:/Users/DJ/Downloads/FIR Filter/FIR Filter/sp30_car_sn5.wav")
noise2_fs, noise2 = wavfile.read("C:/Users/DJ/Downloads/FIR Filter/FIR Filter/sp30_train_sn5.wav")
noise3_fs, noise3 = wavfile.read("C:/Users/DJ/Downloads/FIR Filter/FIR Filter/sp30_station_sn5.wav")

# Convert to float
clean = clean.astype(float)
noise1 = noise1.astype(float)   
noise2 = noise2.astype(float)
noise3 = noise3.astype(float)


cutoff = 3000  # you can adjust based on spectrum

iir_noise1 = apply_IIR_filter(noise1, cutoff, noise1_fs)
iir_noise2 = apply_IIR_filter(noise2, cutoff, noise2_fs)
iir_noise3 = apply_IIR_filter(noise3, cutoff, noise3_fs)

corr_iir_1 = correlation(clean, iir_noise1)
corr_iir_2 = correlation(clean, iir_noise2)
corr_iir_3 = correlation(clean, iir_noise3)

print("\n===== IIR Filtering Correlation with Clean Speech =====")
print("Noise 1:", corr_iir_1)
print("Noise 2:", corr_iir_2)
print("Noise 3:", corr_iir_3)

fir_noise1 = apply_FIR_filter(noise1, cutoff, noise1_fs)
fir_noise2 = apply_FIR_filter(noise2, cutoff, noise2_fs)
fir_noise3 = apply_FIR_filter(noise3, cutoff, noise3_fs)

corr_fir_1 = correlation(clean, fir_noise1)
corr_fir_2 = correlation(clean, fir_noise2)
corr_fir_3 = correlation(clean, fir_noise3)

print("\n===== FIR Filtering Correlation with Clean Speech =====")
print("Noise 1:", corr_fir_1)
print("Noise 2:", corr_fir_2)
print("Noise 3:", corr_fir_3)


def normalize_audio(sig):
    sig = sig / np.max(np.abs(sig))
    return (sig * 32767).astype(np.int16)

# ---- IIR Output Files ----
write("iir_noise1.wav", noise1_fs, normalize_audio(iir_noise1))
write("iir_noise2.wav", noise2_fs, normalize_audio(iir_noise2))
write("iir_noise3.wav", noise3_fs, normalize_audio(iir_noise3))

# ---- FIR Output Files ----
write("fir_noise1.wav", noise1_fs, normalize_audio(fir_noise1))
write("fir_noise2.wav", noise2_fs, normalize_audio(fir_noise2))
write("fir_noise3.wav", noise3_fs, normalize_audio(fir_noise3))

print("\n✅ All filtered audio files saved successfully!")

# ===== Combined FFT Subplots =====

plt.figure(figsize=(12, 8))

signals = [clean, noise1, noise2, noise3]
titles  = ["Clean Speech FFT", "Noisy Speech 1 FFT", "Noisy Speech 2 FFT", "Noisy Speech 3 FFT"]

for i, sig in enumerate(signals):
    N = len(sig)
    freq = np.fft.rfftfreq(N, d=1/clean_fs)
    fft_mag = np.abs(np.fft.rfft(sig))

    plt.subplot(2, 2, i+1)
    plt.plot(freq, fft_mag)
    plt.title(titles[i])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

plt.tight_layout()
plt.show()
