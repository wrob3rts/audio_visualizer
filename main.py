import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('audio.wav')

SamplingRate = 22050
#chirp = librosa.chirp(fmin = 32, fmax = 32 * 2**5, sr=SamplingRate, duration=10, linear=True)
D = np.abs(librosa.stft(y))
mag, phase = librosa.magphase(D)

freqs = librosa.fft_frequencies()
times = librosa.times_like(D)

phase_exp = 2*np.pi*np.multiply.outer(freqs,times)

fig, ax = plt.subplots()
img = librosa.display.specshow(np.diff(np.unwrap(np.angle(phase)-phase_exp, axis=1), axis=1, prepend=0),
                         cmap='hsv',
                         alpha=librosa.amplitude_to_db(mag, ref=np.max)/80 + 1,
                         ax=ax,
                         y_axis='log',
                         x_axis='time')
ax.set_facecolor('#000')
cbar = fig.colorbar(img, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.ax.set(yticklabels=['-π', '-π/2', "0", 'π/2', 'π']);
plt.show()