import librosa
import librosa.display
import matplotlib.pyplot as plt

file = 'audio.wav'
y, sr = librosa.load(file)

plt.figure(figsize=(10,4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.show()

D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
plt.figure(figsize=(10,4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.title('Spectrogram')
plt.colrobar(format='%+2.0f dB')
plt.show()