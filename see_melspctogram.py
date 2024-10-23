import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Mel-Spectrogram을 로드하는 함수
def load_mel_spectrogram(file_path, sample_rate=16000, n_mels=128, n_fft=4096, hop_length=1024):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db, sr, n_fft, hop_length

# Mel-Spectrogram 시각화
def plot_mel_spectrogram(mel_spec_db, sr, hop_length):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram (dB)')
    plt.tight_layout()
    plt.show()

# Mel-Spectrogram을 다시 오디오로 복원하는 함수
def restore_audio_from_mel(mel_spec, sr, n_fft, hop_length, n_iter=100):
    # Griffin-Lim 알고리즘으로 Mel-spectrogram을 오디오 신호로 복원
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter)
    return audio

# Mel-Spectrogram 로드 및 시각화
mel_spec_db, sr, n_fft, hop_length = load_mel_spectrogram('1.wav')

print('mel_spc_db shape:', mel_spec_db.shape)
print('sr:', sr)
print('n_fft:', n_fft)
print('hop_length:', hop_length)

mel_spec = librosa.db_to_power(mel_spec_db)
print('mel_spec shape:', mel_spec.shape)

# Mel-Spectrogram 시각화
plot_mel_spectrogram(mel_spec_db, sr, hop_length)

# Mel-Spectrogram에서 오디오 복원
restored_audio = restore_audio_from_mel(mel_spec, sr, n_fft, hop_length, n_iter=100)

# 복원된 오디오를 wav 파일로 저장
sf.write('1_restore_enhanced.wav', restored_audio, sr)

print("오디오 복원 ::: '1_restore_enhanced.wav'")
