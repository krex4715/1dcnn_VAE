import torch
from VAE import CNN_VAE
import librosa
import numpy as np
import soundfile as sf

# Function to load and normalize the Mel-Spectrogram (Z-score)
def load_mel_spectrogram(file_path, sample_rate=16000, n_mels=128, n_fft=4096, hop_length=1024):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Z-score 정규화
    mel_spec_mean = np.mean(mel_spec_db)
    mel_spec_std = np.std(mel_spec_db)
    mel_spec_normalized = (mel_spec_db - mel_spec_mean) / mel_spec_std
    
    return mel_spec_normalized, mel_spec_mean, mel_spec_std

# 역정규화 함수 (Z-score)
def zscore_denormalize_mel_spectrogram(mel_spec_normalized, mel_spec_mean, mel_spec_std):
    mel_spec_db = mel_spec_normalized * mel_spec_std + mel_spec_mean
    return mel_spec_db


def restore_audio_from_mel(mel_spec, sr, n_fft, hop_length, n_iter=100):
    # Griffin-Lim 알고리즘으로 Mel-spectrogram을 오디오 신호로 복원
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter)
    return audio



# 모델을 불러오고 복원하는 함수
def reconstruct_audio(model, file_path, sample_rate=16000, n_fft=4096, hop_length=1024):
    # 1. Load the audio and normalize the Mel-Spectrogram
    mel_spec_normalized, mel_spec_mean, mel_spec_std = load_mel_spectrogram(file_path, sample_rate, n_fft=n_fft, hop_length=hop_length)
    # 2. Convert to PyTorch tensor and add batch dimension
    mel_spec_tensor = torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0)
    # 3. Pass through the VAE model
    model.eval()
    with torch.no_grad():
        reconstructed, _, _ = model(mel_spec_tensor)
    
    # 4. Convert the output back to numpy and squeeze the batch dimension
    reconstructed_mel_spec_normalized = reconstructed.squeeze(0).numpy()
    
    # 5. Denormalize the reconstructed Mel-Spectrogram
    reconstructed_mel_spec_db = zscore_denormalize_mel_spectrogram(reconstructed_mel_spec_normalized, mel_spec_mean, mel_spec_std)
    
    # 6. Convert Mel-Spectrogram (dB) to power spectrogram
    reconstructed_mel_spec = librosa.db_to_power(reconstructed_mel_spec_db)
    
    # 7. Restore the audio signal from the Mel-Spectrogram
    reconstructed_audio = restore_audio_from_mel(reconstructed_mel_spec, sample_rate, n_fft, hop_length)


    return reconstructed_audio

if __name__ == "__main__":
    # VAE 모델 설정
    input_dim = 157  # Mel-Spectrogram의 time steps 크기
    latent_dim = 16  # Latent space 차원
    vae = CNN_VAE(input_dim, latent_dim)
    vae.load_state_dict(torch.load('vae_model.pth'))

    # 1.wav 파일을 불러와 복원
    reconstructed_audio_1 = reconstruct_audio(vae, '1.wav', n_fft=4096, hop_length=1024)
    sf.write('reconstructed_1.wav', reconstructed_audio_1, 16000)
    print("오디오 복원 ::: 'reconstructed_1.wav'")


    # 2.wav 파일을 불러와 복원
    reconstructed_audio_2 = reconstruct_audio(vae, '2.wav', n_fft=4096, hop_length=1024)
    sf.write('reconstructed_2.wav', reconstructed_audio_2, 16000)
    print("오디오 복원 ::: 'reconstructed_2.wav'")
