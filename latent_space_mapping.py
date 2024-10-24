import torch
from VAE import CNN_VAE
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

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





def get_latent_space(encoder, file_path, sample_rate=16000, n_fft=4096, hop_length=1024):
    mel_spec_normalized, mel_spec_mean, mel_spec_std = load_mel_spectrogram(file_path, sample_rate, n_fft=n_fft, hop_length=hop_length)
    mel_spec_tensor = torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0)
    
    # 인코더를 통해 잠재공간 값 추출
    mu, log_var = encoder(mel_spec_tensor)
    return mu, log_var, mel_spec_mean, mel_spec_std


def interpolate_latent_space(mu_1, mu_2, num_steps=10):
    interpolated_latents = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)  # 0 to 1 사이의 보간 비율
        interpolated_latent = (1 - alpha) * mu_1 + alpha * mu_2
        interpolated_latents.append(interpolated_latent)
    return interpolated_latents



def decode_latent_to_audio(decoder, latent, mel_spec_mean, mel_spec_std, sample_rate=16000, n_fft=4096, hop_length=1024):
    with torch.no_grad():
        reconstructed_mel_spec = decoder(latent).squeeze(0).numpy()
    
    # Mel-Spectrogram을 Denormalize
    reconstructed_mel_spec_db = zscore_denormalize_mel_spectrogram(reconstructed_mel_spec, mel_spec_mean, mel_spec_std)
    
    # Mel-Spectrogram을 Power Spectrogram으로 변환
    reconstructed_mel_spec_power = librosa.db_to_power(reconstructed_mel_spec_db)
    
    # Griffin-Lim 알고리즘으로 오디오 복원
    audio = restore_audio_from_mel(reconstructed_mel_spec_power, sample_rate, n_fft, hop_length)
    
    return audio , reconstructed_mel_spec_db


def visualize_latent_space(mu_1, mu_2, interpolated_latents, num_steps):
    mu_1_np = mu_1.cpu().detach().numpy().flatten()
    mu_2_np = mu_2.cpu().detach().numpy().flatten()

    # 보간된 잠재공간 값들을 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(mu_1_np, label='mu_1 (1.wav)', marker='o')
    plt.plot(mu_2_np, label='mu_2 (2.wav)', marker='x')

    # 보간된 각 잠재공간 값들을 시각화
    for i, latent in enumerate(interpolated_latents):
        latent_np = latent.cpu().detach().numpy().flatten()
        plt.plot(latent_np, label=f'Interpolated mu {i}', linestyle='--')

    plt.title("Latent Space Visualization: mu_1 to mu_2")
    plt.xlabel("Latent Space Dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()





# Mel-Spectrogram을 이미지로 저장하는 함수
def save_mel_spectrogram_image(mel_spec_db, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=16000, hop_length=1024, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram (dB)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()









class CNN_VAE_encoder(torch.nn.Module):
    def __init__(self, cvae):
        super(CNN_VAE_encoder, self).__init__()
        self.encoder = cvae.encoder
        self.fc_mu = cvae.fc_mu
        self.fc_log_var = cvae.fc_log_var

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    

class CNN_VAE_decoder(torch.nn.Module):
    def __init__(self, cvae):
        super(CNN_VAE_decoder, self).__init__()
        self.decoder_input = cvae.decoder_input
        self.decoder = cvae.decoder

    def forward(self, z):
        z = self.decoder_input(z).view(-1, 32, 15)  # Reshape to match the size expected by ConvTranspose1d
        return self.decoder(z)
    






if __name__ == "__main__":
    # VAE 모델 설정
    input_dim = 157  # Mel-Spectrogram의 time steps 크기
    latent_dim = 8  # Latent space 차원
    vae = CNN_VAE(input_dim, latent_dim)
    vae.load_state_dict(torch.load('vae_model.pth'))

    # encoder, decoder 설정
    vae_encoder = CNN_VAE_encoder(vae)
    vae_decoder = CNN_VAE_decoder(vae)

    # 1.wav 파일에서 잠재공간 추출
    mu_1, log_var_1, mel_mean_1, mel_std_1 = get_latent_space(vae_encoder, '1.wav')

    # 2.wav 파일에서 잠재공간 추출
    mu_2, log_var_2, mel_mean_2, mel_std_2 = get_latent_space(vae_encoder, '2.wav')

    print('mu_1:', mu_1)

    print('mu_2:', mu_2)

    # mu 값만을 사용하여 보간
    interpolated_latents = interpolate_latent_space(mu_1, mu_2, num_steps=10)

    # 시각화
    visualize_latent_space(mu_1, mu_2, interpolated_latents, num_steps=10)



    base_dir = './results/'
    os.makedirs(base_dir, exist_ok=True)
    

    # 보간된 잠재공간 값들을 디코더를 통해 오디오로 변환
    for i, latent in enumerate(interpolated_latents):
        # 각 보간된 잠재공간을 사용하여 Mel-Spectrogram 복원 및 오디오 변환
        audio, reconstructed_mel_spec_db = decode_latent_to_audio(vae_decoder, latent, mel_mean_1, mel_std_1)  # mel_mean_1과 mel_std_1 사용

        # 복원된 Mel-Spectrogram을 이미지로 저장
        save_mel_spectrogram_image(reconstructed_mel_spec_db, f'{base_dir}interpolated_mel_{i}.png')
        print(f"Mel-Spectrogram 이미지 저장 ::: 'interpolated_mel_{i}.png'")

        # 변환된 오디오를 파일로 저장
        sf.write(f'{base_dir}interpolated_audio_{i}.wav', audio, 16000)
        print(f"오디오 복원 ::: 'interpolated_audio_{i}.wav'")
