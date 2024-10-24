import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from VAE import CNN_VAE
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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


# Loss function (Reconstruction loss + KL divergence)
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss


# Custom Dataset class
class AudioDataset(Dataset):
    def __init__(self, file_paths, sample_rate=16000):
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.mean_std_values = []  # 각 파일의 mean/std 값을 저장하기 위한 리스트

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel_spec, mel_spec_mean, mel_spec_std = load_mel_spectrogram(self.file_paths[idx], self.sample_rate)
        self.mean_std_values.append((mel_spec_mean, mel_spec_std))  # mean/std 값을 저장
        return torch.tensor(mel_spec, dtype=torch.float32)

    def get_mean_std_values(self):
        return self.mean_std_values  # 저장된 mean/std 값 반환


# Training function
def train_vae(model, dataloader, optimizer, epochs=10, log_interval=1):
    writer = SummaryWriter()  # TensorBoard SummaryWriter 시작
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            loss = vae_loss(recon_batch, batch, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        writer.add_scalar('Loss/train', avg_loss, epoch)  # TensorBoard에 손실 값 기록
        if epoch % log_interval == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    writer.close()  # 학습 완료 후 SummaryWriter 종료


if __name__ == "__main__":

    # List of audio files
    file_paths = ['1.wav', '2.wav']

    # Create an instance of the dataset
    audio_dataset = AudioDataset(file_paths)

    # Create DataLoader
    audio_dataloader = DataLoader(audio_dataset, batch_size=2, shuffle=True)

    # Iterate through the DataLoader
    for i, batch in enumerate(audio_dataloader):
        print(f"Batch {i+1} data shape: {batch.shape}")

    # VAE 모델 설정
    input_dim = 157  # Mel-Spectrogram의 time steps 크기
    latent_dim = 8  # Latent space 차원
    vae = CNN_VAE(input_dim, latent_dim)

    # Optimizer 설정
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # 학습 실행 (10번의 에포크)
    train_vae(vae, audio_dataloader, optimizer, epochs=3000)

    # 모델 저장
    torch.save(vae.state_dict(), 'vae_model.pth')



    # 학습 후 역정규화를 위한 mean/std 값 가져오기
    mean_std_values = audio_dataset.get_mean_std_values()
    print("Mean/Std values for each file:", mean_std_values)
