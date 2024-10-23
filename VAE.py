import torch
import torch.nn as nn
import librosa
import numpy as np
import torch.nn.functional as F

# Define the VAE class
class CNN_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CNN_VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 1D Convolutional layers followed by linear layers for the mean and variance
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Linear layers for mean (mu) and log-variance (log_var)
        self.fc_mu = nn.Linear(480, latent_dim)
        self.fc_log_var = nn.Linear(480, latent_dim)

        # Decoder: Transposed 1D convolution to reconstruct the input
        self.decoder_input = nn.Linear(latent_dim, 480)  # Match the encoder output
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=7, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 128, kernel_size=7, stride=2),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Apply linear layer to map latent space to decoder input
        z = self.decoder_input(z).view(-1, 32, 15)  # Reshape to match the size expected by ConvTranspose1d
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def load_mel_spectrogram(file_path, sample_rate=16000, n_mels=128, n_fft=4096, hop_length=1024):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db, sr, n_fft, hop_length









if __name__ == '__main__':
    # Create an instance of the CNN_VAE

    input_dim = 157  # This should match your input shape (time dimension of the Mel-Spectrogram)
    latent_dim = 256  # Latent space dimension
    vae = CNN_VAE(input_dim, latent_dim)

    # Load the audio file
    mel_spec_db, sr, n_fft, hop_length = load_mel_spectrogram('1.wav')


    # Convert the mel spectrogram to a PyTorch tensor
    mel_spec_db_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)

    print('mel_spec_db_tensor:', mel_spec_db_tensor.shape)
    # Pass the mel spectrogram through the VAE
    output, mu, logvar = vae(mel_spec_db_tensor)


    print('output:', output.shape)
