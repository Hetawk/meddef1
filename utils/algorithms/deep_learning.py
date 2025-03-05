<<<<<<< HEAD
# deep_learning.py

import torch
import torch.nn as nn

class DeepLearningAlgorithms:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(DeepLearningAlgorithms.Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    class DenoisingAutoencoder(Autoencoder):
        def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.2):
            super(DeepLearningAlgorithms.DenoisingAutoencoder, self).__init__(input_dim, hidden_dim, latent_dim)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            noisy_input = self.dropout(x)
            encoded = self.encoder(noisy_input)
            decoded = self.decoder(encoded)
            return decoded

    class VariationalAutoencoder(Autoencoder):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(DeepLearningAlgorithms.VariationalAutoencoder, self).__init__(input_dim, hidden_dim, latent_dim)
            self.mu_layer = nn.Linear(hidden_dim, latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            encoded = self.encoder(x)
            mu = self.mu_layer(encoded)
            logvar = self.logvar_layer(encoded)
            z = self.reparameterize(mu, logvar)
            decoded = self.decoder(z)
            return decoded, mu, logvar

    class CNN(nn.Module):
        def __init__(self, num_classes=10):
            super(DeepLearningAlgorithms.CNN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc1 = nn.Linear(8 * 8 * 32, 1000)
            self.fc2 = nn.Linear(1000, num_classes)
            # Reduce batch size
            self.batch_size = 2

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(DeepLearningAlgorithms.RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(DeepLearningAlgorithms.LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
=======
# deep_learning.py

import torch
import torch.nn as nn

class DeepLearningAlgorithms:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(DeepLearningAlgorithms.Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    class DenoisingAutoencoder(Autoencoder):
        def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.2):
            super(DeepLearningAlgorithms.DenoisingAutoencoder, self).__init__(input_dim, hidden_dim, latent_dim)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            noisy_input = self.dropout(x)
            encoded = self.encoder(noisy_input)
            decoded = self.decoder(encoded)
            return decoded

    class VariationalAutoencoder(Autoencoder):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(DeepLearningAlgorithms.VariationalAutoencoder, self).__init__(input_dim, hidden_dim, latent_dim)
            self.mu_layer = nn.Linear(hidden_dim, latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            encoded = self.encoder(x)
            mu = self.mu_layer(encoded)
            logvar = self.logvar_layer(encoded)
            z = self.reparameterize(mu, logvar)
            decoded = self.decoder(z)
            return decoded, mu, logvar

    class CNN(nn.Module):
        def __init__(self, num_classes=10):
            super(DeepLearningAlgorithms.CNN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc1 = nn.Linear(8 * 8 * 32, 1000)
            self.fc2 = nn.Linear(1000, num_classes)
            # Reduce batch size
            self.batch_size = 2

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(DeepLearningAlgorithms.RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(DeepLearningAlgorithms.LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
>>>>>>> 16c5cfd9eac902321ee831908acfc69f3a52f936
            return out