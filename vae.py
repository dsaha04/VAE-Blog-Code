from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Initialize the Encoder module.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        super(Encoder, self).__init__()
        self.f = nn.Linear(input_dim, hidden_dim)
        self.mu_phi = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_squared_phi = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input data which is B x D
            where B is the batch size and D is the dimension
            of the input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing
            the mu_phi(x) and log_sigma_squared_phi(x) which are
            each B x M where M is the latent dimension.
        """
        h = torch.relu(self.f(x))
        mu_phi = self.mu_phi(h)
        log_sigma_squared_phi = self.log_sigma_squared_phi(h)
        return mu_phi, log_sigma_squared_phi


# Decoder class
class Decoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: bool = False,
    ):
        """
        Initialize the Decoder module.

        Args:
            z_dim (int): Dimension of the latent space.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output data.
            activation (bool, optional): Whether to apply activation function to the output.
                Defaults to False.
        """
        super(Decoder, self).__init__()
        self.f = nn.Linear(z_dim, hidden_dim)
        self.mu_theta = nn.Linear(hidden_dim, output_dim)
        self.log_sigma_squared_theta = nn.Linear(hidden_dim, output_dim)
        self.activation = F.sigmoid if activation else lambda x: x

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Decoder module.

        Args:
            z (torch.Tensor): Latent space data which is B x M
            where B is the batch size and M is the dimension
            of the latent space.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing
            the mu_theta(z) and log_sigma_squared_theta(z) which are
            each B x D where D is the output dimension.
        """
        h = torch.relu(self.f(z))
        mu_theta = self.activation(self.mu_theta(h))
        log_sigma_squared_theta = self.log_sigma_squared_theta(h)
        return mu_theta, log_sigma_squared_theta


# VAE class
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, L: int):
        """
        Initialize the VAE module.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
            L: (int) is the number of samples to draw from the distribution
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, True)
        self.L = L

    def reparameterize(
        self, mu: torch.Tensor, log_sigma_squared: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterize the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent space,
            it is B x M where B is the batch size and M is
            the dimension of the latent space. Specifically,
            mu = [mu_phi(x_1),...,mu_phi(x_B)] where each mu_phi(x_i)
            is a M-dimensional vector.

            log_sigma_squared (torch.Tensor): Log variance of the latent space,
            it is B x M where B is the batch size and M is the dimension of
            the latent space. Specifically, log_sigma_squared =
            [log_sigma_phi(x_1),...,log_sigma_phi(x_B)] where each log_sigma_phi(x_i)
            is a M-dimensional vector representing the diagonal of the log covariance

        Returns:
            torch.Tensor: Reparameterized latent space which is also B x M.
            Ensures that z_i ~ N(mu_phi(x_i), sigma_squared_phi(x_i)) for
            all i = 1,...,B.
        """
        L = self.L if self.training else 1

        std = torch.exp(0.5 * log_sigma_squared).unsqueeze(1).repeat(1, L, 1)
        eps = torch.randn_like(std)
        latents = mu.unsqueeze(1) + eps * std

        return latents.view(-1, latents.size(-1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE module.

        Args:
            x (torch.Tensor): Input data which is B x D
            where B is the batch size and D is the
            input data dimension or ((B x L) x D)
            where L is the number of samples to draw from the distribution
            if we are in the training mode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Tuple containing the reconstructed input, mu_phi, log_sigma_squared_phi,
            mu_theta, and log_sigma_squared_theta which have the sizes specified in
            the submodules.
        """
        mu_phi, log_sigma_squared_phi = self.encoder(x)
        z = self.reparameterize(mu_phi, log_sigma_squared_phi)
        mu_theta, log_sigma_squared_theta = self.decoder(z)

        std = torch.exp(0.5 * log_sigma_squared_theta)
        x_reconstructed = mu_theta + torch.randn_like(mu_theta) * std
        return (
            x_reconstructed,  # (B, D)
            mu_phi,
            log_sigma_squared_phi,
            mu_theta,  # (B * L, D)
            log_sigma_squared_theta,  # (B * L, D)
        )


def log_multivariate_gaussian_density(x, mu, log_sigma_squared, sample_size):
    """
    Compute the log density of a multivariate Gaussian distribution.

    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution which
        B x M where B is the batch size and M is the dimension of
        the latent space.

        Namely for the input data x = [x_1, ..., x_B] where
        each x_i is a M-dimensional vector, here
        mu = [mu_theta(z_1),...,mu_theta(z_B)] where each mu_theta(z_i)
        is a M-dimensional vector. Furthermore, these z_i are sampled
        from N(mu_phi(x_i), sigma_squared_phi(x_i)).

        log_sigma_squared (torch.Tensor): Log variance of the Gaussian
        distribution which is B x M where B is the batch size
        and M is the dimension of the latent space.

        Namely for the input data x = [x_1, ..., x_B] where
        each x_i is a M-dimensional vector, here log_sigma =
        [log_sigma_theta(z_1),...,log_sigma_theta(z_B)] where each
        log_sigma_theta(z_i) is a M-dimensional vector representing
        the diagonal of the log covariance matrix (since we assume
        diagonal covariance matrix). Furthermore, these z_i are sampled
        from N(mu_phi(x_i), sigma_squared_phi(x_i)).

    Returns:
        torch.Tensor: Log density of the multivariate Gaussian
        distribution which is B x 1 where B is the batch size. Specifically,
        we evaluate the log density at each x_i using the corresponding
        mean and covariance matrix.
    """
    k = mu.size(-1)
    x = x.unsqueeze(1).repeat(1, sample_size, 1)  # (B, L, M)
    x = x.view(-1, k)

    diff = x - mu  # (B * L, M)
    diff = diff.unsqueeze(dim=1)  # (B * L, 1, M)

    sigma_squared = torch.exp(log_sigma_squared) + 1e-6  # to avoid degenerate solutions
    log_sigma_squared = torch.log(sigma_squared)

    # Compute the determinant and inverse of the covariance matrix
    log_sigma_squared_det = torch.sum(log_sigma_squared, dim=-1, keepdim=True)  # (B, 1)

    sigma_squared_inv = (1.0 / sigma_squared).unsqueeze(dim=1)  # (B, 1, M)
    sigma_squared_inv_diff = diff * sigma_squared_inv

    # Compute the exponent term
    exponent = -0.5 * (sigma_squared_inv_diff @ diff.transpose(1, 2))
    exponent = exponent.squeeze(-1)  # (B * L, 1)

    # Compute the normalization term
    normalization = -0.5 * (
        k * torch.log(torch.tensor(2 * torch.pi)) + log_sigma_squared_det
    )

    log_density = normalization + exponent
    log_density = log_density.view(-1, sample_size)  # (B, L, M)
    log_density = torch.mean(log_density, dim=1)  # (B, M)

    return log_density


def loss_function(
    x, mu_phi, log_sigma_squared_phi, mu_theta, log_sigma_squared_theta, sample_size
):
    NLL = -log_multivariate_gaussian_density(
        x, mu_theta, log_sigma_squared_theta, sample_size
    )
    KLD = -0.5 * torch.sum(
        1 + log_sigma_squared_phi - mu_phi.pow(2) - log_sigma_squared_phi.exp()
    )
    return NLL.sum() + KLD


def main():
    # Parameters
    input_dim = 784  # 28x28 images flattened
    hidden_dim = 400
    latent_dim = 10
    learning_rate = 1e-3
    num_epochs = 50
    batch_size = 128
    sample_size = 10

    # DataLoader
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, and loss function
    model = VAE(input_dim, hidden_dim, latent_dim, sample_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim)
            data = Variable(data)

            optimizer.zero_grad()

            _, mu_phi, log_sigma_squared_phi, mu_theta, log_sigma_squared_theta = model(
                data
            )

            loss = loss_function(
                data,
                mu_phi,
                log_sigma_squared_phi,
                mu_theta,
                log_sigma_squared_theta,
                sample_size,
            )
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")

    # Testing loop (optional, just for checking the reconstruction)
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.view(-1, input_dim)
            data = Variable(data)

            (
                x_reconstructed,
                mu_phi,
                log_sigma_squared_phi,
                mu_theta,
                log_sigma_squared_theta,
            ) = model(data)

            x_reconstructed = torch.clamp(
                x_reconstructed, min=0, max=1
            )  # clamp to [0, 1] so that we can visualize the images

            # Save the first 8 input images and their reconstructions
            if i == 0:
                import matplotlib.pyplot as plt

                n = 8
                comparison = torch.cat(
                    [
                        data[:n].view(-1, 1, 28, 28),
                        x_reconstructed[:n].view(-1, 1, 28, 28),
                    ]
                )
                comparison = comparison.cpu().numpy()
                fig, axes = plt.subplots(
                    nrows=2, ncols=n, sharex=True, sharey=True, figsize=(12, 4)
                )
                for images, row in zip([comparison[:n], comparison[n:]], axes):
                    for img, ax in zip(images, row):
                        ax.imshow(img.reshape((28, 28)), cmap="gray")
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                plt.show()
                break


if __name__ == "__main__":
    main()
