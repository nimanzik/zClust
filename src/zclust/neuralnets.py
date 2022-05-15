import numpy as np

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Parameters
    ----------
    lrelu_coeff : float (optional)
        Leaky ReLU parameter that controls the angle of its negative
        slope (default: 0.0).
    """

    def __init__(self, lrelu_coeff=0.02):
        super().__init__()
        self.encoder = nn.Sequential(
            # (N, 1, 87, 67) -> (N, 8, 44, 34)
            nn.Conv2d(1, 8, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 8, 44, 34) -> (N, 16, 22, 17)
            nn.Conv2d(8, 16, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 16, 22, 17) -> (N, 32, 11, 9)
            nn.Conv2d(16, 32, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 32, 11, 9) -> (N, 64, 6, 5)
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 64, 6, 5) -> (N, 128, 3, 3)
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 128, 3, 3) -> (N, 1152)
            nn.Flatten(),
            # (N, 1152) -> (N, 9)
            nn.Linear(1152, 9),
            nn.LeakyReLU(lrelu_coeff))

    def forward(self, x):
        """
        Returns
        -------
        z : tensor of shape (n_samples, n_features)
            Encoded vectors. `n_features` indicates the dimension of the
            latent space.
        """
        z = self.encoder(x)
        return z


class ConvDecoder(nn.Module):
    """
    Parameters
    ----------
    lrelu_coeff : float (optional)
        Leaky ReLU parameter that controls the angle of its negative
        slope (default: 0.0).
    """

    def __init__(self, lrelu_coeff=0.02):
        super().__init__()
        self.z2d = nn.Sequential(
            nn.Linear(9, 1152),
            nn. LeakyReLU(lrelu_coeff))
        self.decoder = nn.Sequential(
            # (N, 128, 3, 3) -> (N, 64, 5, 5)
            nn.ConvTranspose2d(128, 64, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 64, 5, 5) -> (N, 32, 11, 9)
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(0, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 32, 11, 9) -> (N, 16, 23, 17)
            nn.ConvTranspose2d(32, 16, (3, 3), stride=(2, 2), padding=(0, 1)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 16, 23, 17) -> (N, 8, 45, 35)
            nn.ConvTranspose2d(16, 8, (3, 3), stride=(2, 2), padding=(1, 0)),
            nn.LeakyReLU(lrelu_coeff),
            # (N, 8, 45, 35) -> (N, 1, 89, 69)
            nn.ConvTranspose2d(8, 1, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(lrelu_coeff))

    def forward(self, z):
        d = self.z2d(z)
        d = d.view(-1, 128, 3, 3)
        x_rec = self.decoder(d)
        # Crop and then return
        return x_rec[:, :, 1:-1, 1:-1]


class ConvAutoEncoder(nn.Module):
    """
    Parameters
    ----------
    lrelu_coeff : float (optional)
        Leaky ReLU parameter that controls the angle of its negative
        slope (default: 0.0).
    """

    def __init__(self, lrelu_coeff=0.02):
        super().__init__()
        self.encoder = ConvEncoder(lrelu_coeff=lrelu_coeff)
        self.decoder = ConvDecoder(lrelu_coeff=lrelu_coeff)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return (x_rec, z)


class STDSoftClustering(nn.Module):
    """
    Soft clustering using a simplified Student's t-distribution.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    n_features : int
        The number of latent space dimensions (the length of encoded vectors).
    centroid : ndarray of shape (n_clusters, n_features)
        A priori (initialized) cluster centroid locations (default: None).
    dof : int (optional)
        The degree of freedom of the Student's t-distribution (default: 1).
    """

    def __init__(self, *, n_clusters, n_features, centroid=None, dof=1):
        super().__init__()

        self.n_clusters = n_clusters
        self.n_features = n_features
        self.dof = dof

        # We should always "register" all possible parameters.
        # `nn.Parameter` will get automatically registered.
        # https://pytorch.org/docs/stable/notes/extending.html#adding-a-module
        if centroid is None:
            centroid = torch.empty(
                (self.n_clusters, self.n_features), dtype=torch.float32)
            nn.init.xavier_uniform_(centroid)
            self.centroid = nn.Parameter(centroid)
        else:
            self.centroid = nn.Parameter(
                torch.from_numpy(centroid.astype(np.float32)))

    def forward(self, z):
        """
        Parameters
        ----------
        z : tensor of shape (n_samples, n_features)
            Embedded vectors returned by `~ConvEncoder` model.

        Returns
        -------
        q : tensor of shape (n_samples, n_clusters)
            Soft-clustring assignments.
        """
        # First, distances with shape of (n_samples, n_clusters, n_features)
        q = z.unsqueeze(1) - self.centroid
        q = torch.mul(q, q)
        q = torch.sum(q, dim=2)   # shape: (n_samples, n_clusters)
        q = 1.0 + (q / self.dof)
        q = torch.pow(q, -0.5 * (self.dof + 1))
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


class ConvAutoEncoderDEC(nn.Module):
    """
    Deep embedded clustering using convolutional *autoencoder*.
    """

    def __init__(
            self, *, n_clusters, n_features, centroid=None, dof=1,
            lrelu_coeff=0.02):

        super().__init__()
        self.encoder = ConvEncoder(lrelu_coeff=lrelu_coeff)
        self.decoder = ConvDecoder(lrelu_coeff=lrelu_coeff)
        self.clusterer = STDSoftClustering(
            n_clusters=n_clusters,
            n_features=n_features,
            centroid=centroid,
            dof=dof)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        q = self.clusterer(z)
        return (q, x_rec, z)


class ConvEncoderDEC(nn.Module):
    """
    Deep embedded clustering using convolutional *encoder*.
    """

    def __init__(
            self, n_clusters, n_features, centroid=None, dof=1,
            lrelu_coeff=0.02):

        super().__init__()
        self.encoder = ConvEncoder(lrelu_coeff=lrelu_coeff)
        self.clusterer = STDSoftClustering(
            n_clusters=n_clusters,
            n_features=n_features,
            centroid=centroid,
            dof=dof)

    def forward(self, x):
        z = self.encoder(x)
        q = self.clusterer(z)
        return (q, z)


__all__ = """
    ConvEncoder
    ConvDecoder
    ConvAutoEncoder
    STDSoftClustering
    ConvAutoEncoderDEC
""".split()
