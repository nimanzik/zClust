from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from log_util import custom_logger

import torch


logger = custom_logger('engine')


class BaseTrainer(ABC):
    def __init__(
            self, *, model, optimizer, loss_fns, data_loader, device,
            loss_weights=None, n_epochs=1, log_interval=10):

        self.model = model
        self.optimizer = optimizer
        self.loss_fns = loss_fns   # list of loss functions
        self.data_loader = data_loader
        self.device = device
        self.n_epochs = n_epochs
        self.loss_weights = loss_weights   # list of floats or None
        self.log_interval = log_interval
        self.n_samples = len(self.data_loader.dataset)
        self.n_batches = len(self.data_loader)
        self.train_id = None
        self.elapsed_time = None
        self.train_costs = None

        if len(self.loss_fns) == 1:
            self.loss_fn = self.loss_fns[0]

    @abstractmethod
    def compute_forward_cost(self, data, i_batch) -> torch.Tensor:
        """
        Performs forward pass by processing inputs through the model and
        then computes the cost.

        Parameters
        ----------
        data : 2-tuple of `torch.Tensor` or one single `torch.Tensor`
            Batch of data.
            * In supervised problems, it is a tuple of two tensors
              containing a batch of features and labels, respectively.
            * In unsupervised problems, it is a single tensor containing
              a batch of features.
        i_batch : int
            Current batch index. Indexing starts from zero.

        Returns
        -------
        cost : torch.Tensor
            How far is the output from being correct
        """
        pass

    def train_one_batch(self, data, i_batch) -> float:
        """
        Parameters
        ----------
        data : 2-tuple of `torch.Tensor` or one single `torch.Tensor`
            Batch of data.
            * In supervised problems, it is a tuple of two tensors
              containing a batch of features and labels, respectively.
            * In unsupervised problems, it is a single tensor containing
              a batch of features.
        i_batch : int
            Current batch index. Indexing starts from zero.

        Returns
        -------
        Batch training cost.
        """

        # Forward
        cost = self.compute_forward_cost(data, i_batch)

        # Backward (zero the parameter gradients first)
        self.optimizer.zero_grad()
        cost.backward()

        # Optimize (update model parameters)
        self.optimizer.step()

        batch_cost = cost.item()

        # -----
        # Logging
        nd = int(np.floor(np.log10(self.n_batches))) + 1
        batch_num = i_batch + 1
        if (batch_num % self.log_interval == 0) \
                or (batch_num == self.n_batches):

            formatted_bc = np.format_float_scientific(
                batch_cost, unique=False, precision=6)

            logger.info(
                f'\u251C\u2500 Batch '   # use box-drawing characters
                f'[{batch_num:{nd}d}/{self.n_batches:{nd}d} '
                f'({(batch_num / self.n_batches * 100):3.0f}%)] '
                f'Cost: {formatted_bc}')

        return batch_cost

    def train_one_epoch(self, i_epoch) -> float:
        """
        Parameters
        ----------
        i_epoch : int
            Current epoch index. Indexing starts from zero.

        Returns
        -------
        Epoch average training cost.
        """
        epoch_num = i_epoch + 1
        logger.info(f'Train epoch {epoch_num}/{self.n_epochs}')

        # Switch to train mode
        self.model.train()

        epoch_cost = 0.0
        for i_batch, data in enumerate(self.data_loader):
            batch_cost = self.train_one_batch(data, i_batch)
            epoch_cost += batch_cost

        # -----
        # Epoch average training cost
        epoch_cost /= self.n_samples
        formatted_ec = np.format_float_scientific(
            epoch_cost, unique=False, precision=6)
        logger.info(
            f'\u2514\u2500 Average running cost: '
            f'\033[1m{formatted_ec}\033[0m')

        return epoch_cost

    def train_all_epochs(self) -> None:
        tic = datetime.now()
        self.train_id = tic.strftime('%Y-%m-%d_%H-%M-%S')

        train_costs = []
        for i_epoch in range(self.n_epochs):
            epoch_cost = self.train_one_epoch(i_epoch)
            train_costs.append(epoch_cost)

        toc = datetime.now()
        self.elapsed_time = (toc - tic).total_seconds()

        self.train_costs = train_costs


class ConvAutoEncoderTrainer(BaseTrainer):
    """
    Class to train a `~ConvAutoEncoder`, a convolutional autoencoder
    model that computes both reconstructed and encoded data.
    """

    def compute_forward_cost(self, xs, *args):
        """
        Parameters
        ----------
        xs : `torch.Tensor` object
            Batch of data containing features (i.e. training samples).

        Returns
        -------
        Reconstruction cost.
        """
        xs = xs.to(self.device)
        xs_rec, _ = self.model(xs)
        rec_cost = self.loss_fn(xs_rec, xs)   # MSELoss
        return rec_cost


class BaseDECTrainer(BaseTrainer):
    """
    Class to train a `~ConvEncoderDEC`, a deep embedded clustering
    model that uses a convolutional encoder only and a soft clusterer.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to from. Must be greater than 1.
    tol : float
        Training will stop if cluster-assignment change in percentage is
        less than `tol`. Must be `0 <= tol <= 100`.
    """

    def __init__(self, *, n_clusters, tol, update_interval, **kwargs):
        super().__init__(**kwargs)

        if tol < 0.0 or tol > 1.00:
            raise ValueError(
                f'Threshold value must be between 0 and 100: {tol}')

        self.n_clusters = n_clusters
        self.tol = tol

    def eval_soft_assignments(self) -> torch.Tensor:
        """
        Evaluate soft clustering assignments (or the memebership
        probabilities, Q) using current state of the DEC model.

        Returns
        -------
        Soft assignments : tensor of shape (n_samples, n_clusters)
        """
        # First, check whether model is in training mode
        training = self.model.training

        # Switch to evaluation mode
        self.model.eval()

        soft_assignments = torch.empty(
            (self.n_samples, self.n_clusters), dtype=torch.float32)

        with torch.no_grad():
            for i_batch, xs in enumerate(self.data_loader):
                xs = xs.to(self.device)
                qs = self.model(xs)[0]   # Model must return Qs as 1st output
                i_start = i_batch * self.data_loader.batch_size
                i_stop = i_start + xs.shape[0]   # Size of current batch
                soft_assignments[i_start: i_stop] = qs.cpu()

        # Should we switch back to train mode?
        if training is True:
            self.model.train()

        return soft_assignments

    def train_all_epochs(self) -> None:
        tic = datetime.now()
        self.train_id = tic.strftime('%Y-%m-%d_%H-%M-%S')

        train_costs = []
        for i_epoch in range(self.n_epochs):

            # Soft clusterings 'before' training this epoch
            q_pre = self.eval_soft_assignments()
            labels_pre = torch.argmax(q_pre, dim=1)

            # Train one epoch
            epoch_cost = self.train_one_epoch(i_epoch)
            train_costs.append(epoch_cost)

            logger.info('Checking stopping criterion.....[!n]')
            # Soft clusterings 'after' training this epoch
            q_post = self.eval_soft_assignments()
            labels_post = torch.argmax(q_post, dim=1)

            delta_labels = torch.sum(labels_pre != labels_post).item()
            delta_labels *= (100.0 / self.n_samples)
            print('done')

            if delta_labels < self.tol:
                logger.warning(
                    f'\033[1mStopped training. Cluster assignment change '
                    f'reached less than {self.tol}%\033[0m')
                break

        toc = datetime.now()
        self.elapsed_time = (toc - tic).total_seconds()
        self.train_costs = train_costs


class ConvEncoderDECTrainer(BaseDECTrainer):
    """
    Class to train a `~ConvAutoEncoderDEC`, a deep embedded clustering
    model that uses a convolutional autoencoder and a soft clusterer.
    """

    def compute_forward_cost(self, xs, i_batch):
        """
        Parameters
        ----------
        xs : `torch.Tensor` object
            Batch of data containing features (i.e. training samples).
        i_batch : int
            Current batch index. Indexing starts from zero.

        Returns
        -------
        Clustering cost.

        Notes
        -----
        As all the other losses in PyTorch, KLDiv function expects the
        first argument to be the output of the model and the second to
        be the observations in the dataset. This differs from the
        standard mathematical notation KL(P ∣∣ Q) where P denotes the
        distribution of the observations and Q denotes the model.
        """
        xs = xs.to(self.device)
        qs = self.model(xs)[0]

        # Auxiliary target distribution
        ps = torch.pow(qs, 2) / torch.sum(qs, dim=0)
        ps = ps / torch.sum(ps, dim=1, keepdim=True)
        ps = ps.detach()

        clust_cost = self.loss_fn(torch.log(qs), ps) / xs.shape[0]   # KLDiv
        return clust_cost


class ConvAutoEncoderDECTrainer(BaseDECTrainer):
    """
    Class to train a `~ConvAutoEncoderDEC`, a deep embedded clustering
    model that uses a convolutional autoencoder and a soft clusterer.
    """

    def compute_forward_cost(self, xs, i_batch):
        """
        Parameters
        ----------
        xs : `torch.Tensor` object
            Batch of data containing features (i.e. training samples).
        i_batch : int
            Current batch index. Indexing starts from zero.

        Returns
        -------
        Sum of reconstruction and clustering cost.

        Notes
        -----
        As all the other losses in PyTorch, KLDiv function expects the
        first argument to be the output of the model and the second to
        be the observations in the dataset. This differs from the
        standard mathematical notation KL(P ∣∣ Q) where P denotes the
        distribution of the observations and Q denotes the model.
        """
        xs = xs.to(self.device)
        qs, xs_rec, _ = self.model(xs)

        # Reconstruction cost (MSELoss)
        rec_cost = self.loss_weights[0] * self.loss_fns[0](xs_rec, xs)

        # Auxiliary target distribution
        ps = torch.pow(qs, 2) / torch.sum(qs, dim=0)
        ps = ps / torch.sum(ps, dim=1, keepdim=True)
        ps = ps.detach()

        # Clustering cost (KLDiv)
        clust_cost = (
            self.loss_weights[1]
            * self.loss_fns[1](torch.log(qs), ps) / xs.shape[0])

        total_cost = rec_cost + clust_cost
        return total_cost


__all__ = """
    BaseTrainer
    ConvAutoEncoderTrainer
    BaseDECTrainer
    ConvEncoderDECTrainer
    ConvAutoEncoderDECTrainer
""".split()
