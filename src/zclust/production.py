from argparse import ArgumentParser
import logging
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import xarray as xr

from pyrocko.util import ensuredirs

from classes import TrainConfig
from dataset import CustomDataset, To3dTensor, MaxAbsScaler as Scaler
from engine import ConvAutoEncoderTrainer, ConvEncoderDECTrainer
from neuralnets import ConvAutoEncoder, ConvEncoderDEC
from log_util import custom_logger, set_loglevel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms


logger = custom_logger('production')


def set_seed_everywhere(seed=2265898541):
    # Default seed is equal to `torch.initial_seed() % 2**32`
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)       # for current GPU
    torch.cuda.manual_seed_all(seed)   # for all GPUs
    # https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = False
    cudnn.deterministic = True


def encode(cae_model, data_loader, device, n_features=None):
    n_samples = len(data_loader.dataset)

    # Switch to evaluation mode
    cae_model.eval()

    with torch.no_grad():
        # First, allocate output tensor
        if n_features is None:
            xdummy = torch.unsqueeze(data_loader.dataset[0], dim=1).to(device)
            zdummy = cae_model.encoder(xdummy)
            _, n_features = zdummy.shape

        codes = torch.empty((n_samples, n_features), dtype=torch.float32)

        for i_batch, xs in enumerate(data_loader):
            xs = xs.to(device)
            i_start = i_batch * data_loader.batch_size
            i_stop = i_start + xs.shape[0]   # Size of current mini-batch
            codes[i_start: i_stop] = cae_model.encoder(xs).cpu()

    return codes


def initialize_cluster_centroid(X, n_clusters, method='kmeans'):
    """
    Initialize cluster centroid locations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training samples to cluster.
    n_clusters : int
        The number of clusters to form.
    method : {'kmeans', 'gmm'}, optional
        The method used to initialize the clusters. Options are 'kmeans'
        (K-Means; default) and 'gmm'(Gaussian Mixture Model).

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
    """
    seed = 13579
    if method == 'kmeans':
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=100,
            max_iter=1000,
            random_state=seed).fit(X)
        return kmeans.cluster_centers_
    else:
        gmm = GaussianMixture(
            n_components=n_clusters,
            n_init=10,
            max_iter=1000,
            random_state=seed).fit(X)
        return gmm.means_


def cluster(dec_model, data_loader, device, n_clusters):
    n_samples = len(data_loader.dataset)

    # Switch to evaluation mode
    dec_model.eval()

    soft_assignments = torch.empty(
        (n_samples, n_clusters), dtype=torch.float32)

    with torch.no_grad():
        for i_batch, xs in enumerate(data_loader):
            xs = xs.to(device)
            qs = dec_model(xs)[0]

            i_start = i_batch * data_loader.batch_size
            i_stop = i_start + xs.shape[0]   # size of current batch
            soft_assignments[i_start: i_stop] = qs.cpu()

    labels = torch.argmax(soft_assignments, dim=1)
    return labels


def main():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(
        title='commands', dest='cmd', required=True)
    train_parser = subparsers.add_parser(
        'train', description='Training Mode')
    encode_parser = subparsers.add_parser(
        'encode', description='Encoding Mode')
    cluster_parser = subparsers.add_parser(
        'cluster', description='Clustering Mode')

    for subparser in (train_parser, encode_parser):
        subparser.add_argument(
            '--step', choices=('pretrain', 'finetune'), required=True,
            help='Training/evaluation step: "pretrain" or "finetune".')

    for subparser in (train_parser, encode_parser, cluster_parser):
        subparser.add_argument(
            '--config', required=True, metavar='FILE', dest='config_file',
            help='Configuration file to use.')
        subparser.add_argument(
            '--num-workers', type=int, default=8, metavar='N',
            help='How many CPUs to use for data loading (default: 8).')
        subparser.add_argument(
            '--no-cuda', dest='use_cuda', action='store_false', default=True,
            help='Disables CUDA training.')
        subparser.add_argument(
            '--need-repro', action='store_true', default=False,
            help='Disables CUDA convolution benchmarking, enables CUDA '
                 'convolution  determinism, and seeds RNGs for python, '
                 'numpy and all devices.')
        subparser.add_argument(
            '--loglevel',
            metavar='LEVEL', default='info',
            choices=('critical', 'error', 'warning', 'info', 'debug'),
            help='Set logger level to "critical", "error", "warning", '
                 '"info", or "debug" (default: info).')

    train_parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='How many mini-batches to wait before logging training '
             'status (default: 10).')
    train_parser.add_argument(
        '--no-plot', dest='make_plot', action='store_false', default=True,
        help='Skips plotting and saving learning curve.')
    train_parser.add_argument(
        '--save', action='store_true', default=False,
        help='Save the current (trained) model, cost values, and logfile.')

    for subparser in (encode_parser, cluster_parser):
        subparser.add_argument(
            '--model', required=True, metavar='FILE',
            help='Trained DEC model. Must be the file "name" only. It will be '
                 'automatically combined with "<run_dirname>/<problem_name>/" '
                 'to get accessed.')

    # Attributes for the main parser and the subparser that was selected
    # by the command line (and not any other subparsers)
    args = parser.parse_args()

    for x in (logger, logging.getLogger('engine')):
        set_loglevel(x, args.loglevel)

    # Reading config file
    config = TrainConfig.read_from_file(args.config_file)
    dconfig = config.dataset_config
    pconfig = config.params_config

    # CUDA for Pytorch
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if args.need_repro:
        set_seed_everywhere()
    else:
        # Improve performance by enabling benchmarking feature
        # see: https://pytorch.org/docs/stable/notes/randomness.html
        cudnn.benchmark = True

    # Data-loading utility
    dataloader_kwargs = {'num_workers': args.num_workers}
    dataloader_kwargs['shuffle'] = False   # For mapping labels to events
    dataloader_kwargs['pin_memory'] = use_cuda is True

    data_set = CustomDataset(
        root_dirpath=dconfig.root_dirpath,
        sample_fname_template=dconfig.sample_fname_template,
        sample_ids=dconfig.sample_ids,
        transform=transforms.Compose([To3dTensor(), Scaler()]))

    data_loader = DataLoader(
        dataset=data_set, batch_size=pconfig.batch_size, **dataloader_kwargs)

    # ----------

    if (hasattr(args, 'step') and args.step == 'pretrain'):
        model = ConvAutoEncoder().to(device)
    else:
        model = ConvEncoderDEC(
            n_clusters=pconfig.n_clusters,
            n_features=pconfig.n_features).to(device)

    # Training mode
    if args.cmd == 'train':
        trainer_kwargs = {
            'optimizer': optim.Adam(model.parameters(),
                                    lr=pconfig.learning_rate),
            'data_loader': data_loader,
            'device': device,
            'n_epochs': pconfig.num_epochs,
            'log_interval': args.log_interval}

        if args.step == 'pretrain':
            trainer = ConvAutoEncoderTrainer(
                model=model,
                loss_fns=[nn.MSELoss(reduction='mean')],
                loss_weights=[1.0],
                **trainer_kwargs)
        else:
            # args.step == 'finetune'
            logger.info('Loading pretrained model.....[!n]')
            model.load_state_dict(
                torch.load(
                    config.save_dirpath.joinpath(pconfig.pretrained_model),
                    map_location=device),
                strict=False)
            print('done')

            logger.info('Initializing clusters.....[!n]')
            codes = encode(model, data_loader, device, pconfig.n_features)
            centroid = initialize_cluster_centroid(
                codes.numpy(), pconfig.n_clusters, pconfig.clust_init_method)
            centroid = torch.from_numpy(centroid.astype(np.float32)).to(device)

            # https://discuss.pytorch.org/t/copy-weights-inside-the-model/65712/2   # noqa
            with torch.no_grad():
                model.clusterer.centroid.copy_(centroid)
            print('done')

            trainer = ConvEncoderDECTrainer(
                model=model,
                loss_fns=[nn.KLDivLoss(reduction='sum')],
                loss_weights=[1.0],
                n_clusters=pconfig.n_clusters,
                tol=pconfig.tol,
                update_interval=pconfig.update_interval,
                **trainer_kwargs)

        # <----
        # Ready to train the model
        trainer.train_all_epochs()

        # Handle output files
        save_dirpath = config.save_dirpath
        out_fname_template = f'{trainer.train_id}_{args.step}_{{}}'

        if args.make_plot:
            # Plot learning curve
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(
                range(1, len(trainer.train_costs) + 1), trainer.train_costs)
            ax.set(xlabel='Number of epoch', ylabel='Average cost')
            fig_fname = save_dirpath.joinpath(
                out_fname_template.format('learning_curve.png')).as_posix()
            ensuredirs(fig_fname)
            fig.savefig(fig_fname, dpi=150, bbox_inches='tight')

        if args.save:
            # Save trained model
            model_fname = save_dirpath.joinpath(
                out_fname_template.format('model.pth')).as_posix()
            ensuredirs(model_fname)
            torch.save(model.state_dict(), model_fname)

            # Save training cost values
            costs_fname = save_dirpath.joinpath(
                out_fname_template.format('costs.npy')).as_posix()
            ensuredirs(costs_fname)
            np.save(costs_fname, trainer.train_costs)

            # Write log-file
            log_fname = save_dirpath.joinpath(
                out_fname_template.format('log.txt')).as_posix()
            with open(log_fname, 'w') as fid:
                fid.write(f'{str(config)}\n\n')

                # Command-line args
                cla = "\n".join(sys.argv[1:])
                fid.write(f'{cla}\n\n')

                if trainer.elapsed_time < 60:
                    fid.write(f'Elapsed time: {trainer.elapsed_time:.2f} sec')
                else:
                    fid.write(
                        f'Elapsed time: {trainer.elapsed_time / 60.:.2f} min')

        if args.make_plot or args.save:
            m = save_dirpath.joinpath(f'{trainer.train_id}_{args.step}_*')
            logger.info(f'Output files: \033[1m{m}\033[0m')

    # <----
    # Encoding or clustering mode
    else:
        model_fpath = config.save_dirpath.joinpath(args.model).resolve()
        if not model_fpath.exists():
            raise FileNotFoundError(f'model not found: {model_fpath}')

        model.load_state_dict(
            torch.load(model_fpath, map_location=device), strict=True)

        if args.cmd == 'encode':
            codes = encode(model, data_loader, device)
            codes = xr.DataArray(
                data=codes.numpy(),
                dims=['sample_ids', 'latent_features'],
                coords={'sample_ids': data_loader.dataset.sample_ids})

            # Save into netCDF file
            codes_fname = model_fpath.name.replace('model.pth', 'encoded.nc')
            codes_fpath = model_fpath.parent.joinpath(codes_fname)
            codes.to_netcdf(path=codes_fpath, mode='w')
            logger.info(f'Output file: \033[1m{codes_fpath}\033[0m')

        elif args.cmd == 'cluster':
            labels = cluster(model, data_loader, device, pconfig.n_clusters)
            labels = xr.DataArray(
                data=labels.numpy(),
                dims=['sample_ids'],
                coords={'sample_ids': data_loader.dataset.sample_ids})

            # Save into netCDF file
            labels_fname = model_fpath.name.replace('model.pth', 'labels.nc')
            labels_fpath = model_fpath.parent.joinpath(labels_fname)
            labels.to_netcdf(path=labels_fpath, mode='w')
            logger.info(f'Output file: \033[1m{labels_fpath}\033[0m')


if __name__ == '__main__':
    main()
