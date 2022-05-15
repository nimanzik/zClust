import torch
from torch.utils.data import Dataset
import xarray as xr


class To3dTensor(object):
    def __call__(self, x):
        return torch.tensor(x[None, :, :], dtype=torch.float32)


class VectorNormScaler(object):
    def __call__(self, x):
        return x / torch.linalg.norm(x)


class MaxAbsScaler(object):
    def __call__(self, x):
        return x / torch.max(torch.abs(x))


class CustomDataset(Dataset):
    def __init__(
            self, root_dirpath, sample_fname_template, sample_ids,
            transform=None):
        self.root_dirpath = root_dirpath
        self.sample_fname_template = sample_fname_template
        self.sample_ids = sample_ids
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_fname = self.sample_fname_template.substitute(
            sample_id=sample_id)
        sample_fpath = self.root_dirpath.joinpath(sample_fname)
        sample = xr.load_dataarray(sample_fpath.as_posix()).data   # ndarray

        if self.transform:
            sample = self.transform(sample)

        return sample


__all__ = """
    To3dTensor
    VectorNormScaler
    MaxAbsScaler
    CustomDataset
""".split()
