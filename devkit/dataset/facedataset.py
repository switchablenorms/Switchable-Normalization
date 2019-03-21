from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import math
import torch
from torch.utils.data.sampler import Sampler
from torch.distributed import get_world_size, get_rank

def pil_loader(filename):
    with Image.open(filename) as img:
        img = img.convert('RGB')
    return img
 
class FaceDataset(Dataset):
    def __init__(self, istraining = True, args = None, transform=None):
        self.transform = transform
        self.istraining = istraining
        self.args = args
        self.metas = []

        if istraining:
            with open(args.train_list) as f:
                lines = f.readlines()
            print("building dataset from %s"%args.train_list)
            self.num = len(lines)
            for line in lines:
                path, cls = line.rstrip().split()
                self.metas.append((args.train_root + '/' + path, int(cls)))
        else:
            with open(args.probe_list) as f:
                lines = f.readlines()
            print("building dataset from %s" % args.probe_list)

            for line in lines:
                path= line.rstrip()
                self.metas.append((args.probe_root + '/' + path, 0))

            with open(args.distractor_list) as f:
                lines = f.readlines()
            print("building dataset from %s" % args.distractor_list)

            for line in lines:
                path = line.rstrip()
                self.metas.append((args.distractor_root + '/' + path, 0))

            self.num = len(self.metas)
        self.initialized = False
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, idx):
        filename = self.metas[idx][0]
        cls = self.metas[idx][1]

        img = pil_loader(filename)

        ## transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, shuffle=True, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.shuffle = shuffle
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class BigdataSampler(Sampler):

    def __init__(self,
                 dataset,
                 num_sub_epochs,
                 finegrain_factor,
                 shuffle=True,
                 psudo_index=None,
                 seed=2333,
                 auto_step=True,
                 first_step=True,
                 world_size=None,
                 rank=None):
        self.dataset = dataset
        self.num_sub_epochs = num_sub_epochs
        self.shuffle = shuffle
        self.psudo_index = psudo_index
        self.finegrain_factor = finegrain_factor
        self.auto_step = auto_step
        self.first_step = first_step

        self.world_size = world_size if world_size is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()

        self.np_rng = np.random.RandomState(seed)
        self.split_index = self.num_sub_epochs - 1

        # pad length to make total number of indices divisible by (world_size x num_sub_epochs x finegrain_factor)
        self.origin_num_samples = len(self.dataset)
        num_units = self.world_size * self.num_sub_epochs * self.finegrain_factor
        self.unit_num_samples = math.ceil(self.origin_num_samples / num_units)
        self.padded_num_samples = self.unit_num_samples * num_units
        self.sampler_length = self.padded_num_samples // self.world_size // self.num_sub_epochs

    def reset_split(self):
        section_num_samples = self.padded_num_samples // self.num_sub_epochs // self.finegrain_factor
        unit_beg_indices = np.arange(0, self.padded_num_samples, section_num_samples)
        unit_end_indices = unit_beg_indices + section_num_samples
        unit_array = np.stack([unit_beg_indices, unit_end_indices]).T
        if self.shuffle:
            # shuffle operation (1): shuffle at unit level
            self.np_rng.shuffle(unit_array)
        self.split_array = unit_array.reshape([self.num_sub_epochs, self.finegrain_factor, 2])

    def step(self):
        assert self.split_index >= 0 and self.split_index < self.num_sub_epochs

        # increase split_index
        if self.split_index == self.num_sub_epochs - 1:
            self.reset_split()
            self.split_index = 0
        else:
            self.split_index += 1

        # generate indices from current split ranges
        indices = []
        for beg_index, end_index in self.split_array[self.split_index]:
            indices.extend(list(range(beg_index, end_index)))
        indices = np.array(indices)

        # process extra indices which >= origin_num_samples
        if self.psudo_index is None:
            extra_indices = self.np_rng.randint(0, self.origin_num_samples, len(indices))
        else:
            extra_indices = np.ones_like(indices) * self.psudo_index
        extra_mask = indices >= self.origin_num_samples
        indices = indices * (1 - extra_mask) + extra_indices * extra_mask

        # shuffle operation (2): shuffle at index level
        if self.shuffle:
            self.np_rng.shuffle(indices)

        # subsample for each distributed rank
        offset = self.sampler_length * self.rank
        indices = indices[offset: offset + self.sampler_length]

        self.curr_indices = indices.tolist()

    def __iter__(self):
        if self.auto_step:
            self.step()
        elif self.first_step:
            self.step()
            self.first_step = False
        if not hasattr(self, 'curr_indices'):
            raise RuntimeError('Should call sampler.step() to generate self.indices before using sampler')
        return iter(self.curr_indices)

    def __len__(self):
        return self.sampler_length

    def state_dict(self):
        return {
            'np_rng_state': self.np_rng.get_state(),
            'split_array': self.split_array,
            'split_index': self.split_index,
        }

    def load_state_dict(self, state_dict):
        self.np_rng.set_state(state_dict['np_rng_state'])
        self.split_array = state_dict['split_array']
        self.split_index = state_dict['split_index']
