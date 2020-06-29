import torch
import itertools
from collections import Counter, OrderedDict, defaultdict
from torch.utils.data import DataLoader
import random
import numpy as np
import pickle
import os

class IterableTitles(torch.utils.data.IterableDataset):
    def __init__(self, root_dir, dataset, level, msl_title, msl_cat):
        super(IterableTitles).__init__()
        file_pattern = "{dataset}_{{split}}_{level}_{msl_title}_{msl_cat}_{{input_key}}"
        self.datafile_pattern = os.path.join(
            root_dir,
            file_pattern.format(dataset=dataset, level=level, msl_title=msl_title, msl_cat=msl_cat))
        len_filename = 'lens_{}_{}.pkl'.format(msl_title, msl_cat)
        with open(os.path.join(root_dir, len_filename), 'rb') as f:
            len_dict = pickle.load(f)
        self.len_dict = len_dict
        assert self.len_dict[(dataset, 'title-and-desc', level)][0] == self.len_dict[(dataset, 'category', level)][0]
        self.length = self.len_dict[(dataset, 'title-and-desc', level)][0]
        self.dataset = dataset
        self.level = level

    def __len__(self):
        return self.length

    def __iter__(self):
        split_input_key_to_memmap = {}
        for split in ['title-and-desc', 'category']:
            for input_key in ['input_ids',
                              'attention_mask',
                              'token_type_ids']:
                datafile = self.datafile_pattern.format(split=split, input_key=input_key)
                fp = np.memmap(
                    datafile, dtype='int64', mode='r+',
                    shape=self.len_dict[(self.dataset, split, self.level)])
                split_input_key_to_memmap[(split, input_key)] = fp
        while True:
            i = random.choice(range(self.length))
            yield (
                split_input_key_to_memmap[('title-and-desc', 'input_ids')][i],
                split_input_key_to_memmap[('title-and-desc', 'token_type_ids')][i],
                split_input_key_to_memmap[('title-and-desc', 'attention_mask')][i],
                split_input_key_to_memmap[('category', 'input_ids')][i],
                split_input_key_to_memmap[('category', 'token_type_ids')][i],
                split_input_key_to_memmap[('category', 'attention_mask')][i],
            )


class MultiStreamDataLoader:

    def __init__(self, 
                root_dir, msl_title, msl_cat, batch_size,
                datasets):
        len_filename = 'lens_{}_{}.pkl'.format(msl_title, msl_cat)
        with open(os.path.join(root_dir, len_filename), 'rb') as f:
            len_dict = pickle.load(f)
        print(len_dict)
        self.len_dict = dict([(k, v[0]) for k,v in len_dict.items()])  # in len_dict shape is stored
        self.batch_size = batch_size
        self.total_samples = sum(self.len_dict.values())//2  # title and category => duplicate
        self.dataset_lvl_to_iter = {}
        self.random_weights = {}
        for dataset in datasets:
            for key in self.len_dict:
                if key[0] == dataset and key[1] == 'title-and-desc':
                    level = key[2]
                    dataset_iter = IterableTitles(root_dir, dataset, level, msl_title, msl_cat)
                    self.dataset_lvl_to_iter[(dataset, level)] = iter(DataLoader(dataset_iter, batch_size=None))
                    self.random_weights[(dataset, level)] = self.len_dict[key]

    def __len__(self):
        return self.total_samples//self.batch_size
    

    def __iter__(self):
        # dataset_keys = list(self.len_dict.keys())
        print(self.dataset_lvl_to_iter.keys())
        while True:
            buffer = []
            labels = []
            keys = list(self.dataset_lvl_to_iter.keys())
            weights = [self.random_weights[k] for k in keys]

            key_choices = random.choices(
                keys,
                weights=weights,
                k=self.batch_size,
            )
            # key_choices = random.choices(dataset_keys, weights=list(self.len_dict.values()), k=self.batch_size)
            for key in key_choices:
                buffer.extend(
                    [next(self.dataset_lvl_to_iter[key])]
                )
            yield (torch.stack([b[0] for b in buffer]),
                   torch.stack([b[1] for b in buffer]),
                   torch.stack([b[2] for b in buffer]),
                   torch.stack([b[3] for b in buffer]),
                   torch.stack([b[4] for b in buffer]),
                   torch.stack([b[5] for b in buffer]),
                  )
