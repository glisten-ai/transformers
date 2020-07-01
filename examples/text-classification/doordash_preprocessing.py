import torch
import itertools
from collections import Counter, OrderedDict, defaultdict
from torch.utils.data import DataLoader
import random
import numpy as np
import pickle
import os


class IterableTitles(torch.utils.data.IterableDataset):
    def __init__(self, cuisine, root_dir, max_seq_length, len_dict):
        super(IterableTitles).__init__()
        self.cuisine = cuisine
        self.max_seq_length = max_seq_length
        self.datafile_pattern = os.path.join(
            root_dir, 
            'titles_{}_{}_{{}}'.format(cuisine, max_seq_length))
        self.shape = (len_dict[cuisine], max_seq_length)
        self.length = len_dict[cuisine]

    def __len__(self):
        return self.length

    def __iter__(self):
        input_key_to_memmap = {}
        for input_key in ['input_ids',
                          'attention_mask',
                         'token_type_ids']:
            datafile = self.datafile_pattern.format(input_key)
            fp = np.memmap(
                datafile, dtype='int64', mode='r+',
                shape=self.shape)
            input_key_to_memmap[input_key] = fp
        while True:
            i = random.choice(range(self.shape[0]))
        #for i in itertools.cycle(range(self.shape[0])):
            yield (
                input_key_to_memmap['input_ids'][i],
                input_key_to_memmap['token_type_ids'][i],
                input_key_to_memmap['attention_mask'][i])


class MultiStreamDataLoader:
    def __init__(self, targets_to_categories, batch_size, rootdir):
        self.targets = list(sorted(targets_to_categories.keys()))
        self.targets_to_categories = targets_to_categories
        
        self.categories_to_target = OrderedDict()
        self.targets_to_categories_weights= {}
        self.categories_to_dataset = {}
        self.categories_to_dataset_iter = {}
        with open(os.path.join(rootdir, 'lens.pkl'), 'rb') as f:
            self.categories_to_len = pickle.load(f)
        
        for t, list_of_c in targets_to_categories.items():
            for c in list_of_c:
                self.categories_to_target[c] = t
                dataset = IterableTitles(c, rootdir, 64, self.categories_to_len)
                self.categories_to_dataset[c] = dataset
                self.categories_to_dataset_iter[c] = iter(DataLoader(dataset, batch_size=None))

        for t, c_list in self.targets_to_categories.items():
            self.targets_to_categories_weights[t] = [self.categories_to_len[c] for c in c_list]

        self.batch_size = batch_size
        self.total_samples = sum(self.categories_to_len.values())


    def __len__(self):
        return self.total_samples//self.batch_size
    

    def __iter__(self):
        while True:
            buffer = []
            labels = []
            target_choices = random.choices(self.targets, k=self.batch_size)
            category_choices = []
            for t in target_choices:
                category_choices.append(random.choices(
                    self.targets_to_categories[t],
                    weights=self.targets_to_categories_weights[t],
                    k=1)[0])

            category_counter = Counter(category_choices)
            category_labels = []
            for category, num_sample in category_counter.items():
                category_labels.extend([category for _ in range(num_sample)])
                l_num = self.targets.index(self.categories_to_target[category])
                labels.extend([l_num for _ in range(num_sample)])
                buffer.extend(
                    [next(self.categories_to_dataset_iter[category]) for _ in range(num_sample)]
                )
            yield (torch.stack([b[0] for b in buffer]),
                   torch.stack([b[1] for b in buffer]),
                   torch.stack([b[2] for b in buffer]),
                   torch.tensor(labels),
                   # category_labels
                  )
