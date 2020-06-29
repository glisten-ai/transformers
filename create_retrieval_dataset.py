from collections import defaultdict
import csv
from tqdm import tqdm 
import re
import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

from google.cloud import storage
import argparse 

from transformers import BertTokenizer
import wandb

parser = argparse.ArgumentParser(description='Download and process raw data.')
parser.add_argument('--target_dir', dest='target_dir')
parser.add_argument('--datasets', dest='datasets', nargs='+')



def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def format_category(category):
    name = ' '.join(category.strip().lower().split())
    name = name.replace("&", "and").replace('-', ' ').replace("'", '').replace(',', '').replace('/', '')
    return name

def format_text(text):
    ft = re.sub(r'[^\x00-\x7f]',r'', text.lower())
    return ft

def relative_pos(og_indices, new_indices):
    idxs = []
    for n in new_indices:
        idxs.append(og_indices.index(n))
    return idxs

def tokenize_dataset(dataset, max_level, msl_title=128, msl_cat=10):

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    tokenization_by_category_level = []
    zero_indices = None
    for level in range(max_level):
        print('tokenizer level:', level)
        print('dataset size', len(dataset))
        titles_and_desc = []
        cats = []
        indices = []
        for idx, entry in tqdm(enumerate(dataset)):
            if not entry['category']:  # no categories
                continue
            if len(entry['category'])-1 < level:
                continue
            indices.append(idx)
            title = entry['title']
            desc = entry['description']
            category = entry['category'][level]
            titles_and_desc.append((title, desc))
            cats.append(category)
            
        if len(titles_and_desc) == 0:
            # Nothing with category in this level
            break
        if level > 0:
            relative_indices = relative_pos(zero_indices, indices)
            tokenized_titles_and_desc = {}
            for input_key, arr in tokenization_by_category_level[0][0].items():
                tokenized_titles_and_desc[input_key] = np.array(arr)[relative_indices]
        else:
            print(len(titles_and_desc), flush=True)
            zero_indices = indices
            start = time.time()
            tokenized_titles_and_desc = tokenizer.batch_encode_plus(
                titles_and_desc, max_length=msl_title, pad_to_max_length=True)
            print(time.time()-start)
        tokenized_cats = tokenizer.batch_encode_plus(cats, max_length=msl_cat, pad_to_max_length=True)
        tokenization_by_category_level.append((tokenized_titles_and_desc, tokenized_cats))
    return tokenization_by_category_level

def main():

    args = parser.parse_args()

    datadir = ""
    rootdir = args.target_dir
    filename = os.path.join(args.target_dir, "train.csv")
    download_blob("glisten", "datasets/doordash/train.csv", os.path.join(datadir, "train.csv"))
    print("Downloaded", filename)


    dataset = []
    with open(os.path.join(datadir, 'train.csv'), newline='',encoding = "utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset.append({
                'title': format_text(row['Title']),
                'description': format_text(row['Description']),
                'category': [format_category(row['Section']), format_category(row['Restaurant Category'])]
            })

    print("Finished parsing CSV", filename)
    dataset_name = "tokenized-doordash"

    file_pattern = "{dataset}_{split}_{level}_{msl_title}_{msl_cat}_{input_key}"
    lens = {}
    msl_title = 128
    msl_cat = 10
    
    
    tokenized_dataset = tokenize_dataset(dataset,
                                max_level=7, msl_title=msl_title, msl_cat=msl_cat)
        
    print(dataset_name, flush=True)
    print("got dataset")
    for level in range(len(tokenized_dataset)):
        print('level:', level, flush=True)
        titles, cats = tokenized_dataset[level]
        split_to_arr = {'title-and-desc': titles, 'category': cats}
        for split in ['title-and-desc', 'category']:
            tokenized_output = split_to_arr[split]
            for input_key, arr in tokenized_output.items():
                filename = os.path.join(rootdir,
                                        file_pattern.format(
                                            dataset=dataset_name,
                                            split=split,
                                            level=level,
                                            msl_title=msl_title,
                                            msl_cat=msl_cat,
                                            input_key=input_key))
                arr = np.array(arr)
                fp = np.memmap(filename, dtype='int64', mode='w+',
                                shape=arr.shape)
                lens[(dataset_name, split, level)] = arr.shape
                fp[:] = arr[:]
                del fp

    f = os.path.join(rootdir, 'lens_{}_{}.pkl'.format(msl_title, msl_cat))
    with open(f, 'wb') as file:
        # Keep lens which is needed for loading a memmap
        pickle.dump(lens, file)

    # Store dataset on W&B
    artifact = wandb.Artifact(dataset_name, type='dataset')
    artifact.add_dir(rootdir)
    run = wandb.init(project="transformers", job_type='training')
    print(run.log_artifact(artifact))
   
    

if __name__ == "__main__":
    main()
