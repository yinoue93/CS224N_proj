import tensorflow as tf
import numpy as np
import os
import re
import json
import pickle

# Metadata + ~50 characters, then sliding window of (t+1)
# Feed dict should pass in an intial state (previous final state)
# Train on entire song & Batching for different songs?
# Train on individual window examples


def abc_filenames(datapath):
    return [os.path.join(datapath, f) for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]


def abc_batch(iterable, n=1):
    l = len(iterable)
    batches = []
    for ndx in range(0, l, n):
        if min(ndx + n, l) - ndx == n:
            batches.append(iterable[ndx:(ndx + n)])
    return batches


def read_abc_pickle(train_file):
    with open(train_file, 'r') as fd:
        return pickle.load(fd)


def compute_save_vocabulary(datapath):
    # Iterate through whole dataset directory
    filenames = abc_filenames(datapath)
    unique_characters = set([])
    for filename in filenames:
        characters = read_abc(filename)
        unique_characters.update(characters)

    vocabulary = dict(zip(unique_characters, range(len(unique_characters))))
    with open('vocabulary.json', 'w') as v:
        json.dump(vocabulary, v)


def load_vocabulary():
    with open('vocabulary.json', 'r') as v:
        return json.load(v)


def get_abs_files(datapath):
    filenames = abc_filenames(datapath)
    abc_songs = [] # Encoded as indicies
    for filename in filenames:
        characters = read_abc(filename)
        abc_songs.append(characters)
    return abc_songs


def abc_to_index(filename, vocabulary):
    characters = read_abc(filename)
    character_indicies = [vocabulary[char] for char in characters]
    return character_indicies


def read_abc(filename, exclude_title=True):
    with open(filename, 'r') as f:
        data = [line for line in f]
        if exclude_title:
            data = data[1:]
        # 4 metadata 'symbols'
        metadata = [re.split(":|\r\n", meta)[1].lower() for meta in data[:-1]]
        return metadata + list(re.split("\r\r\n",data[-1])[0])


def abc_producer(char_ids, batch_size):
    pass


def main(_):
    datapath = "sample_data"
    compute_save_vocabulary(datapath)
    vocabulary = load_vocabulary()
    print vocabulary
    filename = "sample_data/Zycanthos jig_0.abc"
    abc_indecies = abc_to_index(filename, vocabulary)


if __name__ == "__main__":
    tf.app.run()
