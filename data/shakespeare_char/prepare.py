"""
Prepare the tiny shakepeare dataset for char level language modeling.
Will save train.bin, valid.bin containing ids and meta.pkl containing
information about encoder, decoder and related information. 
"""
import requests
import numpy as np
import pickle
import os

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

# download data
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()

print(f"length of dataset in chars: {len(data)}")

chars = sorted(list(set(data)))

print(f"all unique chars: {''.join(chars)}")
vocab_size = len(chars)
print(f"vocab size is: {vocab_size}")

# bidirectional mapping from char to integer

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode: take a string and map it to a list of indices
encode = lambda s: [stoi[ch] for ch in s]
# decode: take a list of indices and map it to a string
decode = lambda idx: ''.join([itos[i] for i in idx])

# create train test split

train_ratio = 0.8
n = len(data)

train_data = data[:int(n * train_ratio)]
valid_data = data[int(n * train_ratio):]

train_ids = encode(train_data)
valid_ids = encode(valid_data)

train_ids = np.array(train_ids, np.uint16)
valid_ids = np.array(valid_ids, np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
valid_ids.tofile(os.path.join(os.path.dirname(__file__), 'valid.bin'))

print(f"training dataset contains {len(train_ids)} tokens")

print(f"valid dataset contains {len(valid_ids)} tokens")

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)