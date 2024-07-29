import torch
import numpy as np
import pickle

from model import GPTConfig, GPT

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

dataset = 'data/shakespeare_char'

train_ids = np.fromfile(f'{dataset}/train.bin', np.uint16)
valid_ids = np.fromfile(f'{dataset}/valid.bin', np.uint16)

train_ids = torch.tensor(train_ids, dtype=torch.long)
valid_ids = torch.tensor(valid_ids, dtype=torch.long)

meta = pickle.loads(open(f'{dataset}/meta.pkl', 'rb').read())
stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

torch.manual_seed(42)

config = GPTConfig(
    block_size=block_size,
    n_layer=n_layer,
    n_embed=n_embd,
    n_head=n_head,
    dropout=dropout,
    vocab_size=vocab_size
)

model = GPT(config).to(device)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_ids if split == 'train' else valid_ids
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))