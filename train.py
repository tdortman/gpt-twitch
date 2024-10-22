import glob
import os

import torch

from models import GPTLanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_all_files_to_string(directory):
    combined_string = ""
    for filepath in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                combined_string += file.read() + "\n"
    return combined_string


def prepare_data(text, device):
    lines = text.splitlines()
    lines = [line for line in lines if all(c.isascii() for c in line)]
    chars = sorted(list(set("".join(lines))))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(string):
        return [stoi[c] for c in string]

    def decode(tokens):
        return "".join([itos[i] for i in tokens])

    encoded_lines = [torch.tensor(encode(line), dtype=torch.long) for line in lines]
    data = torch.cat(encoded_lines)

    n = len(data)
    train_data = data[: int(n * 0.8)].to(device)
    val_data = data[int(n * 0.8) :].to(device)

    return train_data, val_data, encode, decode, vocab_size


text = read_all_files_to_string("data")


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(
    train_data,
    val_data,
    eval_interval,
    block_size,
    batch_size,
    model,
):
    out = {}
    mapping = {"train": train_data, "val": val_data}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(mapping[split], block_size, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(
    model,
    train_data,
    val_data,
    block_size,
    batch_size,
    learning_rate,
    max_epochs,
    eval_interval,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(max_epochs):
        if epoch % eval_interval == 0:
            losses = estimate_loss(
                train_data,
                val_data,
                eval_interval,
                block_size,
                batch_size,
                model,
            )

            print(
                f"Epoch {epoch}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}"
            )

        x_batch, y_batch = get_batch(train_data, block_size, batch_size)
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 64  # what is the maximum context length for predictions?
max_epochs = 5000
eval_interval = 50
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

train_data, val_data, encode, decode, vocab_size = prepare_data(text, device)

model = GPTLanguageModel(
    vocab_size,
    n_embd,
    block_size,
    n_layer,
    n_head,
    device,
    dropout,
).to(device)

train_model(
    model,
    train_data,
    val_data,
    block_size,
    batch_size,
    learning_rate,
    max_epochs,
    eval_interval,
)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
