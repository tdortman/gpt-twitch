import glob
import multiprocessing as mp
import os

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model import GPT
from trainer import Trainer


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read() + "\n"


def read_all_files_to_string(directory):
    filepaths = [
        filepath
        for filepath in glob.glob(os.path.join(directory, "**", "*"), recursive=True)
        if os.path.isfile(filepath)
    ]

    if not filepaths:
        raise ValueError("No files found in the input directory.")

    combined_string = ""
    with mp.Pool(min(len(filepaths), mp.cpu_count())) as executor:
        results = executor.map(read_file, filepaths)
        combined_string = "".join(results)

    return combined_string


def prepare_data(text: str):
    if not text:
        raise ValueError(
            "The input text is empty. Please check the file reading process."
        )

    lines = text.splitlines()
    lines = [line for line in lines if all(c.isascii() for c in line)]

    if not lines:
        raise ValueError("No valid ASCII lines found in the input text.")

    chars = sorted(list(set("".join(lines[:100_000]))))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(string):
        return [stoi[c] for c in string]

    def decode(tokens):
        return "".join([itos[i] for i in tokens])

    encoded_lines = [torch.tensor(encode(line), dtype=torch.long) for line in lines]

    if not encoded_lines:
        raise ValueError("No lines were encoded. Check the encoding process.")

    return torch.cat(encoded_lines), encode, decode, vocab_size


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


batch_size = 128
block_size = 128
max_epochs = 1000
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
save_interval = 10


directory = "data"
snapshot_path = f"{directory.replace('/', '_')}.pt"

text = read_all_files_to_string(directory)
train_data, encode, decode, vocab_size = prepare_data(text)


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def load_train_objs():
    train_dataset = TextDataset(train_data, block_size)

    model = GPT(
        vocab_size,
        n_embd,
        block_size,
        n_layer,
        n_head,
        dropout,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    return train_dataset, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(
    save_interval: int,
    total_epochs: int,
    batch_size: int,
    snapshot_path: str = "snapshot.pt",
):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_loader = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_loader, optimizer, save_interval, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    main(save_interval, max_epochs, batch_size)
