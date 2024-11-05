import glob
import multiprocessing as mp
import os
import pickle
from collections import Counter
from typing import Callable, Tuple

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import GPT
from trainer import Trainer


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read() + "\n"


def read_pickle_in_chunks(file_path: str, chunk_size=1024):
    file_size = os.path.getsize(file_path)
    progress_bar = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc="Reading pickle file",
    )

    buffer = bytearray()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer.extend(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()
    data = pickle.loads(buffer)
    return data


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


def enc(sentence: str, stoi: dict[str, int]) -> list:
    tokens = [stoi.get(word, stoi["<unk>"]) for word in sentence.split()]
    tokens.append(stoi["\n"])
    return tokens


def dec(tokens: list, itos: dict[int, str]) -> str:
    words = [itos[token] for token in tokens]
    return " ".join(words).replace(" \n", "\n")


def prepare_data(
    text: str,
    vocab_limit: int,
    cache_path: str,
) -> Tuple[torch.Tensor, Callable[[str], list], Callable[[list], str], int]:
    try:
        data, stoi, itos, vocab_size = read_pickle_in_chunks(cache_path)
        print("Loaded data from cache.")
        return (
            data,
            lambda x: enc(x, stoi),
            lambda x: dec(x, itos),
            vocab_size,
        )
    except (FileNotFoundError, EOFError):
        print("No cache found, preparing data from scratch.")

    if not text:
        raise ValueError(
            "The input text is empty. Please check the file reading process."
        )

    lines = text.splitlines()
    lines = [line for line in lines if all(c.isascii() for c in line)]

    if not lines:
        raise ValueError("No valid ASCII lines found in the input text.")

    word_counts = Counter(word for line in lines for word in line.split())
    most_common_words = [word for word, _ in word_counts.most_common(vocab_limit)]
    words = most_common_words + ["<unk>", "\n"]

    vocab_size = len(words)
    stoi = {word: i for i, word in enumerate(words)}
    itos = {i: word for i, word in enumerate(words)}

    encoded_lines = [
        torch.tensor(enc(line, stoi), dtype=torch.long)
        for line in lines
        if enc(line, stoi)
    ]

    if not encoded_lines:
        raise ValueError("No lines were encoded. Check the encoding process.")

    data = torch.cat(encoded_lines)

    with open(cache_path, "wb") as f:
        pickle.dump((data, stoi, itos, vocab_size), f)
        print("Data cached successfully.")

    return data, lambda x: enc(x, stoi), lambda x: dec(x, itos), vocab_size


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
max_epochs = 2000
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
save_interval = 10
num_groups = 3
vocab_limit = 25_000

eval_str = r"""
    Scene 1
=======
[Enter Theseus, Hippolyta, and Philostrate, with others.]


THESEUS
Now, fair Hippolyta, our nuptial hour
Draws on apace. Four happy days bring in
Another moon. But, O, methinks how slow
This old moon wanes! She lingers my desires
Like to a stepdame or a dowager
Long withering out a young man's revenue.

HIPPOLYTA
Four days will quickly steep themselves in night;
Four nights will quickly dream away the time;
And then the moon, like to a silver bow
New-bent in heaven, shall behold the night
Of our solemnities.

THESEUS  Go, Philostrate,
Stir up the Athenian youth """


directory = "data"
snapshot_path = f"snapshot_{directory.replace('/', '_')}.pt"
cache_path = f"cache_{directory.replace('/', '_')}.pkl"

text = read_all_files_to_string(directory)
train_data, encode, decode, vocab_size = prepare_data(text, vocab_limit, cache_path)

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
        num_groups,
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
    if not os.getenv("EVAL"):
        ddp_setup()
        dataset, model, optimizer = load_train_objs()
        train_loader = prepare_dataloader(dataset, batch_size)
        trainer = Trainer(model, train_loader, optimizer, save_interval, snapshot_path)
        trainer.train(total_epochs)
        destroy_process_group()
    else:
        model = GPT(
            vocab_size,
            n_embd,
            block_size,
            n_layer,
            n_head,
            dropout,
            num_groups,
        ).cuda()
        model.load_state_dict(torch.load(snapshot_path)["MODEL_STATE"])
        context = torch.tensor([encode(eval_str)], dtype=torch.long).cuda()
        print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))


if __name__ == "__main__":
    main(save_interval, max_epochs, batch_size, snapshot_path)
