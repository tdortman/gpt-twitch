#!/usr/bin/env python
import torch
import typer
from torch.distributed import destroy_process_group
from torch.optim import AdamW
from typer import Option

from model import GPT
from trainer import Trainer
from utils import (
    TextDataset,
    ddp_setup,
    prepare_data,
    prepare_dataloader,
    read_all_files_to_string,
)

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


app = typer.Typer()


@app.command(help="Generate text from a trained model")
def generate(
    vocab_limit: int = Option(25000, help="Maximum number of unique words to include in the vocabulary"),
    directory: str = Option("data", help="Directory containing the text files to train on"),
    n_embd: int = Option(384, help="Embedding dimension to use for the model"),
    n_head: int = Option(6, help="Number of attention heads to use for the model"),
    n_layer: int = Option(6, help="Number of blocks to use for the model"),
    dropout: float = Option(0.2, help="Dropout rate to use for the model"),
    num_groups: int = Option(3, help="Number of groups to use grouped attention"),
    block_size: int = Option(128, help="Block size to use for the model"),
    num_tokens: int = Option(100, help="Number of tokens to generate"),
    reset_cache: bool = Option(False, help="Reset the cache before generating"),
    start_str: str = Option(eval_str, help="String to start generation with"),
):
    snapshot_path = f"snapshot_{directory.replace('/', '_')}.pt"
    cache_path = f"cache_{directory.replace('/', '_')}.pkl"
    text = read_all_files_to_string(directory)
    _, encode, decode, vocab_size = prepare_data(text, vocab_limit, cache_path, reset_cache)

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
    context = torch.tensor([encode(start_str)], dtype=torch.long).cuda()
    print(decode(model.generate(context, max_new_tokens=num_tokens)[0].tolist()))


@app.command(help="Train a new model")
def train(
    vocab_limit: int = Option(25000, help="Maximum number of unique words to include in the vocabulary"),
    save_interval: int = Option(10, help="Number of epochs between saves"),
    batch_size: int = Option(128, help="Batch size to use for training"),
    directory: str = Option("data", help="Directory containing the text files to train on"),
    learning_rate: float = Option(0.0003, help="Learning rate to use for training"),
    n_embd: int = Option(384, help="Embedding dimension to use for the model"),
    n_head: int = Option(6, help="Number of attention heads to use for the model"),
    n_layer: int = Option(6, help="Number of blocks to use for the model"),
    dropout: float = Option(0.2, help="Dropout rate to use for the model"),
    num_groups: int = Option(3, help="Number of groups to use for grouped attention"),
    block_size: int = Option(128, help="Block size to use for the model"),
    total_epochs: int = Option(2000, help="Maximum number of epochs to train for"),
    reset_cache: bool = Option(False, help="Reset the cache before preparing data"),
):
    ddp_setup()
    snapshot_path = f"snapshot_{directory.replace('/', '_')}.pt"
    cache_path = f"cache_{directory.replace('/', '_')}.pkl"

    text = read_all_files_to_string(directory)
    train_data, _, _, vocab_size = prepare_data(text, vocab_limit, cache_path, reset_cache)

    dataset = TextDataset(train_data, block_size)

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

    train_loader = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(
        model,
        train_loader,
        optimizer,
        save_interval,
        snapshot_path,
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    app()
