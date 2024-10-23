import tensorflow as tf
import glob
import os
from models_tf import GPTLanguageModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_all_files_to_string(directory):
    combined_string = ""
    for filepath in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
        logging.info(f"Reading file {filepath}")
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                combined_string += file.read() + "\n"
    return combined_string


def create_data_generator(data, block_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)

    logging.info(
        f"Creating data generator with block size {block_size} and batch size {batch_size}"
    )
    logging.info(f"Dataset size: {len(data)}")

    windows = dataset.window(block_size + 1, shift=1, drop_remainder=True)
    windows = windows.flat_map(lambda x: x.batch(block_size + 1))

    def split_input_target(sequence):
        return sequence[:-1], sequence[1:]

    dataset = windows.map(split_input_target)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_data(text):
    lines = text.splitlines()
    lines = [line for line in lines if all(c.isascii() for c in line)]
    chars = sorted(list(set("".join(lines))))
    vocab_size = len(chars)

    logging.info(f"Vocabulary size: {vocab_size}")

    # Create character to integer mappings
    char_to_idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(chars),
            values=tf.constant(list(range(len(chars))), dtype=tf.int64),
        ),
        default_value=-1,
    )

    idx_to_char = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(range(len(chars))), dtype=tf.int64),
            values=tf.constant(chars),
        ),
        default_value="",
    )

    def encode(text_tensor):
        return char_to_idx.lookup(tf.strings.unicode_split(text_tensor, "UTF-8"))

    def decode(indices):
        return tf.strings.reduce_join(idx_to_char.lookup(indices))

    dataset = tf.data.Dataset.from_tensor_slices(lines)
    encoded_data = dataset.map(encode)
    all_data = tf.concat(list(encoded_data), axis=0)

    n = tf.shape(all_data)[0]
    split_index = tf.cast(tf.cast(n, tf.float32) * 0.8, tf.int32)
    train_data = all_data[:split_index]
    val_data = all_data[split_index:]

    logging.info(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

    return train_data, val_data, encode, decode, vocab_size


# Training configuration
batch_size = 32
block_size = 512
max_epochs = 1000
eval_interval = 250
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

text = read_all_files_to_string("data")
train_data, val_data, encode, decode, vocab_size = prepare_data(text)

train_dataset = create_data_generator(train_data, block_size, batch_size)
val_dataset = create_data_generator(val_data, block_size, batch_size)

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
)


optimizer = tf.keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.95,
    epsilon=1e-8,
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy"],
)

# Load existing weights or train
if os.getenv("LOAD_MODEL") and os.path.exists("model_tf.keras"):
    logging.info("Loading existing model...")
    model.load_weights("model_tf.keras")
else:
    logging.info("Training new model...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "model_tf.keras", save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        ),
    ]

    steps_per_epoch = len(train_data) // batch_size
    validation_steps = len(val_data) // batch_size

    model.fit(
        train_dataset,
        epochs=max_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )

if os.getenv("SAVE_MODEL"):
    model.save_weights("model_tf.keras")

context = tf.zeros((1, 1), dtype=tf.int64)
generated = model.generate(context, max_new_tokens=100, temperature=0.7, top_k=50)
print(decode(generated[0]))
