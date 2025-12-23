import marimo

__generated_with = "0.18.4"
app = marimo.App()

with app.setup:
    import copy
    import os
    import time
    import zipfile
    from importlib.metadata import version
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    import tiktoken
    import torch
    # previous notebooks
    from ch04 import GPTModel, generate_text_simple
    from ch05 import (load_weights_into_gpt, text_to_token_ids,
                      token_ids_to_text)
    # additional utility
    from gpt_download import download_and_load_gpt2
    from torch.utils.data import DataLoader, Dataset

    # detect available device (CPU or GPU)
    # skip the condition for MPS devices for simplicity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


@app.cell
def _():
    mo.md(r"""
    # Chapter 6: Finetuning for Text Classification
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Show library versions
    """)
    return


@app.cell
def _():
    pkgs = ["matplotlib", 
            "numpy", 
            "tiktoken", 
            "torch",
            "tensorflow" # For OpenAI's pretrained weights
           ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    return


@app.cell
def _():
    mo.md(r"""
    ## 6.2 Preparing the dataset
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Set paths to download dataset for fine-tuning
    """)
    return


@app.cell
def _():
    dataset_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / "data" / "ch06"
    os.makedirs(data_root, exist_ok=True)

    zip_path = data_root / "sms_spam_collection.zip"
    extracted_path = data_root / "sms_spam_collection"
    data_file_path = extracted_path / "SMSSpamCollection.tsv"

    print(f"Downloading dataset to {zip_path}...")
    return (
        data_file_path,
        data_root,
        dataset_url,
        extracted_path,
        project_root,
        zip_path,
    )


@app.cell
def _():
    mo.md(r"""
    Define a function to download, extract, and rename
    """)
    return


@app.function
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"File {data_file_path} already exists. Skipping download.")
        return

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    os.remove(zip_path) # cleanup zip file

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


@app.cell
def _():
    mo.md(r"""
    Prepare dataset by using the above function
    """)
    return


@app.cell
def _(data_file_path, dataset_url, extracted_path, zip_path):
    try:
        download_and_unzip_spam_data(dataset_url, zip_path, extracted_path, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        _backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(_backup_url, zip_path, extracted_path, data_file_path)
    return


@app.cell
def _():
    mo.md(r"""
    This dataset has Tab separater without header. Each rows are the pair of spam/ham pair and corresponding text (message). The ham means not spam.
    """)
    return


@app.cell
def _(data_file_path):
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    df
    return (df,)


@app.cell
def _():
    mo.md(r"""
    Show the statistics. Spam messages are anomalous and its number is much smallear than ham messages.
    """)
    return


@app.cell
def _(df):
    print(df["Label"].value_counts())
    return


@app.cell
def _():
    mo.md(r"""
    Undersample to balance the number of spam and ham
    """)
    return


@app.function
def create_balanced_dataset(df):
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


@app.cell
def _():
    mo.md(r"""
    Execute and check the result
    """)
    return


@app.cell
def _(df):
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())
    return (balanced_df,)


@app.cell
def _():
    mo.md(r"""
    Mapping `str` to `int` for numerical operations
    """)
    return


@app.cell
def _(balanced_df):
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})    
    balanced_df
    return


@app.cell
def _():
    mo.md(r"""
    Create splits for training, validation, and testing.
    For [pandas.DataFrame.sample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html), the `frac=1` means 100% sampling, and the `.reset_index(drop=True)` means the operation delete the original index (row number).
    """)
    return


@app.function
def random_split(df, train_frac, validation_frac):
    """
    Randomly split a DataFrame into training, validation, and test sets.

    :param df: The DataFrame to split.
    :param train_frac: Fraction (0-1) of data to use for training.
    :param validation_frac: Fraction (0-1) of data to use for validation.
    :return: A tuple of (train_df, validation_df, test_df).
    """
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


@app.cell
def _():
    mo.md(r"""
    Create the splits with the ratio for

    $$
    \text{train} : \text{valid} : \text{test} = 0.7 : 0.1 : 0.2
    $$

    and save these dataframes as CSV
    """)
    return


@app.cell
def _(balanced_df, data_root):
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # Test size is implied to be 0.2 as the remainder

    train_path = data_root / "train.csv"
    valid_path = data_root / "validation.csv"
    test_path = data_root / "test.csv"
    train_df.to_csv(train_path, index=None)
    validation_df.to_csv(valid_path, index=None)
    test_df.to_csv(test_path, index=None)
    return test_path, train_path, valid_path


@app.cell
def _():
    mo.md(r"""
    ## 6.3 Creating data loaders
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Get tiktoken encoder and the token ID for `<|endoftext|>` to use it as padding token ID
    """)
    return


@app.cell
def _():
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    return (tokenizer,)


@app.cell
def _():
    mo.md(r"""
    Define [Map-style datasets](https://docs.pytorch.org/docs/stable/data.html#map-style-datasets) for use with a DataLoader.
    Token sequences are padded to a uniform length, either specified explicitly or set to the maximum sequence length in the dataset.
    """)
    return


@app.class_definition
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # set sequence length to pad
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        # torch.long == torch.int64
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


@app.cell
def _():
    mo.md(r"""
    Construct a dataset for training. We can set the `max_length` to our context length to ensure it.
    After the creation, check the `max_length` and the shape of each data.
    """)
    return


@app.cell
def _(tokenizer, train_path):
    train_dataset = SpamDataset(
        csv_file=train_path,
        max_length=None,  # use the max length of actual sequences
        tokenizer=tokenizer
    )

    print(f"{train_dataset.max_length=}")
    # a tuple of seqeuence and its label
    print(f"{train_dataset[0]=}")  
    # the length of sequence is identical with the max_length
    print(f"{train_dataset[0][0].shape=}")
    return (train_dataset,)


@app.cell
def _():
    mo.md(r"""
    Create validation and test datasets also. We share the `max_length` for each construction to ensure the same condition.
    """)
    return


@app.cell
def _(test_path, tokenizer, train_dataset, valid_path):
    val_dataset = SpamDataset(
        csv_file=valid_path,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=test_path,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    return test_dataset, val_dataset


@app.cell
def _():
    mo.md(r"""
    Construct [daloaders](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) by using the datasets
    """)
    return


@app.cell
def _(test_dataset, train_dataset, val_dataset):
    num_workers = 0  # ensure compatibility (for all CPUs)
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # randomize
        num_workers=num_workers,
        drop_last=True,  # ignore incomplete batch
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return test_loader, train_loader, val_loader


@app.cell
def _():
    mo.md(r"""
    Iterate entirely for training dataloader to get the final batch and check the shape. We can see surely the final batch is complete shape.
    """)
    return


@app.cell
def _(train_loader):
    # iteration for doing nothing
    for input_batch, target_batch in train_loader:
        pass

    # we expect 8 pairs in a batch even it is the final batch
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)
    return


@app.cell
def _():
    mo.md(r"""
    Check the number of batches for each dataloaders
    """)
    return


@app.cell
def _(test_loader, train_loader, val_loader):
    _num_train = len(train_loader.dataset)
    _num_val = len(val_loader.dataset)
    _num_test = len(test_loader.dataset)
    print(f"{_num_train} training batches")
    print(f"{_num_val} validation batches")
    print(f"{_num_test} test batches")

    # check the ratio is as expected or not
    _num_all = _num_train + _num_val + _num_test
    _train_ratio = _num_train / _num_all
    _val_ratio = _num_val / _num_all
    _test_ratio = _num_test / _num_all
    print(f"Train:Val:Test = {_train_ratio:.2f}:{_val_ratio:.2f}:{_test_ratio:.2f}")
    return


@app.cell
def _():
    mo.md(r"""
    ## 6.4 Initializing a model with pretrained weights
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Use the same configuration of the model with the pretraining
    """)
    return


@app.cell
def _(train_dataset):
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )
    return BASE_CONFIG, CHOOSE_MODEL


@app.cell
def _(BASE_CONFIG, CHOOSE_MODEL, project_root):
    # just parse the `str` inside the brace
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    models_dir = project_root / "models" / "gpt2"
    os.makedirs(models_dir, exist_ok=True)

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

    pretrained_model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(pretrained_model, params)

    # This will be finetuned
    model = copy.deepcopy(pretrained_model)
    return model, pretrained_model


@app.cell
def _(model):
    model.eval()
    return


@app.cell
def _():
    mo.md(r"""
    Check the model is pretrained and its inference is proper by using a simple sequence
    """)
    return


@app.cell
def _(BASE_CONFIG, pretrained_model, tokenizer):
    _text_1 = "Every effort moves you"

    _token_ids = generate_text_simple(
        model=pretrained_model,
        idx=text_to_token_ids(_text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )
    print(f"{_token_ids=}")

    print("token_ids_to_text(_token_ids, tokenizer)=")
    print(f"{token_ids_to_text(_token_ids, tokenizer)}")
    return


@app.cell
def _():
    mo.md(r"""
    See the behavior before the fine-tuning. The answer does not follow the initial instructions, and answer the question without yes or no.
    """)
    return


@app.cell
def _(BASE_CONFIG, pretrained_model, tokenizer):
    _text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    _token_ids = generate_text_simple(
        model=pretrained_model,
        idx=text_to_token_ids(_text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(_token_ids, tokenizer))
    return


@app.cell
def _():
    mo.md(r"""
    ## 6.5 Adding a classification head
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Freeze all layers initially to reuse the most of weights
    """)
    return


@app.cell
def _(model):
    for _param in model.parameters():
        _param.requires_grad = False
    return


@app.cell
def _():
    mo.md(r"""
    Define new linear layer to classify spam or ham, and replace the current last layer to output expected tokens with the new linear layer
    """)
    return


@app.cell
def _(BASE_CONFIG, model):
    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    return


@app.cell
def _():
    mo.md(r"""
    Allow gradient calculations for the transformer blocks and the last `LayerNorm` module to train
    """)
    return


@app.cell
def _(model):
    for _param in model.trf_blocks[-1].parameters():
        _param.requires_grad = True

    for _param in model.final_norm.parameters():
        _param.requires_grad = True
    return


@app.cell
def _():
    mo.md(r"""
    Try inference with this architecture before fine-tuning by using following inputs
    """)
    return


@app.cell
def _(tokenizer):
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)
    return (inputs,)


@app.cell
def _():
    mo.md(r"""
    The output size is $(\text{batch size}, \text{num tokens}, \text{num classes})$.
    """)
    return


@app.cell
def _(inputs, model):
    model.to("cpu")  # ensure device matching
    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)
    return (outputs,)


@app.cell
def _():
    mo.md(r"""
    We use the classification result for the last token only because this is the only token can have causal correlation with all tokens based on causal attention mask.
    """)
    return


@app.cell
def _(outputs):
    print("Last output token:", outputs[:, -1, :])
    return


@app.cell
def _():
    mo.md(r"""
    ## 6.6 Calculating the classification loss and accuracy
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Map the output to class label
    """)
    return


@app.cell
def _(outputs):
    _probas = torch.softmax(outputs[:, -1, :], dim=-1)
    _label = torch.argmax(_probas)
    print("Class label:", _label.item())
    return


@app.cell
def _():
    mo.md(r"""
    We can get the same result by just taking `argmax` only because `softmax` is monotonic
    """)
    return


@app.cell
def _(outputs):
    _logits = outputs[:, -1, :]
    _label = torch.argmax(_logits)
    print("Class label:", _label.item())
    return


@app.cell
def _():
    mo.md(r"""
    Define the accuracy metric to evaluate
    """)
    return


@app.function
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                # Logits of last output token
                # (num_samples, num_tokens, num_logits) ->  (num_samples, num_logits)
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples


@app.cell
def _():
    mo.md(r"""
    Compute initial accuracies without training
    """)
    return


@app.cell
def _(model, test_loader, train_loader, val_loader):
    # no assignment model = model.to(device) necessary for nn.Module classes
    model.to(device)
    # For reproducibility due to the shuffling in the training data loader
    torch.manual_seed(123) 

    _train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    _val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    _test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {_train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {_val_accuracy*100:.2f}%")
    print(f"Test accuracy: {_test_accuracy*100:.2f}%")
    return


@app.cell
def _():
    mo.md(r"""
    Use cross entropy loss to train instead of the accuracy because of these differentiability. See [this](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) for the definition.
    """)
    return


@app.function
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


@app.cell
def _():
    mo.md(r"""
    Compute average loss by using each dataloader
    """)
    return


@app.function
# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@app.cell
def _():
    mo.md(r"""
    Compute initial losses without training
    """)
    return


@app.cell
def _(model, test_loader, train_loader, val_loader):
    model.to(device)
    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        _train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        _val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        _test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {_train_loss:.3f}")
    print(f"Validation loss: {_val_loss:.3f}")
    print(f"Test loss: {_test_loss:.3f}")
    return


@app.cell
def _():
    mo.md(r"""
    ## 6.7 Finetuning the model on supervised data
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Define fine-tuning trainer that calculates accuracies in the end of each epochs
    """)
    return


@app.function
# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """
    Train the model using the provided data loaders and optimizer.

    :param model: The model to train.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param optimizer: Optimizer for updating model weights.
    :param device: Device to run the training on (e.g., 'cpu' or 'cuda').
    :param num_epochs: Number of epochs to train.
    :param eval_freq: Frequency (in steps) to evaluate the model.
    :param eval_iter: Number of batches to use for evaluation.
    :return: Tuple of lists containing training losses, validation losses,
             training accuracies, validation accuracies, and total examples seen.
    """
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


@app.cell
def _():
    mo.md(r"""
    Define fine-tuning evaluater
    """)
    return


@app.function
# Same as chapter 5
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


@app.cell
def _():
    mo.md(r"""
    Execute the training in 5 epochs, and the accuracy will be improved more than 90%. It takes about 3 minuites for RTX3070.
    """)
    return


@app.cell
def _(model, train_loader, val_loader):
    _start_time = time.time()

    model.to(device)
    torch.manual_seed(123)
    _optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, _optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - _start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    return (
        examples_seen,
        num_epochs,
        train_accs,
        train_losses,
        val_accs,
        val_losses,
    )


@app.cell
def _():
    mo.md(r"""
    Plot the losses and verify it works well from the fact that losses decrease rapidlly. Especially, validation losses looks like training losses and it shows the traning avoids overfitting. These facts justify 5 epochs are enough for this fine-tuning.
    """)
    return


@app.cell
def _(data_root):
    def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()

        # Create a second x-axis for examples seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Examples seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.savefig(data_root / f"{label}-plot.pdf")
        plt.show()
    return (plot_values,)


@app.cell
def _(examples_seen, num_epochs, plot_values, train_losses, val_losses):
    _epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    _examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(_epochs_tensor, _examples_seen_tensor, train_losses, val_losses)
    return


@app.cell
def _():
    mo.md(r"""
    Plot the accuracies also. We used 5 bathes to evaluate these (see the `eval_iter` argument).
    """)
    return


@app.cell
def _(examples_seen, num_epochs, plot_values, train_accs, val_accs):
    _epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    _examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(_epochs_tensor, _examples_seen_tensor, train_accs, val_accs, label="accuracy")
    return


@app.cell
def _():
    mo.md(r"""
    The results for the final model and entire dataloader show the accuracies are higher than 90%. The fact that the test accuracy is smaller than others shows small overfitting. This differences may be removed by hyper parameter tuning for `drop_rate`, `weight_decay`, and so on.
    """)
    return


@app.cell
def _(model, test_loader, train_loader, val_loader):
    _train_accuracy = calc_accuracy_loader(train_loader, model, device)
    _val_accuracy = calc_accuracy_loader(val_loader, model, device)
    _test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {_train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {_val_accuracy*100:.2f}%")
    print(f"Test accuracy: {_test_accuracy*100:.2f}%")
    return


@app.cell
def _():
    mo.md(r"""
    ## 6.8 Using the LLM as a spam classifier
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Define a function to do preprocessing, inference, and postprocessing to answer whether the input text is spam or ham
    """)
    return


@app.function
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    Classify a given text as "spam" or "not spam" using the provided model and tokenizer.

    :param text: The input text to classify.
    :param model: The trained classification model.
    :param tokenizer: The tokenizer used to encode the text.
    :param device: The device to run the model on (e.g., "cpu" or "cuda").
    :param max_length: The maximum length for the input sequence. If None, use the model's context length.
    :param pad_token_id: The token ID used for padding sequences.
    :return: "spam" if the text is classified as spam, otherwise "not spam".
    """
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )    
    # Alternatively, a more robust version is the following one, which handles the max_length=None case better
    # max_len = min(max_length,supported_context_length) if max_length else supported_context_length
    # input_ids = input_ids[:max_len]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


@app.cell
def _():
    mo.md(r"""
    This is an example for spam
    """)
    return


@app.cell
def _(model, tokenizer, train_dataset):
    _text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        _text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))
    return


@app.cell
def _():
    mo.md(r"""
    This is an example for ham
    """)
    return


@app.cell
def _(model, tokenizer, train_dataset):
    _text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        _text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))
    return


@app.cell
def _():
    mo.md(r"""
    We can save the result for the fine-tuning by this
    """)
    return


@app.cell
def _(model, project_root):
    fine_model_path = project_root / "models" / "ch06"
    os.makedirs(fine_model_path, exist_ok=True)
    torch.save(model.state_dict(), fine_model_path / "review_classifier.pth")
    return (fine_model_path,)


@app.cell
def _():
    mo.md(r"""
    And load the model by this
    """)
    return


@app.cell
def _(fine_model_path, model):
    _model_state_dict = torch.load(fine_model_path / "review_classifier.pth", map_location=device, weights_only=True)
    model.load_state_dict(_model_state_dict)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
