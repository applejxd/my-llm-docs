import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    from importlib.metadata import version

    import matplotlib.pyplot as plt
    import tiktoken
    import torch
    import torch.nn as nn
    from ch03 import MultiHeadAttention


@app.cell
def _(mo):
    mo.md(r"""
    # Chapter 4: Implementing a GPT model from Scratch To Generate Text
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.1 Coding an LLM architecture
    """)
    return


@app.cell
def _():
    print("matplotlib version:", version("matplotlib"))
    print("torch version:", version("torch"))
    print("tiktoken version:", version("tiktoken"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    This "Dummy" means untrained model just to check forward path algorithms.
    """)
    return


@app.class_definition
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


@app.class_definition
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


@app.class_definition
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x


@app.cell
def _():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }
    return (GPT_CONFIG_124M,)


@app.cell
def _(mo):
    mo.md(r"""
    Use GPT2 embedding and create a batch contains two phrases
    """)
    return


@app.cell
def _():
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    return batch, tokenizer


@app.cell
def _(mo):
    mo.md(r"""
    Randomly initialize the GPT model and inference logits with untrained weights
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, batch):
    torch.manual_seed(123)
    _model = DummyGPTModel(GPT_CONFIG_124M)

    _logits = _model(batch)
    print("Output shape:", _logits.shape)
    print(_logits)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.2 Normalizing activations with layer normalization
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    check the behavior of layer normalization with this sample layer
    """)
    return


@app.cell
def _():
    torch.manual_seed(123)

    # create 2 training examples with 5 dimensions (features) each
    batch_example = torch.randn(2, 5) 

    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)
    return batch_example, out


@app.cell
def _(mo):
    mo.md(r"""
    statistics before normalization
    """)
    return


@app.cell
def _(out):
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)
    return mean, var


@app.cell
def _(mo):
    mo.md(r"""
    statistics after normalization
    """)
    return


@app.cell
def _(mean, out, var):
    out_norm = (out - mean) / torch.sqrt(var)
    print("Normalized layer outputs:\n", out_norm)

    mean_norm = out_norm.mean(dim=-1, keepdim=True)
    var_norm = out_norm.var(dim=-1, keepdim=True)
    print("Mean:\n", mean_norm)
    print("Variance:\n", var_norm)
    return mean_norm, var_norm


@app.cell
def _(mo):
    mo.md(r"""
    Disable PyTorch scientific notation for readability
    """)
    return


@app.cell
def _(mean_norm, var_norm):
    torch.set_printoptions(sci_mode=False)
    print("Mean:", mean_norm)
    print("Variance:", var_norm)
    return


@app.class_definition
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


@app.cell
def _(batch_example):
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)

    mean_ln = out_ln.mean(dim=-1, keepdim=True)
    var_ln = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    print("Mean:\n", mean_ln)
    print("Variance:\n", var_ln)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.3 Implementing a feed forward network with GELU activations
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right]\right)
    $
    """)
    return


@app.class_definition
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


@app.cell
def _():
    gelu, relu = GELU(), nn.ReLU()

    # Some sample data
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    return


@app.class_definition
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


@app.cell
def _(GPT_CONFIG_124M):
    print(GPT_CONFIG_124M["emb_dim"])
    return


@app.cell
def _(GPT_CONFIG_124M):
    ffn = FeedForward(GPT_CONFIG_124M)

    # input shape: [batch_size, num_token, emb_size]
    _x = torch.rand(2, 3, 768) 
    print(ffn(_x).shape)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.4 Adding shortcut connections
    """)
    return


@app.class_definition
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


@app.function
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


@app.cell
def _():
    layer_sizes = [3, 3, 3, 3, 3, 1]  

    sample_input = torch.tensor([[1., 0., -1.]])

    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)
    return layer_sizes, sample_input


@app.cell
def _(layer_sizes, sample_input):
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print_gradients(model_with_shortcut, sample_input)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.5 Connecting attention and linear layers in a transformer block
    """)
    return


@app.class_definition
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


@app.cell
def _(GPT_CONFIG_124M):
    torch.manual_seed(123)

    _x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(GPT_CONFIG_124M)
    _output = block(_x)

    print("Input shape:", _x.shape)
    print("Output shape:", _output.shape)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.6 Coding the GPT model
    """)
    return


@app.class_definition
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


@app.cell
def _(GPT_CONFIG_124M, batch):
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    _out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", _out.shape)
    print(_out)
    return (model,)


@app.cell
def _(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    return (total_params,)


@app.cell
def _(model):
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    return


@app.cell
def _(model, total_params):
    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
    return


@app.cell
def _(total_params):
    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size of the model: {total_size_mb:.2f} MB")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.7 Generating text
    """)
    return


@app.function
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


@app.cell
def _(tokenizer):
    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    return (encoded_tensor,)


@app.cell
def _(GPT_CONFIG_124M, encoded_tensor, model):
    model.eval() # disable dropout

    out_gen = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output:", out_gen)
    print("Output length:", len(out_gen[0]))
    return (out_gen,)


@app.cell
def _(out_gen, tokenizer):
    decoded_text = tokenizer.decode(out_gen.squeeze(0).tolist())
    print(decoded_text)
    return


if __name__ == "__main__":
    app.run()
