import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    from importlib.metadata import version
    import marimo as mo

    import torch
    import torch.nn as nn


@app.cell
def _():
    mo.md(r"""
    # Chapter 3: Coding Attention Mechanisms
    """)
    return


@app.cell
def _():
    print("torch version:", version("torch"))
    return


@app.cell
def _():
    mo.md(r"""
    ## 3.3 Attending to different parts of the input with self-attention
    """)
    return


@app.cell
def _():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]  
    )
    return (inputs,)


@app.cell
def _(inputs):
    x_2 = inputs[1]  # second input element
    d_in = inputs.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2
    return d_in, d_out


@app.cell
def _():
    mo.md(r"""
    ### 3.4.2 Implementing a compact SelfAttention class
    """)
    return


@app.cell
def _(d_in, d_out, inputs):
    class SelfAttention_v1(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value

            attn_scores = queries @ keys.T  # omega
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
    return


@app.cell
def _():
    mo.md(r"""
    ### 3.5.3 Implementing a compact causal self-attention class
    """)
    return


@app.cell
def _(inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    print(
        batch.shape
    )  # 2 inputs with 6 tokens each, and each token has embedding dimension 3
    return (batch,)


@app.cell
def _():
    mo.md(r"""
    ### 3.6.2 Implementing multi-head attention with weight splits
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Define this class in a cell without other expressions to be imported by other notebooks.
    Use `@app.class_definition` decorator for this class to export (see this Python file itself).
    """)
    return


@app.class_definition
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`,
        # this will result in errors in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


@app.cell
def _(batch):
    torch.manual_seed(123)

    batch_size, context_length, _d_in = batch.shape
    _d_out = 2
    mha = MultiHeadAttention(_d_in, _d_out, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
