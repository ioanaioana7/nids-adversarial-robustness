"""FT-Transformer implementat de la zero in PyTorch.

Referinta: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
NeurIPS 2021.

Un Transformer clasic nu se aplica direct pe date tabelare: nu exista o secventa
peste care sa se faca atentie. FT-Transformer rezolva asta TOKENIZAND fiecare
caracteristica: fiecare coloana devine un vector de dimensiune d_token (numeric
prin proiectie liniara invatata, categoric prin embedding). Se adauga un token
[CLS] si peste secventa rezultata (1 + n_features tokeni) se aplica blocuri de
Transformer cu self-attention multi-head. Clasificarea se face din reprezentarea
finala a lui [CLS].

Atentia este scrisa explicit (nu prin nn.MultiheadAttention) ca mecanismul sa fie
vizibil in cod pentru redactarea lucrarii.

NOTA pentru extinderi viitoare (NU implementat aici): spre deosebire de RF/XGBoost,
aceasta retea e diferentiabila end-to-end, ceea ce ar permite atacuri adversariale
bazate pe gradient (FGSM/PGD). Vezi hook-ul documentat din predictor.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


class FeatureTokenizer(nn.Module):
    """Transforma (x_num, x_cat) intr-o secventa de tokeni [CLS] + cate unul per coloana.

    Numericele primesc cate un vector de pondere si unul de bias propriu, deci
    fiecare coloana are propria proiectie invatata catre spatiul de tokeni.
    Categoricele primesc cate un tabel de embedding per coloana.
    """

    def __init__(self, n_numeric: int, cardinalities: list[int], d_token: int):
        super().__init__()
        self.n_numeric = n_numeric
        self.d_token = d_token

        if n_numeric > 0:
            self.numeric_weight = nn.Parameter(torch.empty(n_numeric, d_token))
            self.numeric_bias = nn.Parameter(torch.empty(n_numeric, d_token))
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cardinalities])
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializare uniforma cu deviatie 1/sqrt(d_token), ca in lucrarea originala."""
        bound = 1 / math.sqrt(self.d_token)
        if self.n_numeric > 0:
            nn.init.uniform_(self.numeric_weight, -bound, bound)
            nn.init.uniform_(self.numeric_bias, -bound, bound)
        for embedding in self.categorical_embeddings:
            nn.init.uniform_(embedding.weight, -bound, bound)
        nn.init.uniform_(self.cls_token, -bound, bound)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x_num (torch.Tensor): [batch, n_numeric], deja standardizate.
            x_cat (torch.Tensor): [batch, n_categorical], indici intregi.

        Returns:
            torch.Tensor: [batch, 1 + n_features, d_token].
        """
        tokens = []
        if self.n_numeric > 0:
            # [batch, n_num, 1] * [n_num, d] -> [batch, n_num, d]
            tokens.append(x_num.unsqueeze(-1) * self.numeric_weight + self.numeric_bias)
        if len(self.categorical_embeddings) > 0:
            tokens.append(torch.stack(
                [embedding(x_cat[:, j]) for j, embedding in enumerate(self.categorical_embeddings)],
                dim=1))

        feature_tokens = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(feature_tokens.shape[0], -1, -1)
        return torch.cat([cls, feature_tokens], dim=1)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention multi-head, scrisa explicit.

    Pentru fiecare cap h: Attention(Q,K,V) = softmax(Q K^T / sqrt(d_head)) V.
    Capetele sunt concatenate si proiectate inapoi in d_token.
    """

    def __init__(self, d_token: int, n_heads: int, dropout: float):
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) trebuie divizibil cu n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = d_token // n_heads

        self.query = nn.Linear(d_token, d_token)
        self.key = nn.Linear(d_token, d_token)
        self.value = nn.Linear(d_token, d_token)
        self.projection = nn.Linear(d_token, d_token)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, tokens, d_token] -> [batch, n_heads, tokens, d_head]."""
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Parameters:
            x (torch.Tensor): [batch, tokens, d_token].
            return_attention (bool): daca True, returneaza si ponderile de atentie
                (utile pentru interpretabilitate in O4).

        Returns:
            torch.Tensor | tuple: iesirea [batch, tokens, d_token], optional cu
                ponderile [batch, n_heads, tokens, tokens].
        """
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        weights = self.dropout(F.softmax(scores, dim=-1))

        context = (weights @ v).transpose(1, 2).reshape(x.shape)
        out = self.projection(context)
        return (out, weights) if return_attention else out


class TransformerBlock(nn.Module):
    """Bloc pre-norm: LayerNorm -> atentie -> rezidual, apoi LayerNorm -> FFN -> rezidual."""

    def __init__(self, d_token: int, n_heads: int, ffn_hidden: int,
                 attn_dropout: float, ffn_dropout: float, residual_dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_token)
        self.attention = MultiHeadSelfAttention(d_token, n_heads, attn_dropout)
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, d_token),
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x)))
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class FTTransformer(nn.Module):
    """Feature-Tokenizer + Transformer pentru clasificare multi-clasa tabelara."""

    def __init__(self, n_numeric: int, cardinalities: list[int], n_classes: int,
                 d_token: int = config.TRANSFORMER_D_TOKEN,
                 n_blocks: int = config.TRANSFORMER_N_BLOCKS,
                 n_heads: int = config.TRANSFORMER_N_HEADS,
                 attn_dropout: float = config.TRANSFORMER_ATTN_DROPOUT,
                 ffn_dropout: float = config.TRANSFORMER_FFN_DROPOUT,
                 residual_dropout: float = config.TRANSFORMER_RESIDUAL_DROPOUT,
                 ffn_hidden_multiplier: float = config.TRANSFORMER_FFN_HIDDEN_MULTIPLIER):
        super().__init__()
        ffn_hidden = int(d_token * ffn_hidden_multiplier)
        self.tokenizer = FeatureTokenizer(n_numeric, cardinalities, d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_hidden,
                             attn_dropout, ffn_dropout, residual_dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x_num (torch.Tensor): [batch, n_numeric] standardizate.
            x_cat (torch.Tensor): [batch, n_categorical] indici.

        Returns:
            torch.Tensor: logits [batch, n_classes].
        """
        x = self.tokenizer(x_num, x_cat)
        for block in self.blocks:
            x = block(x)
        # Clasificam din tokenul [CLS] (pozitia 0).
        return self.head(self.head_norm(x[:, 0]))

    def hyperparameters(self) -> dict:
        """Hiperparametrii arhitecturali, pentru metadata de experiment."""
        return {
            "d_token": config.TRANSFORMER_D_TOKEN,
            "n_blocks": config.TRANSFORMER_N_BLOCKS,
            "n_heads": config.TRANSFORMER_N_HEADS,
            "attn_dropout": config.TRANSFORMER_ATTN_DROPOUT,
            "ffn_dropout": config.TRANSFORMER_FFN_DROPOUT,
            "residual_dropout": config.TRANSFORMER_RESIDUAL_DROPOUT,
            "ffn_hidden": int(config.TRANSFORMER_D_TOKEN * config.TRANSFORMER_FFN_HIDDEN_MULTIPLIER),
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }
