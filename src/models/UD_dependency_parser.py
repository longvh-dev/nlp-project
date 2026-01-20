from typing import Any, List, Optional, Tuple, Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy


class Biaffine(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_dim, in_dim + 1, in_dim + 1))

    def forward(self, x, y):
        # x: (B, T, D)
        # y: (B, T, D)
        B, T, D = x.size()
        ones = torch.ones(B, T, 1, device=x.device)
        x_ = torch.cat([x, ones], dim=-1)
        y_ = torch.cat([y, ones], dim=-1)

        # (B, out_dim, T, T)
        s = torch.einsum("bxi,oij,byj->boxy", x_, self.U, y_)
        return s


class UDDependencyParser(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        num_relations: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        arc_mlp_dim: int = 128,
        rel_mlp_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bi-LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # //2 for bidirectional
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Arc (Head) Prediction layers
        self.arc_head = nn.Linear(hidden_dim * 2, arc_mlp_dim)  # *2 for bidirectional
        self.arc_dep = nn.Linear(arc_mlp_dim, 1)

        # Label (Relation) Prediction layers
        self.rel_head = nn.Linear(hidden_dim * 2, rel_mlp_dim)  # *2 for bidirectional
        self.rel_dep = nn.Linear(rel_mlp_dim, num_relations)

        self.arc_biaffine = Biaffine(arc_mlp_dim, 1)
        self.rel_biaffine = Biaffine(rel_mlp_dim, num_relations)

        self.dropout = nn.Dropout(dropout)

        # Loss functions
        self.arc_loss = nn.CrossEntropyLoss(
            ignore_index=-1
        )  # ignore -1 for padded heads
        self.rel_loss = nn.CrossEntropyLoss(
            ignore_index=-1
        )  # ignore -1 for padded rels

        # Metrics
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_relations, ignore_index=-1
        )
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_relations, ignore_index=-1
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_relations, ignore_index=-1
        )

    def _forward(self, input_ids, word_starts, attention_mask):
        # input_ids: (batch_size, seq_len) - subword tokens
        # word_starts: (batch_size, max_words) - indices of first subword for each word
        # attention_mask: (batch_size, seq_len) - mask for subword tokens

        # 1. Get subword embeddings
        subword_embeddings = self.embedding(
            input_ids
        )  # (batch_size, seq_len, embedding_dim)

        # 2. Extract word-level representations from subword embeddings
        # This is a simplified approach: take the embedding of the first subword token for each word
        batch_size, max_words = word_starts.shape
        word_embeddings = torch.zeros(
            batch_size, max_words, self.hparams.embedding_dim, device=self.device
        )

        # Iterate over batch to handle varying numbers of words per sentence
        for i in range(batch_size):
            # Get valid word_starts for the current sentence (excluding padding)
            valid_word_starts = word_starts[i][
                word_starts[i] != 0
            ]  # Assuming 0 is padding or start of sentence

            # If word_starts has a 0 and it corresponds to the first token [CLS]
            # then we should take the embedding of that first token.
            # However, word_starts from dataloader includes [CLS] position (0) and then actual words.
            # We need to correctly map word_starts to subword_embeddings.

            # Let's consider the structure: input_ids = [CLS_ID, subword1, subword2, ..., SEP_ID, PAD, PAD]
            # word_starts = [0, idx_word1, idx_word2, ..., 0, 0] (padded with 0)
            # The actual word start indices are where word_starts[i] is > 0 and within subword_embeddings length.

            # The first entry in word_starts should be for [CLS], which is 0.
            # Subsequent entries are for actual words.

            # For simplicity, let's just select the first subword embedding for each word index.
            # We need to ensure that the indices are within the bounds of subword_embeddings.

            # Filter out padding word_starts (assuming 0 is padding for word_starts and real word starts are >0)
            # Or, more robustly, filter out word_starts that point to padded input_ids

            # The current collate_fn puts 0 for padded word_starts, and the first word_start is 0 (for [CLS])
            # So, we should select `word_starts[i, j]` as the index into `subword_embeddings[i]`.

            # Create a mask for valid word_starts (non-padding indices)
            actual_word_start_indices = word_starts[i, :]

            # Filter out indices that are out of bounds or are padding (assuming padding is 0 and valid indices are > 0)
            # A more robust check: actual_word_start_indices < attention_mask[i].sum()
            valid_mask = (actual_word_start_indices < attention_mask[i].sum()) & (
                actual_word_start_indices >= 0
            )

            if valid_mask.any():
                selected_indices = actual_word_start_indices[valid_mask]
                word_embeddings[i, valid_mask] = subword_embeddings[i, selected_indices]

        # 3. Bi-LSTM Encoder
        # Apply dropout to word embeddings before LSTM
        word_embeddings = self.dropout(word_embeddings)
        encoder_outputs, _ = self.encoder(
            word_embeddings
        )  # (batch_size, max_words, hidden_dim * 2)
        encoder_outputs = self.dropout(encoder_outputs)

        # 4. Arc Prediction
        arc_mlp_output = torch.tanh(
            self.arc_mlp(encoder_outputs)
        )  # (batch_size, max_words, hidden_dim)

        # Calculate scores for all possible head-dependent pairs
        # head_scores: (batch_size, max_words, hidden_dim)
        # dep_scores: (batch_size, max_words, hidden_dim)
        head_scores = arc_mlp_output  # For heads
        dep_scores = arc_mlp_output  # For dependents

        # (batch_size, max_words, max_words) - score of word_j being head of word_i
        # dot product between dependent representation and head representation
        arc_scores = torch.bmm(dep_scores, head_scores.transpose(1, 2))

        # 5. Label Prediction (simplified: independent of head prediction)
        rel_mlp_output = torch.tanh(
            self.rel_mlp(encoder_outputs)
        )  # (batch_size, max_words, hidden_dim)
        rel_scores = self.rel_predictor(
            rel_mlp_output
        )  # (batch_size, max_words, num_relations)

        return arc_scores, rel_scores

    def forward(self, input_ids, word_starts, attention_mask):
        # input_ids: (batch_size, seq_len) - subword tokens
        # word_starts: (batch_size, max_words) - indices of first subword for each word
        # attention_mask: (batch_size, seq_len) - mask for subword tokens

        # 1. Get subword embeddings
        subword_embeddings = self.embedding(
            input_ids
        )  # (batch_size, seq_len, embedding_dim)

        # 2. Extract word-level representations from subword embeddings
        # This is a simplified approach: take the embedding of the first subword token for each word
        batch_size, max_words = word_starts.shape
        word_embeddings = torch.zeros(
            batch_size, max_words, self.hparams.embedding_dim, device=self.device
        )

        # Iterate over batch to handle varying numbers of words per sentence
        for i in range(batch_size):
            # Get valid word_starts for the current sentence (excluding padding)
            valid_word_starts = word_starts[i][
                word_starts[i] != 0
            ]  # Assuming 0 is padding or start of sentence

            # If word_starts has a 0 and it corresponds to the first token [CLS]
            # then we should take the embedding of that first token.
            # However, word_starts from dataloader includes [CLS] position (0) and then actual words.
            # We need to correctly map word_starts to subword_embeddings.

            # Let's consider the structure: input_ids = [CLS_ID, subword1, subword2, ..., SEP_ID, PAD, PAD]
            # word_starts = [0, idx_word1, idx_word2, ..., 0, 0] (padded with 0)
            # The actual word start indices are where word_starts[i] is > 0 and within subword_embeddings length.

            # The first entry in word_starts should be for [CLS], which is 0.
            # Subsequent entries are for actual words.

            # For simplicity, let's just select the first subword embedding for each word index.
            # We need to ensure that the indices are within the bounds of subword_embeddings.

            # Filter out padding word_starts (assuming 0 is padding for word_starts and real word starts are >0)
            # Or, more robustly, filter out word_starts that point to padded input_ids

            # The current collate_fn puts 0 for padded word_starts, and the first word_start is 0 (for [CLS])
            # So, we should select `word_starts[i, j]` as the index into `subword_embeddings[i]`.

            # Create a mask for valid word_starts (non-padding indices)
            actual_word_start_indices = word_starts[i, :]

            # Filter out indices that are out of bounds or are padding (assuming padding is 0 and valid indices are > 0)
            # A more robust check: actual_word_start_indices < attention_mask[i].sum()
            valid_mask = (actual_word_start_indices < attention_mask[i].sum()) & (
                actual_word_start_indices >= 0
            )

            if valid_mask.any():
                selected_indices = actual_word_start_indices[valid_mask]
                word_embeddings[i, valid_mask] = subword_embeddings[i, selected_indices]

        # 3. Bi-LSTM Encoder
        # Apply dropout to word embeddings before LSTM
        word_embeddings = self.dropout(word_embeddings)
        encoder_outputs, _ = self.encoder(
            word_embeddings
        )  # (batch_size, max_words, hidden_dim * 2)
        encoder_outputs = self.dropout(encoder_outputs)

        arc_h = F.relu(self.arc_head(encoder_outputs))
        arc_d = F.relu(self.arc_dep(encoder_outputs))

        rel_h = F.relu(self.rel_head(encoder_outputs))
        rel_d = F.relu(self.rel_dep(encoder_outputs))

        arc_scores = self.arc_biaffine(arc_d, arc_h).squeeze(1)
        rel_scores = self.rel_biaffine(rel_d, rel_h)

        return arc_scores, rel_scores

    def _step(self, batch: Dict[str, Any], batch_idx: int = 0):
        input_ids = batch["input_ids"]  # (batch_size, seq_len)
        word_starts = batch["word_starts"]  # (batch_size, max_words)
        attention_mask = batch["attention_mask"]  # (batch_size, seq_len)
        heads = batch["heads"]  # Gold heads (batch_size, max_words)
        rels = batch["rels"]  # Gold relations (batch_size, max_words)

        B, T = input_ids.size()
        arc_scores, rel_scores = self.forward(input_ids, word_starts, attention_mask)

        # # --------------------------------------------------
        # # 1. Build word-level mask (exclude PAD words)
        # # --------------------------------------------------
        # word_mask = heads != -1  # (B, T)

        # # --------------------------------------------------
        # # 2. Extract word-level representations
        # # --------------------------------------------------
        # # Encode at subword level
        # x = self.embedding(input_ids)
        # x = self.dropout(x)
        # enc_sub, _ = self.encoder(x)
        # enc_sub = self.dropout(enc_sub)

        # # Gather first-subword representations for each word
        # enc_words = torch.zeros(B, T, enc_sub.size(-1), device=enc_sub.device)
        # for b in range(B):
        #     valid = word_mask[b]
        #     idx = word_starts[b, valid]
        #     enc_words[b, valid] = enc_sub[b, idx]
        # # --------------------------------------------------
        # # 3. Arc & relation representations
        # # --------------------------------------------------
        # arc_h = F.relu(self.arc_head(enc_words))
        # arc_d = F.relu(self.arc_dep(enc_words))
        # rel_h = F.relu(self.rel_head(enc_words))
        # rel_d = F.relu(self.rel_dep(enc_words))

        # arc_scores = self.arc_biaffine(arc_d, arc_h).squeeze(1)  # (B, T, T)
        # rel_scores = self.rel_biaffine(rel_d, rel_h)  # (B, R, T, T)

        # # --------------------------------------------------
        # # 4. ARC LOSS (CrossEntropy over heads)
        # # --------------------------------------------------
        # arc_scores_flat = arc_scores[word_mask]
        # heads_flat = heads[word_mask]
        # arc_loss = self.arc_loss(arc_scores_flat, heads_flat)

        # # --------------------------------------------------
        # # 5. RELATION LOSS (conditioned on GOLD heads)
        # # --------------------------------------------------
        # # rel_scores: (B, R, T, T) -> (B, T, T, R)
        # rel_scores = rel_scores.permute(0, 2, 3, 1)

        # # clamp to avoid -1 indexing
        # heads_clamped = heads.clamp(min=0)

        # # select relation scores corresponding to gold heads
        # rel_scores_gold = rel_scores[
        #     torch.arange(B).unsqueeze(1), torch.arange(T), heads_clamped
        # ]  # (B, T, R)

        # rel_scores_flat = rel_scores_gold[word_mask]
        # rels_flat = rels[word_mask]
        # rel_loss = self.rel_loss(rel_scores_flat, rels_flat)

        # loss = arc_loss + rel_loss
        # return loss, arc_loss, rel_loss

        ####################################################################
        # Calculate arc loss
        # arc_scores: (batch_size, max_words, max_words)
        # heads: (batch_size, max_words) - indices of true heads

        # Flatten for CrossEntropyLoss
        # Only consider actual words (non-padded and not [CLS] for head prediction)
        # The first token is [CLS], which typically doesn't have a head or relation prediction in this context.
        # So we should exclude it from loss calculation, along with padding.

        # Mask for actual words (excluding [CLS] and padding)
        # In DependencyDataset, heads and rels are padded with -1.
        # [CLS] token is at index 0. Actual words start from index 1.

        active_words_mask = (
            (heads != -1) & (rels != -1) & (heads != 0)
        )  # Exclude CLS (index 0) and padding (-1)

        # Arc loss: predict head for each dependent word
        # (batch_size * num_active_words, max_words)
        flat_arc_scores = arc_scores[active_words_mask].view(-1, arc_scores.shape[-1])
        flat_heads = heads[active_words_mask].view(-1)

        arc_loss = self.arc_loss(flat_arc_scores, flat_heads)

        # Calculate relation loss
        # (batch_size * num_active_words, num_relations)
        flat_rel_scores = rel_scores[active_words_mask].view(-1, rel_scores.shape[-1])
        flat_rels = rels[active_words_mask].view(-1)

        rel_loss = self.rel_loss(flat_rel_scores, flat_rels)

        total_loss = arc_loss + rel_loss

        # Log metrics (accuracy for relations, for arcs it's more complex (UAS/LAS))
        # For arcs, we can compute a simple token-level accuracy (how many words predicted correct head)
        # For relations, we can compute token-level accuracy.

        # Arc prediction for accuracy: get argmax from scores
        predicted_heads = torch.argmax(arc_scores, dim=-1)  # (batch_size, max_words)

        # Select predictions for active words
        active_predicted_heads = predicted_heads[active_words_mask]
        active_gold_heads = heads[active_words_mask]

        arc_accuracy = (
            (active_predicted_heads == active_gold_heads).float().mean()
            if active_gold_heads.numel() > 0
            else torch.tensor(0.0, device=self.device)
        )

        # Relation prediction for accuracy
        predicted_rels = torch.argmax(rel_scores, dim=-1)  # (batch_size, max_words)

        # Select predictions for active words
        active_predicted_rels = predicted_rels[active_words_mask]
        active_gold_rels = rels[active_words_mask]

        rel_accuracy = (
            (active_predicted_rels == active_gold_rels).float().mean()
            if active_gold_rels.numel() > 0
            else torch.tensor(0.0, device=self.device)
        )

        return total_loss, arc_loss, rel_loss, arc_accuracy, rel_accuracy

    def training_step(self, batch, batch_idx):
        loss, arc_loss, rel_loss = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/arc_loss", arc_loss)
        self.log("train/rel_loss", rel_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, arc_loss, rel_loss = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, arc_loss, rel_loss = self._step(batch)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # def training_step(self, batch, batch_idx):
    # total_loss, arc_loss, rel_loss, arc_accuracy, rel_accuracy = self._step(
    #     batch, batch_idx
    # )
    # self.log(
    #     "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
    # )
    # self.log("train/arc_loss", arc_loss, on_step=False, on_epoch=True)
    # self.log("train/rel_loss", rel_loss, on_step=False, on_epoch=True)
    # self.log("train/arc_accuracy", arc_accuracy, on_step=False, on_epoch=True)
    # self.log("train/rel_accuracy", rel_accuracy, on_step=False, on_epoch=True)
    # return total_loss

    # def validation_step(self, batch, batch_idx):
    #     total_loss, arc_loss, rel_loss, arc_accuracy, rel_accuracy = self._step(
    #         batch, batch_idx
    #     )
    #     self.log(
    #         "val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
    #     )
    #     self.log("val/arc_loss", arc_loss, on_step=False, on_epoch=True)
    #     self.log("val/rel_loss", rel_loss, on_step=False, on_epoch=True)
    #     self.log("val/arc_accuracy", arc_accuracy, on_step=False, on_epoch=True)
    #     self.log("val/rel_accuracy", rel_accuracy, on_step=False, on_epoch=True)

    # def test_step(self, batch, batch_idx):
    #     total_loss, arc_loss, rel_loss, arc_accuracy, rel_accuracy = self._step(
    #         batch, batch_idx
    #     )
    #     self.log("test/total_loss", total_loss, on_step=False, on_epoch=True)
    #     self.log("test/arc_loss", arc_loss, on_step=False, on_epoch=True)
    #     self.log("test/rel_loss", rel_loss, on_step=False, on_epoch=True)
    #     self.log("test/arc_accuracy", arc_accuracy, on_step=False, on_epoch=True)
    #     self.log("test/rel_accuracy", rel_accuracy, on_step=False, on_epoch=True)

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #     return optimizer
