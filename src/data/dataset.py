import torch
from torch.utils.data import Dataset, DataLoader
from path import Path
from lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
from transformers import AutoTokenizer


def read_conllu(path: Path):
    sentences = []
    sent = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0]:
                continue
            sent.append(
                {
                    "form": cols[1],
                    "upos": cols[3],
                    "head": int(cols[6]),
                    "deprel": cols[7],
                }
            )
    return sentences


class VUDataset(Dataset):
    def __init__(self, path, word2id, pos2id, rel2id):
        self.sents = read_conllu(path)
        self.word2id = word2id
        self.pos2id = pos2id
        self.rel2id = rel2id

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sent = self.sents[idx]
        words = [self.word2id.get(w["form"], 1) for w in sent]
        pos = [self.pos2id[w["upos"]] for w in sent]
        heads = [w["head"] for w in sent]
        rels = [self.rel2id[w["deprel"]] for w in sent]
        return (
            torch.tensor(words),
            torch.tensor(pos),
            torch.tensor(heads),
            torch.tensor(rels),
        )


class DependencyDataset(Dataset):
    def __init__(self, sentences, tokenizer, rel_vocab=None):
        self.sentences = sentences
        self.tokenizer = tokenizer

        if rel_vocab is None:
            self.rel_vocab = {"<pad>": 0, "root": 1}
            for sent in sentences:
                for token in sent:
                    rel = token["deprel"]
                    if rel not in self.rel_vocab:
                        self.rel_vocab[rel] = len(self.rel_vocab)
        else:
            self.rel_vocab = rel_vocab

        self.id2rel = {v: k for k, v in self.rel_vocab.items()}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        words = [token["form"] for token in sent]

        input_ids = [self.tokenizer.cls_token_id]
        word_starts = [0]
        current_idx = 1

        for word in words:
            subwords = self.tokenizer.encode(word, add_special_tokens=False)
            if len(subwords) == 0:
                subwords = [self.tokenizer.unk_token_id]
            input_ids.extend(subwords)
            word_starts.append(current_idx)
            current_idx += len(subwords)

        input_ids.append(self.tokenizer.sep_token_id)

        heads = [0] + [token["head"] for token in sent]
        rels = [0]
        for token in sent:
            rels.append(self.rel_vocab.get(token["deprel"], 0))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "word_starts": torch.tensor(word_starts, dtype=torch.long),
            "heads": torch.tensor(heads, dtype=torch.long),
            "rels": torch.tensor(rels, dtype=torch.long),
        }


def collate_fn(batch):
    max_len_ids = max(len(x["input_ids"]) for x in batch)
    max_len_words = max(len(x["word_starts"]) for x in batch)

    input_ids_batch = []
    mask_batch = []
    word_starts_batch = []
    heads_batch = []
    rels_batch = []

    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len_ids - len(ids)

        ids_padded = torch.cat([ids, torch.tensor([1] * pad_len, dtype=torch.long)])
        input_ids_batch.append(ids_padded)

        mask_batch.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]))

        ws = item["word_starts"]
        ws_pad = max_len_words - len(ws)
        ws_padded = torch.cat([ws, torch.zeros(ws_pad, dtype=torch.long)])
        word_starts_batch.append(ws_padded)

        h = item["heads"]
        h_padded = torch.cat([h, torch.full((ws_pad,), -1, dtype=torch.long)])
        heads_batch.append(h_padded)

        r = item["rels"]
        r_padded = torch.cat([r, torch.full((ws_pad,), -1, dtype=torch.long)])
        rels_batch.append(r_padded)

    return {
        "input_ids": torch.stack(input_ids_batch),  # Stack xong vẫn là Long
        "attention_mask": torch.stack(mask_batch),
        "word_starts": torch.stack(word_starts_batch),
        "heads": torch.stack(heads_batch),
        "rels": torch.stack(rels_batch),
    }


class ConlluDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_file: str = "train.conllu",
        val_file: str = "dev.conllu",
        test_file: str = "test.conllu",
        tokenizer: Any = None, # This will be instantiated by Hydra
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.tokenizer = tokenizer # Store tokenizer passed by hydra
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """No special preparation needed for CoNLL-U files, assumed to be present."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        if not self.data_train and not self.data_val and not self.data_test:
            # Read all sentences
            train_sentences = read_conllu(Path(self.hparams.data_dir) / self.hparams.train_file)
            val_sentences = read_conllu(Path(self.hparams.data_dir) / self.hparams.val_file)
            test_sentences = read_conllu(Path(self.hparams.data_dir) / self.hparams.test_file)

            # Create training dataset, which will build the rel_vocab
            self.data_train = DependencyDataset(train_sentences, self.tokenizer)
            
            # Use the rel_vocab from the training dataset for validation and test
            rel_vocab = self.data_train.rel_vocab
            self.data_val = DependencyDataset(val_sentences, self.tokenizer, rel_vocab=rel_vocab)
            self.data_test = DependencyDataset(test_sentences, self.tokenizer, rel_vocab=rel_vocab)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
