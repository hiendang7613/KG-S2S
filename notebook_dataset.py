"""Dataset utilities replicated from the provided notebook."""

from typing import List, Dict, Optional
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer


tokenizer = T5Tokenizer.from_pretrained("t5-small", padding=True)

def _tokenize(text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt")["input_ids"][0][:-1]


class Hop1Index:
    """Simple helper to fetch 1-hop neighbours from sorted triples."""

    def __init__(self, triples: np.ndarray, num_entities: int, key_col: int = 0, max_context_size: int = 64):
        self.max_context_size = max_context_size
        self.shuffle = False
        self.key_col = key_col
        self.triples = triples[triples[:, key_col].argsort()]
        keys, values_offset = np.unique(self.triples[:, key_col], return_index=True)
        values_offset = np.append(values_offset, len(self.triples))
        self.key_to_start = -1 * np.ones(num_entities, dtype=int)
        self.key_to_start[keys] = values_offset[:-1]
        self.key_to_end = -1 * np.ones(num_entities, dtype=int)
        self.key_to_end[keys] = values_offset[1:]

    def __getitem__(self, item: int, rel_id: Optional[int] = None):
        start = self.key_to_start[item]
        end = self.key_to_end[item]
        if start == -1:
            return np.zeros((0, 2), dtype=int)
        context = self.triples[start:end, [1, 2 - self.key_col]]
        if rel_id is not None:
            context = context[context[:, 0] == rel_id][:, 1]
        if len(context) > self.max_context_size:
            ids = np.random.choice(len(context), self.max_context_size, replace=False)
            context = context[ids]
        if self.shuffle:
            np.random.shuffle(context)
        return context

    def get_context(self, item: int, rel_id: Optional[int] = None):
        return self.__getitem__(item, rel_id)


class RotatE:
    """Simple RotatE embedding wrapper used in the notebook."""

    def __init__(self, k: int, max_rel_size: int, entity_embedding: torch.Tensor, relation_embedding: torch.Tensor):
        self.internal_k = 2 * k
        self.max_rel_size = max_rel_size
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

    def __call__(self, e_s_id: int, e_p_id: int) -> torch.Tensor:
        e_s = self.entity_embedding[e_s_id]
        e_p = self.relation_embedding[e_p_id]
        e_s_real, e_s_img = torch.chunk(e_s, 2, dim=0)
        theta_pred, _ = torch.chunk(e_p, 2, dim=0)
        embedding_range = (6 / (self.internal_k * self.max_rel_size)) ** 0.5
        e_p_real = torch.cos(theta_pred / (embedding_range / np.pi))
        e_p_img = torch.sin(theta_pred / (embedding_range / np.pi))
        e_o_real = e_s_real * e_p_real - e_s_img * e_p_img
        e_o_img = e_s_real * e_p_img + e_s_img * e_p_real
        return torch.cat([e_o_real, e_o_img], dim=0)


class KGCDataset(Dataset):
    """Dataset used in the notebook for structured KGC."""

    def __init__(self, kg_data: Dict[str, np.ndarray], structural_model: RotatE, num_ents: int = 14541):
        self.structural_model = structural_model
        self.num_ents = num_ents
        self.id_triplets = {
            'train': kg_data['train_triplet_id'],
            'valid': kg_data['valid_triplet_id'],
            'test': kg_data['test_triplet_id'],
        }
        self.tokens_triplets = {
            'train': kg_data['train_triplet_tokens'],
            'valid': kg_data['valid_triplet_tokens'],
            'test': kg_data['test_triplet_tokens'],
        }
        self.decs_triplets = {
            'train': kg_data['train_triplet_decs'],
            'valid': kg_data['valid_triplet_decs'],
            'test': kg_data['test_triplet_decs'],
        }
        self.get_neigs_0 = {
            s: Hop1Index(self.id_triplets['train'], self.num_ents, 0) for s in ['train', 'valid', 'test']
        }
        self.get_neigs_2 = {
            s: Hop1Index(self.id_triplets['train'], self.num_ents, 2) for s in ['train', 'valid', 'test']
        }
        self.mask_token = _tokenize('')
        self.eos_token = torch.tensor([tokenizer.eos_token_id])
        self.predict_head_token = _tokenize('predict head :')
        self.predict_tail_token = _tokenize('predict tail :')
        self.start_decs_token = _tokenize('[')
        self.end_decs_token = _tokenize(']')
        self.inversion_token = _tokenize('inversion of ')
        self.empty_token = torch.tensor([], dtype=torch.long)
        self.p_dropout = 0.0
        self.split = 'train'

    def __len__(self):
        return len(self.tokens_triplets[self.split])

    def __getitem__(self, idx: int):
        return self.get(idx, self.split)

    def get(self, idx: int, split: str = 'train', full_mask_part_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        head_lbl, relation, tail_lbl = self.tokens_triplets[split][idx]
        head_id, rel_id, tail_id = self.id_triplets[split][idx]
        head_decs, tail_decs = self.decs_triplets[split][idx]

        full_mask_part_idx = 2 if full_mask_part_idx is None and random.randint(0, 1) else full_mask_part_idx
        inversion = False

        if full_mask_part_idx:
            source = [
                self.predict_tail_token,
                head_lbl,
                self.start_decs_token,
                head_decs,
                self.end_decs_token,
                self.inversion_token if inversion else self.empty_token,
                relation,
            ]
            target = [tail_lbl]
            neighboors_0 = self.get_neigs_0[split][head_id]
            neighboors_0 = neighboors_0[(neighboors_0[:, 0] != rel_id) | (neighboors_0[:, 1] != tail_id)]
            neighboors_2 = self.get_neigs_2[split][head_id]
            neighboors_2 = neighboors_2[(neighboors_2[:, 0] != rel_id) | (neighboors_2[:, 1] != tail_id)]
        else:
            source = [
                self.predict_head_token,
                tail_lbl,
                self.start_decs_token,
                tail_decs,
                self.end_decs_token,
                self.inversion_token if inversion else self.empty_token,
                relation,
            ]
            target = [head_lbl]
            neighboors_0 = self.get_neigs_0[split][tail_id]
            neighboors_0 = neighboors_0[(neighboors_0[:, 0] != rel_id) | (neighboors_0[:, 1] != head_id)]
            neighboors_2 = self.get_neigs_2[split][tail_id]
            neighboors_2 = neighboors_2[(neighboors_2[:, 0] != rel_id) | (neighboors_2[:, 1] != head_id)]

        source.append(self.eos_token)
        target.append(self.eos_token)
        source_tensor = torch.cat(source)
        target_tensor = torch.cat(target)
        attention_mask = torch.ones_like(source_tensor)

        return {
            'input_ids': source_tensor,
            'attention_mask': attention_mask,
            'labels': target_tensor,
            'triplet': self.id_triplets[split][idx],
            'neighboors_0_id': neighboors_0,
            'neighboors_2_id': neighboors_2,
        }


class DataCollatorForSeq2Seq:
    """Minimal data collator used with HuggingFace Trainer."""

    def __init__(self, tokenizer: T5Tokenizer, label_pad_token_id: int = -100, data_names: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.data_names = data_names or []

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for name in self.data_names:
            padding_value = self.label_pad_token_id if name == 'labels' else self.tokenizer.pad_token_id
            tensors = [f[name] for f in features]
            batch[name] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        return batch
