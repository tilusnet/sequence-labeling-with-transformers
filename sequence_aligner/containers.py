import torch
from dataclasses import dataclass
from typing import List, Tuple

IntList = List[int]  # A list of token_ids
IntListList = List[IntList]  # A List of List of token_ids, e.g. a Batch
IntPair = Tuple[int, int]
IntPairList = List[IntPair]
IntPairListList = List[IntPairList]  # A List of list of int pairs, e.g. for offsets


@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    labels: IntList
    offsets: IntPairList


class TrainingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        self.offsets: IntPairListList = []
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []
        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
            self.offsets.append(ex.offsets)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)
