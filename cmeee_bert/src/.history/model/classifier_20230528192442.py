
from audioop import bias
from typing import Optional
from unicodedata import bidirectional

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
import transformers
from transformers import BertPreTrainedModel, BertConfig, BertModel, BartModel, PreTrainedModel, BartConfig
from transformers.file_utils import ModelOutput
import numpy as np

from ee_data import EE_label2id1, NER_PAD

NER_PAD_ID = EE_label2id1[NER_PAD]

from ee_data import EE_label2id1, NER_PAD

NER_PAD_ID = EE_label2id1[NER_PAD]


@dataclass
class NEROutputs(ModelOutput):
    """
    NOTE: `logits` here is the CRF decoding result.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.LongTensor] = None


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.num_labels = num_labels
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        self.loss_fct = CrossEntropyLoss()

    def _pred_labels(self, _logits):
        return torch.argmax(_logits, dim=-1)

    def forward(self, hidden_states, labels=None, no_decode=False):
        _logits = self.layers(hidden_states)
        loss, pred_labels = None, None

        if labels is None:
            pred_labels = self._pred_labels(_logits)    
        else:
            loss = self.loss_fct(_logits.view(-1, self.num_labels), labels.view(-1))
            if not no_decode:
                pred_labels = self._pred_labels(_logits)

        return NEROutputs(loss, pred_labels)


class CRFClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.num_labels = num_labels
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        self.crf = CRF(num_labels, batch_first=True)

    def _pred_labels(self, emissions, mask, label_pad_token_id):
        '''NOTE: This is where to modify for CRF.
        You need to finish the code to predict labels
        You can add input arguments.
        '''
        max_length = max([len(emission) for emission in emissions])
        _logits = self.crf.decode(emissions, mask)
        labels_list = [torch.tensor(logit + [label_pad_token_id] * (max_length - len(logit))) for logit in _logits]
        pred_labels = torch.stack(labels_list)
        return pred_labels

    def forward(self, hidden_states, attention_mask, labels=None, no_decode=False, label_pad_token_id=NER_PAD_ID):    
        '''NOTE: This is where to modify for CRF.
        You need to finish the code to compute loss and predict labels.
        '''
        emissions = self.layers(hidden_states)
        loss, pred_labels = None, None
        attention_mask = attention_mask.bool()

        if labels == None:
            pred_labels = self._pred_labels(emissions, attention_mask, label_pad_token_id)
        else:
            loss = -1 * self.crf(emissions, labels, attention_mask)
            if not no_decode:
                pred_labels = self._pred_labels(emissions, attention_mask, label_pad_token_id)

        return NEROutputs(loss, pred_labels)


def _group_ner_outputs(output1: NEROutputs, output2: NEROutputs):
    """ logits: [batch_size, seq_len] ==> [batch_size, seq_len, 2] """
    grouped_loss, grouped_logits = None, None

    if not (output1.loss is None or output2.loss is None):
        grouped_loss = (output1.loss + output2.loss) / 2

    if not (output1.logits is None or output2.logits is None):
        grouped_logits = torch.stack([output1.logits, output2.logits], dim=-1)

    return NEROutputs(grouped_loss, grouped_logits)
