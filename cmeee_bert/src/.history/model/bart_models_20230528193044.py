
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
from classifier import *

class BartForLinearHeadNER(PreTrainedModel):
    def __init__(self, hidden_size, num_labels, hidden_dropout):
        config = BartConfig(vocab_size=21128, max_position_embeddings=512, pad_token_id=0)
        super().__init__(config)
        self.bart = BartModel.from_pretrained("fnlp/bart-base-chinese")
        self.classifier = LinearClassifier(hidden_size, num_labels, hidden_dropout)
        #self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, labels=None, labels2=None, output_attentions=None, output_hidden_states=None, return_dict=None,
            no_decode=False,
    ):
        # print(input_ids.shape)
        sequence_output = self.bart(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # decoder_input_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        # print(sequence_output.shape)
        output = self.classifier.forward(sequence_output, labels, no_decode=no_decode)
        return output    
class BartForLinearHeadNestedNER(PreTrainedModel):
    def __init__(self, hidden_size, num_labels1, num_labels2, hidden_dropout):
        config = BartConfig(vocab_size=21128, max_position_embeddings=512, pad_token_id=0)
        super().__init__(config)
        self.bart = BartModel.from_pretrained("fnlp/bart-base-chinese")
        self.classifier1 = LinearClassifier(hidden_size, num_labels1, hidden_dropout)
        self.classifier2 = LinearClassifier(hidden_size, num_labels2, hidden_dropout)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, labels=None, labels2=None, output_attentions=None, output_hidden_states=None, return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bart(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            decoder_input_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        # print(sequence_output.shape)
        output1 = self.classifier1.forward(sequence_output, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, labels2, no_decode=no_decode)
        return _group_ner_outputs(output1, output2)  
