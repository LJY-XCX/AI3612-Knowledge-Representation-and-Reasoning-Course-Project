
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


class BertForLinearHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, labels, no_decode=no_decode)

        return output


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


class BertForLinearHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)

        self.classifier1 = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = LinearClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        '''NOTE: This is where to modify for Nested NER.

        Use the above function _group_ner_outputs for combining results.

        '''
        output1 = self.classifier1.forward(sequence_output, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, labels2, no_decode=no_decode)
        return _group_ner_outputs(output1, output2)

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


class BertForCRFHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.classifier = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        
        return output
    
class BartForCRFHeadNER(PreTrainedModel):
    def __init__(self, hidden_size, num_labels, hidden_dropout):
        config = BartConfig(vocab_size=21128, max_position_embeddings=512, pad_token_id=0)
        super().__init__(config)
        self.bart = BartModel.from_pretrained("fnlp/bart-base-chinese")
        self.classifier = CRFClassifier(hidden_size, num_labels, hidden_dropout)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, labels=None, labels2=None, output_attentions=None, output_hidden_states=None,
            return_dict=None, no_decode=False,
    ):
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
        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        return output  

class BertForCRFHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)

        self.classifier1 = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output1 = self.classifier1.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, attention_mask, labels2, no_decode=no_decode)
        return _group_ner_outputs(output1, output2)

# from lightningnlp.task.named_entity_recognition import NerPipeline
class GlobalPointer(PreTrainedModel):
    def __init__(self, hidden_size, num_labels, hidden_dropout):
        self.model = NerPipeline(model_name_or_path="xusenlin/cmeee-global-pointer", model_name="global-pointer", model_type="bert")
        self.classifier = LinearClassifier(hidden_size, num_labels, hidden_dropout)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, labels=None, labels2=None, output_attentions=None, output_hidden_states=None,
            return_dict=None, no_decode=False,
    ):
        sequence_output = self.bart(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # decoder_input_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,BertForGlobalPointer
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        return output

from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
class Roberta(nn.Module):
    def __init__(self, hidden_size, num_labels, hidden_dropout):
        # config = RobertaConfig(vocab_size=21128, max_position_embeddings=512, pad_token_id=0)
        super().__init__()
        self.robert = RobertaModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.classifier = CRFClassifier(hidden_size, num_labels, hidden_dropout)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, labels=None, labels2=None, output_attentions=None, output_hidden_states=None,
            return_dict=None, no_decode=False,
    ):
        sequence_output = self.robert(
            input_ids,
            # attention_mask=attention_mask,
            # # token_type_ids=token_type_ids,
            # # decoder_input_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )[0]
        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        return output  

class BertForCRFHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)

        self.classifier1 = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output1 = self.classifier1.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, attention_mask, labels2, no_decode=no_decode)
        return _group_ner_outputs(output1, output2)
