
from audioop import bias
from typing import Optional
from unicodedata import bidirectional

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers import AutoModel, ErniePreTrainedModel, ErnieConfig, ErnieModel
from transformers.file_utils import ModelOutput

from classifier import *
from global_pointer import *

from ee_data import EE_label2id1, NER_PAD

NER_PAD_ID = EE_label2id1[NER_PAD]

def _group_ner_outputs(output1: NEROutputs, output2: NEROutputs):
    """ logits: [batch_size, seq_len] ==> [batch_size, seq_len, 2] """
    grouped_loss, grouped_logits = None, None

    if not (output1.loss is None or output2.loss is None):
        grouped_loss = (output1.loss + output2.loss) / 2

    if not (output1.logits is None or output2.logits is None):
        grouped_logits = torch.stack([output1.logits, output2.logits], dim=-1)

    return NEROutputs(grouped_loss, grouped_logits)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss
class ErnieForLinearHeadNER(ErniePreTrainedModel):
    config_class = ErnieConfig
    base_model_prefix = "ernie"

    def __init__(self, config: ErnieConfig, num_labels1: int):
        super().__init__(config)
        self.config = config
        self.bert = ErnieModel.from_pretrained("nghuyong/ernie-health-zh")
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

class ErnieForGlobalPointer(ErniePreTrainedModel):
    config_class = ErnieConfig
    base_model_prefix = "ernie"
    def __init__(self, config: ErnieConfig, num_labels1: int, inner_dim, RoPE=True):
        #encodr: RoBerta-Large as encoder
        #inner_dim: 64
        #ent_type_size: ent_cls_num
        super().__init__(config)
        self.encoder = ErnieModel.from_pretrained("nghuyong/ernie-health-zh")
        self.config=config
        self.inner_dim = inner_dim
        self.hidden_size = self.encoder.config.hidden_size
        self.RoPE = RoPE
        self.num_labels = num_labels1
        self.loss_fct = loss_fun
        
        self.metric = MetricsCalculator()

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.num_labels *2) #原版的dense2是(inner_dim * 2, ent_type_size * 2)
        self.init_weights()
        
    def _pred_labels(self, _logits):
        return torch.argmax(_logits, dim=-1)
    

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, no_decode = False):
        no_decode = False
        #context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        context_outputs = self.encoder(input_ids, attention_mask)
        last_hidden_state = context_outputs.last_hidden_state
        outputs = self.dense_1(last_hidden_state)
        qw, kw = outputs[...,::2], outputs[..., 1::2] #从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None] #logits[:, None] 增加一个维度
        logits = self.add_mask_tril(logits, mask=attention_mask)
        if labels is None:
            pred_labels = self._pred_labels(logits)   
        else:
            labels = labels.long()
            labels = labels.view(labels.shape[0], logits.shape[1], logits.shape[2], logits.shape[3])
            loss = self.loss_fct(logits, labels)
            f1 = self.metric.get_sample_f1(logits, labels)
            logits = logits.contiguous().view(logits.shape[0],-1)
            logits = logits.detach()
            logits = torch.gt(logits, 0)
            logits = logits.to(torch.int8)
            output = NEROutputs(loss, logits)
            
        return output

    
