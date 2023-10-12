from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from classifier import *

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