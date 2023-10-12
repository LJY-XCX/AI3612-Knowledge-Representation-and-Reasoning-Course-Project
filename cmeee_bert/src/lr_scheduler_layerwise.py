
import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List

from sklearn.metrics import precision_recall_fscore_support
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup
from logger import get_logger

def get_layerwise_grouped_parameters(args, model):

    no_decay = ["bias", "LayerNorm.weight"]
    num_layers = model.config.num_hidden_layers

    ## set grouped parameters for classifier and pooler ##
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if "classifer" in n or "pooler" in n],
        "weight_decay": 0.0,
        "lr": args.learning_rate,
    },
    ]

    ## set grouped parameters layerwise ##
    
    layers = [getattr(model, 'bert').embeddings] + list(getattr(model, 'bert').encoder.layer)
    layers.reverse()
    lr = args.learning_rate

    for layer in layers:
        lr *= args.lr_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            }
        ]

    return optimizer_grouped_parameters


def get_optimizer_and_scheduler(args, model, train_steps, correct_bias=True):
    optimizer_grouped_parameters = get_layerwise_grouped_parameters(args, model)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon,
                      correct_bias=correct_bias)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=train_steps)
    return optimizer, scheduler
    

