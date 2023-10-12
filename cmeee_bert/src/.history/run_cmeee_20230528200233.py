#!/user/bin/env python3
import os
import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List
import argparse

from sklearn.metrics import precision_recall_fscore_support
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention
from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, BertLayer, BartTokenizer, BartModel
from transformers import RobertaTokenizer, RobertaModel

from args import ModelConstructArgs, CBLUEDataArgs
from logger import get_logger
from ee_data import EE_label2id2, EEDataset, EE_NUM_LABELS1, EE_NUM_LABELS2, EE_NUM_LABELS, CollateFnForEE, \
    EE_label2id1, NER_PAD, EE_label2id
from model import *
from metrics import ComputeMetricsForNER, ComputeMetricsForNestedNER, extract_entities

from lr_scheduler_layerwise import get_optimizer_and_scheduler


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

MODEL_CLASS = {
    'linear': BertForLinearHeadNER, 
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
    'crf_nested': BertForCRFHeadNestedNER,
    'bart_linear': BartForLinearHeadNER,
    'bart_linear_nested': BartForLinearHeadNestedNER,
    'bart_crf': BartForCRFHeadNER,
    'roberta': Roberta,
    'ernie_linear' : ErnieForLinearHeadNER,
    'global_pointer': BertForGlobalPointer,
    'ernie_global_pointer': ErnieForGlobalPointer
}


def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    return logger, train_args, model_args, data_args

def get_model_with_tokenizer(model_args):

    if model_args.model_type == 'bart':
        if model_args.head_type == 'linear':
            model = BartForLinearHeadNER(hidden_size=768, num_labels=EE_NUM_LABELS, hidden_dropout=0.2)
        elif model_args.head_type == 'linear_nested':
            model = BartForLinearHeadNestedNER(hidden_size=768, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2, hidden_dropout=0.2)
        else:
            model = BartForCRFHeadNER(hidden_size=768, num_labels=EE_NUM_LABELS, hidden_dropout=0.2)
        tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

    elif model_args.model_type == 'roberta':
        model = Roberta(hidden_size=768, num_labels=EE_NUM_LABELS, hidden_dropout=0.2)
        tokenizer = RobertaTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    elif model_args.model_type == 'bert':
        model_class = MODEL_CLASS[model_args.head_type]
        if 'nested' not in model_args.head_type:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
        else:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2)
        tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    
        if "ernie_linear" in model_args.head_type:
            config = ErnieConfig.from_pretrained(model_args.model_path)
            tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
            
        model = model_class(config, num_labels1=EE_NUM_LABELS)

    elif "global" in model_args.head_type:
        if "ernie" not in model_args.head_type:
            ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}
            config = BertConfig.from_pretrained(model_args.model_path)
            model = model_class(config, num_labels1=len(ent2id), inner_dim=64)
        else:
            ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}
            config = ErnieConfig.from_pretrained(model_args.model_path)
            model = model_class(config, num_labels1=len(ent2id), inner_dim=64)
    
    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []

        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    plus_parser = argparse.ArgumentParser(description='Some additional arguments for CMeEE')
    plus_parser.add_argument('--layerwise', action='store_true', help='Whether to use layerwise learning rate')
    plus_parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate')
    plus_parser.add_argument('--augment', action='store_true', help='Whether to use data augmentation')


    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)

    for_nested_ner = 'nested' in model_args.head_type

    # ===== Get datasets =====
    if train_args.do_train:
        train_dataset = EEDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner, augment=train_args.augment)
        dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner, augment=train_args.augment)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()

    if train_args.layerwise:
        t_total = int(len(train_dataset) / train_args.per_device_train_batch_size / train_args.gradient_accumulation_steps * train_args.num_train_epochs)
        optimizer, scheduler = get_optimizer_and_scheduler(train_args, model, train_steps=t_total)
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler),
        )

    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
        )

    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        test_dataset = EEDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner, augment=False)
        logger.info(f"Testset: {len(test_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=for_nested_ner)


if __name__ == '__main__':
    main()
