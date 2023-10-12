
import numpy as np

from typing import List, Union, NamedTuple, Tuple, Counter
from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        '''NOTE: You need to finish the code of computing f1-score.
        '''
        pred_entities = extract_entities(predictions)
        label_entities = extract_entities(labels)
        inter_number = 0
        pred_number = 0
        label_number = 0

        for i in range(len(pred_entities)):
            pred_set = set(pred_entities[i])
            label_set = set(label_entities[i])
            pred_number += len(pred_set)
            label_number += len(label_set)
            inter_number += len(set.intersection(pred_set, label_set))

        F1_score = 2 * inter_number / (pred_number + label_number)
        return { "f1": F1_score }


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        # '''NOTE: You need to finish the code of computing f1-score.
        # '''

        pred_entities1 = extract_entities(predictions[:, :, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, :, 1], for_nested_ner=True, first_labels=False)
        label_entities1 = extract_entities(labels1, for_nested_ner=True, first_labels=True)
        label_entities2 = extract_entities(labels2, for_nested_ner=True, first_labels=False)
        inter_number = 0
        pred_number = 0
        label_number = 0

        for i in range(len(pred_entities1)):
            pred_set = set.union(set(pred_entities1[i]), set(pred_entities2[i]))
            label_set = set.union(set(label_entities1[i]), set(label_entities2[i]))
            pred_number += len(pred_set)
            label_number += len(label_set)
            inter_number += len(set.intersection(pred_set, label_set))
        
        F1_score = 2 * inter_number / (pred_number + label_number)

        return { "f1": F1_score }


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    
    bs, lens = batch_labels_or_preds.shape

    def get_type(b_idx, start_idx, end_idx):

        entity_counter = Counter()
        for idx in range(start_idx, end_idx+1):
            entity_counter[id2label[batch_labels_or_preds[b_idx][idx]].split('-')[1]] += 1
        
        max_value = max(entity_counter.values())
        candidates = [k for k, v in entity_counter.items() if v == max_value]
        entity_type = sorted(candidates, key= lambda x:_LABEL_RANK[x], reverse=True)[0]
        return entity_type
    
    for b in range(bs):
        entities_list = []
        start_idx = -1
        end_idx = 0

        for i in range(lens):
            
            if id2label[batch_labels_or_preds[b][i]] in ['[PAD]', 'O']:  ##  NER_PAD and NO_ENT cases
                if start_idx != -1:
                    entities_list.append((start_idx, end_idx, get_type(b, start_idx, end_idx)))
                    start_idx = -1

            elif id2label[batch_labels_or_preds[b][i]][0] == 'B':  ##  The label is the beginning of an entity
                if start_idx != -1:
                    entities_list.append((start_idx, end_idx, get_type(b, start_idx, end_idx)))
                start_idx = i
                end_idx = i

            else:  ##  The label is inside an entity
                if start_idx != -1:
                    end_idx += 1

        batch_entities.append(entities_list)

    return batch_entities



if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-8:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-8:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
    