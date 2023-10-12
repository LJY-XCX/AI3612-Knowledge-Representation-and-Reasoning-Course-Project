import json


def calculate_f1_score(ground_truth_file, predicted_file):
    # Load the ground truth and prediction data from the JSON files
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.loads(f.read())
    with open(predicted_file, 'r') as f:
        predicted_data = json.loads(f.read())
    print(ground_truth_data.__len__(), predicted_data.__len__())

    TP = 0  # True Positives
    ground_length = 0
    predicted_length = 0

    for i in range(len(ground_truth_data)):
        if i > len(predicted_data) - 1:
            break

        ground_truth_entities = ground_truth_data[i]['entities']

        predicted_entities = predicted_data[i]['entities']

        ground_length += sum([len(ground_entity) for ground_entity in ground_truth_entities])
        predicted_length += sum([len(predicted_entity) for predicted_entity in predicted_entities])

        for predicted_entity in predicted_entities:
            for ground_entity in ground_truth_entities:
                if predicted_entity['entity'] == ground_entity['entity'] and \
                        predicted_entity['type'] == ground_entity['type'] and \
                        predicted_entity['start_idx'] == ground_entity['start_idx'] and \
                        predicted_entity['end_idx'] == ground_entity['end_idx']:
                    TP += 1
                    break

    f1 = 2 * TP / (ground_length + predicted_length)

    return f1

if __name__ == '__main__':
    print(calculate_f1_score(predicted_file='../ckpts/chatgpt_api/CMeEE_consistency.json',
                             ground_truth_file='../data/CBLUEDatasets/CMeEE/select_dev.json'))