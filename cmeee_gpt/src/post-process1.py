import json
import re

## Adjust the start index and end index of predicted entities

entity_type_dict = {'疾病': 'dis', '临床表现': 'sym', '医疗程序': 'pro', '医疗设备': 'equ', '药物': 'dru', '医学检验项目': 'ite',
                    '身体': 'bod', '科室': 'dep', '微生物类': 'mic'}

with open("/mnt/d/something_useful/learning/class/Knowledge/cmeee/ckpts/chatgpt_api/CMeEE_consistency.json", 'r') as f:
    origin_json = json.load(f)

# with open("/mnt/d/something_useful/learning/class/Knowledge/cmeee/data/CBLUEDatasets/CMeEE/select_dev.json", 'r') as f:
#     label_json = json.load(f)

# print(len(origin_json), len(label_json))
for i in range(150):
    ## Ensure that the length of predicted file and label file are equal
    # if 'entities' not in origin_json[i]:
    #     origin_json[i]['entities'] = []
    # if origin_json[i]['text'] != label_json[i]['text']:
    #     origin_json.insert(i, {'text': label_json[i]['text'], 'entities': []})
    #     assert origin_json[i]['text'] == label_json[i]['text']
    origin_text = origin_json[i]['text']
    origin_entities = origin_json[i]['entities']
    new_entities = []
    exist_entity_names = []
    for origin_entity in origin_entities:
        entity_name = origin_entity['entity']
        entity_type = origin_entity['type']
        if entity_name in exist_entity_names:
            continue
        exist_entity_names.append(entity_name)
        for match in re.finditer(entity_name, origin_text):
            start_idx = match.start()
            end_idx = match.end() - 1
            new_entities.append({'entity': entity_name, 'type': entity_type, 'start_idx': start_idx, 'end_idx': end_idx})
    origin_json[i]['entities'] = new_entities


with open("/mnt/d/something_useful/learning/class/Knowledge/cmeee/ckpts/chatgpt_api/CMeEE_consistency_post.json", "w", encoding="utf8") as f:
    json.dump(origin_json, f, indent=2, ensure_ascii=False)