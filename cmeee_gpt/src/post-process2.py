import json
import re

## Remove redundant entities

entity_type_dict = {'疾病': 'dis', '临床表现': 'sym', '医疗程序': 'pro', '医疗设备': 'equ', '药物': 'dru', '医学检验项目': 'ite',
                    '身体': 'bod', '科室': 'dep', '微生物类': 'mic'}

with open("/mnt/d/something_useful/learning/class/Knowledge/cmeee/ckpts/chatgpt_api/CMeEE_consistency_post.json", 'r') as f:
    origin_json = json.load(f)

# assert len(origin_json) == 500
for idx in range(150):
    entities = origin_json[idx]['entities']
    text = origin_json[idx]['text']
    i = 0
    
    while i < len(entities):
        
        j = 0
        while j < len(entities):
            
            if j == i: 
                j += 1
                continue
            if entities[i]['entity'] in entities[j]['entity'] and entities[i]['type'] != 'bod' and int(entities[i]['start_idx']) >= int(entities[j]['start_idx']) and int(entities[i]['end_idx']) <= int(entities[j]['end_idx']):

                if  not ((int(entities[i]['start_idx']) == int(entities[j]['start_idx'])) and (int(entities[i]['end_idx']) == int(entities[j]['end_idx']))):
                    print(entities[i])
                    print(entities[j])
                    del entities[i]
                    i -= 1 
                    break
            
            j += 1
        i += 1

    origin_json[idx]['entities'] = entities

with open("/mnt/d/something_useful/learning/class/Knowledge/cmeee/ckpts/chatgpt_api/CMeEE_consistency_post2.json", "w", encoding="utf8") as f:
    json.dump(origin_json, f, indent=2, ensure_ascii=False)