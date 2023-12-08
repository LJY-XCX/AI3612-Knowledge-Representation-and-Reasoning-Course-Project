# Readme

## Running the Code

```
cd src
bash run_cmeee.sh
```


In the file run_cmeee.sh, you can choose the classifier using the TASK_ID variable, select the output and model save path with OUTPUT_DIR, and customize various arguments in the latter part of the file.

## Pretrained Models

The BERT pretrained model is locally loaded, while other models are downloaded from Hugging Face:

1. BART: fnlp/bart-base-chinese
2. Roberta: hfl/chinese-roberta-wwm-ext
3. GlobalPointer: xusenlin/cmeee-global-pointer
4. ErnieHealth: nghuyong/ernie-health-zh

Simply install the transformers package and ensure a smooth network connection. When calling the from_pretrained method, the program will automatically download pretrained models.

## Layer-wise Learning Rate Decay

Code related to layer-wise learning rate decay can be found in ./lr_schedule_layerwise.py. You can adjust the relevant parameters in plus_parser in run_cmeee.py to enable learning rate decay and customize decay rates.

## Data Augmentation

Code for synonym replacement is available in ./augmentation.py, and its specific application can be found in the InputExample class in ee_data. Adjust the relevant parameters in plus_parser in run_cmeee.py to enable or disable data augmentation.