# Readme

## 代码运行方式

在src目录下运行

>bash run_cmeee.sh

在文件$\texttt{run_cmeee.sh}$中，可以通过$TASK\_ID$选择分类器，$OUTPUT\_DIR$选择输出及模型保存路径，文件的后半部分包含大量的auguments，可任意调整。

## 预训练模型

bert预训练模型从本地读取，其余模型从hugging face下载：
bart: <https://huggingface.co/fnlp/bart-base-chinese>
Roberta: <https://huggingface.co/hfl/chinese-roberta-wwm-ext>
GlobalPointer: <https://huggingface.co/xusenlin/cmeee-global-pointer>
ErnieHealth: <https://huggingface.co/nghuyong/ernie-health-zh>
只需安装transformers包且网络通畅，调用from_pretrained方法时，程序会自动下载预训练模型。

## 逐层学习率衰减

逐层学习率衰减的相关代码见./lr_schedule_layerwise.py。你可以调整run_cmeee.py中plus_parser的相关参数来决定是否启用学习率衰减以及衰减率的调整请使用。

## 数据增强

同义词替换的相关代码见./augmentation.py，具体应用方式见ee_data中的InputExample类。你可以调整run_cmeee.py中plus_parser的相关参数来决定是否启用数据增强。
