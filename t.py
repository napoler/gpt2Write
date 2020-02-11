import pytorch_transformers
import torch
import os
import json
import random
import numpy as np
import argparse

from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder
import pre_process_data as ppd
from tokenizations import tokenization_bert

def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    if ppd.is_default_file_type():  # 是否采用默认json类型，默认编码为utf-8
        if ppd.DEFAULT_FILE_TYPE in data_path:
            with open(data_path, 'r', encoding='utf8') as f:
                print('reading lines')
                lines = json.load(f)
                lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        else:
            raise Exception("请使用json文件类型，或者自定义文件类型，请看pre_process_data.py文件load方法")
    else:  # 自定义数据源的，调用pre_process_data.py中的load方法
        lines = ppd.load()
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')
# full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab_small_terry_ai.txt')
full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab_small_terry_ai.txt')
 
build_files("data/train.json",'temp/',100,full_tokenizer,64)

line=" [TT] 【宠物狗/狗狗品种/宠物狗价格/宠物狗品种图片大全】- 赶集网 [/TT]    * **泰迪犬** [PAD]  [SEP] 英文名：toy poodl "
print(full_tokenizer.tokenize(text=line))

print(full_tokenizer.convert_tokens_to_ids('[PT]'))