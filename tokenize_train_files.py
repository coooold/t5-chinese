#!/usr/bin/env python
# coding=utf-8
from utils import *
import numpy as np
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

train_data_path = 'data/train'
tokenized_data_path = 'data/tokenized'
vocab_file = 'data/vocab/clue_vocab.txt'


def build_file(train_data_file, tokenizer, tokenized_data_file):
    print('building ' + train_data_file)

    # 不重复分割
    # if os.path.exists(tokenized_data_file):
    #     return

    with open(train_data_file, 'r', encoding='utf8') as f:
        raw = f.read()
        articles = raw.replace("\r", '').split("\n\n")
        # 人民日报按照文章切割
        articles_len = len(articles)
        full_line = []

        for i in range(articles_len):
            print('process article {} of {}'.format(i, articles_len))
            content = articles[i]
            content = content.replace('\n', '[SEP]').strip()
            full_line.extend(tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(content)
            ))

        ids_np = np.array(full_line, dtype=np.int16)
        print('writing to ' + tokenized_data_file)
        ids_np.tofile(tokenized_data_file)


tokenizer = get_tokenizer(
    vocab_file=vocab_file
)
count = 1
for root, subdirs, files in os.walk(train_data_path):
    for file in files:
        train_data_file = train_data_path + '/' + file
        tokenized_data_file = tokenized_data_path + '/' + '{}.txt'.format(count)
        build_file(train_data_file=train_data_file,
                   tokenizer=tokenizer,
                   tokenized_data_file=tokenized_data_file)
