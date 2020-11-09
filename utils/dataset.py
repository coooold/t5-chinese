#!/usr/bin/env python
# coding=utf-8
from torch.utils.data.dataset import Dataset
import numpy as np
from random import random, randint, shuffle
from math import ceil
import os


class T5Dataset(Dataset):
    def __init__(self,
                 max_length,
                 tokenized_file_path,
                 tokenizer
                 ):
        self.extra_ids = []
        for i in range(100):
            self.extra_ids.append(
                tokenizer.convert_tokens_to_ids('<extra_id_{}>'.format(i))
            )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.span = 5  # 遮蔽的长度
        for f in self.scan_files(tokenized_file_path):
            print("load training {}".format(f))
            raw = np.fromfile(f, dtype=np.int16).tolist()
            self.data += raw

    @staticmethod
    def scan_files(tokenized_data_path: str):
        """扫描目录，读取所有文件"""
        train_files = []
        for root, subdirs, files in os.walk(tokenized_data_path):
            for file in files:
                train_files.append(tokenized_data_path + '/' + file)
        # shuffle(train_files)
        return train_files

    def __len__(self):
        return (len(self.data) - self.max_length) // self.max_length

    def __getitem__(self, i):
        pos = self.max_length * i
        sample = self.data[pos: pos + self.max_length - 2]
        return self.process_sample(sample)

    def process_sample(self, sample):
        """
        处理段落，采用teacher-force style，有监督训练数据
        https://huggingface.co/transformers/model_doc/t5.html#training
        """
        spans = np.array_split(sample, ceil(len(sample) / 5))

        input_ids = []
        target_ids = []
        input_extra_id_idx = 0
        label_extra_id_idx = 0
        for span in spans:
            is_mask = randint(0, 99) < 15  # 15%概率遮蔽
            if is_mask:
                input_ids.append(self.extra_ids[input_extra_id_idx])
                input_extra_id_idx += 1
                target_ids.extend(span)
            else:
                target_ids.append(self.extra_ids[label_extra_id_idx])
                label_extra_id_idx += 1
                input_ids.extend(span)

        # attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        input_ids.extend(
            [0] * (self.max_length - len(input_ids))
        )
        # target_attention_mask = [1] * len(target_ids) + [0] * (self.max_length - len(target_ids))
        target_ids.extend(
            [0] * (self.max_length - len(target_ids))
        )

        return {
            'input_ids': input_ids,
            'labels': target_ids
        }
