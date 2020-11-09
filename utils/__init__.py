#!/usr/bin/env python
# coding=utf-8
import warnings
import os
import logging
import torch
from transformers import T5Tokenizer
from transformers import BertTokenizer
from .dataset import T5Dataset
from .args import parse_args

# 屏蔽 tf 无效日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_tokenizer(vocab_file):
    tokenizer = BertTokenizer(
        vocab_file=vocab_file,
        do_basic_tokenize=True
    )

    special_tokens_dict = {'additional_special_tokens': ["[SEP]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer
