#!/usr/bin/env python
# coding=utf-8
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import HfArgumentParser, TrainingArguments


def parse_args(json_file):
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    return parser.parse_json_file(json_file=json_file)


@dataclass()
class ModelArguments:
    """
    model arguments
    """
    model_config_file: str = field(metadata={"help": "path to model config json"})
    data_dir: str = field(metadata={"help": "tokenized data path"})
    vocab_file: str = field(metadata={"help": "tokenize_file"})
    pretrained_model_path: Optional[str] = field(default=None, metadata={"help": "pretrained model file"})

