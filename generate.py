#!/usr/bin/env python
# coding=utf-8
from utils import *
from transformers import T5ForConditionalGeneration
import torch

args = {
    "vocab_file": "data/vocab/clue_vocab.txt",
    "pretrained_model_path": "data/model/checkpoint-60",
    "max_length": 512
}

tokenizer = get_tokenizer(
    vocab_file=args["vocab_file"]
)

model = T5ForConditionalGeneration.from_pretrained(args["pretrained_model_path"])

input_ids = tokenizer.encode("衣橱通向一个叫作<extra_id_0>的世界")
input_ids.extend(
    [0] * (args["max_length"] - len(input_ids))
)
inputs = torch.LongTensor(input_ids).unsqueeze(0)

out = model.generate(input_ids=inputs)
print(tokenizer.decode(out[0]))
