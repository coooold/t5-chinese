#!/usr/bin/env python
# coding=utf-8
from utils import *
from transformers import T5Config, T5ForConditionalGeneration
from transformers import (
    Trainer,
    set_seed,
)

(model_args, training_args) = parse_args(os.path.abspath("train_args.json"))

set_seed(training_args.seed)

tokenizer = get_tokenizer(
    vocab_file=model_args.vocab_file
)

# https://huggingface.co/transformers/model_doc/t5.html#training
model_config = T5Config.from_json_file(model_args.model_config_file)
model = T5ForConditionalGeneration(config=model_config)

# 计算参数数量
num_parameters = 0
for parameter in model.parameters():
    num_parameters += parameter.numel()
logger.info('number of parameters: {}'.format(num_parameters))

train_dataset = T5Dataset(model_config.n_positions,
                          tokenized_file_path=model_args.data_dir,
                          tokenizer=tokenizer)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset)

# 开始训练
trainer.train(
    model_path=model_args.pretrained_model_path
)
trainer.save_model()
