#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021-9-27 13:51
# @Author : lauqasim
# @Software: PyCharm

import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import DataLoader
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
# Preparing train data
# Chinese or Tibetan

# reference https://towardsdatascience.com/how-to-build-a-wordpiece-tokenizer-for-bert-f505d97dddbb to train tokenizer
# initialize
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)
# and train
tokenizer.train(files=['all.bo1000'], vocab_size=30_522, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
# save model dir
dirs = './bert-botok'
if not os.path.exists(dirs):
    os.mkdir('./bert-botok')
tokenizer.save_model('./bert-botok')
# initialize the tokenizer using the tokenizer we initialized and saved to file
tokenizer = BertTokenizer.from_pretrained('./bert-botok', max_len=512)

# reference https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
# and https://www.youtube.com/watch?v=35Pdoyi6ZoQ
def mlm(tensor):
    # create random array of floats with equal dims to input_ids
    rand = torch.rand(tensor.shape)
    # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    mask_arr = (rand < 0.15) * (tensor > 2)
    # loop through each row in input_ids tensor (cannot do in parallel)
    for i in range(tensor.shape[0]):
        # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        # mask input_ids
        tensor[i, selection] = 4 # our custom [MASK] token == 4
    return tensor

input_ids = []
mask = []
labels = []

with open('all.bo1000', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    # Then we encode our data using the tokenizer
    # making sure to include key parameters like max_length, padding, and truncation.
    sample = tokenizer(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    labels.append(sample.input_ids)
    mask.append(sample.attention_mask)
    input_ids.append(mlm(sample.input_ids.detach().clone()))

input_ids = torch.cat(input_ids)
mask = torch.cat(mask)
labels = torch.cat(labels)

encodings = {
    'input_ids': input_ids,
    'attention_mask': mask,
    'labels': labels
}

# Building the DataLoader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings
    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}
dataset = Dataset(encodings)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
from transformers import BertConfig
# Initializing the Model
config = BertConfig(
    vocab_size=30_522,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)
# Then, we import and initialize our RoBERTa model with a language modeling (LM) head.
from transformers import BertForMaskedLM
model = BertForMaskedLM(config)
# Training Preparation
# Setup GPU/CPU usage.
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Here, I use cpu.
device = torch.device('cpu')
# and move our model over to the selected device
model.to(device)

from transformers import AdamW

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

epochs = 2
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)

        # extract loss
        # I meet a error, so I make a change.
        # loss = outputs.loss
        loss = outputs[0]
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        # save pre-train model
        model.save_pretrained('./bert-botok')

# test, but tok
from transformers import pipeline
fill = pipeline('fill-mask', model='./bert-botok', tokenizer='./bert-botok')
# ཤེལ་ཕོར་ནང་དུ་ཆུ་ཁོལ་བླུགས་ན་ཤེལ་ཕོར་ཆག་ངེས།
print(fill(f'1. ཞིང་ལས་དང་གྲོང་གསེབ་ལ་མ་དངུལ་གཏོང་ཤུགས་སྔར་བས་ཆེ་རུ་གཏོང་དགོ{fill.tokenizer.mask_token}།'))
# result [{'sequence': '[CLS] 1. ཞང ་ ལས ་ དང ་ གང ་ གསབ ་ ལ ་ མ ་ དངལ ་ གཏང ་ ཤགས ་ སར ་ བས ་ ཆ ་ ར ་ གཏང ་ དག ། [SEP]', 'score': 0.8536086678504944, 'token': 1401, 'token_str': '[PAD]'}, {'sequence': '[CLS] 1. ཞང ་ ལས ་ དང ་ གང ་ གསབ ་ ལ ་ མ ་ དངལ ་ གཏང ་ ཤགས ་ སར ་ བས ་ ཆ ་ ར ་ གཏང ་ དག ་ ། [SEP]', 'score': 0.010686839930713177, 'token': 35, 'token_str': '་'}, {'sequence': '[CLS] 1. ཞང ་ ལས ་ དང ་ གང ་ གསབ ་ ལ ་ མ ་ དངལ ་ གཏང ་ ཤགས ་ སར ་ བས ་ ཆ ་ ར ་ གཏང ་ དག པ ། [SEP]', 'score': 0.003326073521748185, 'token': 62, 'token_str': 'པ'}, {'sequence': '[CLS] 1. ཞང ་ ལས ་ དང ་ གང ་ གསབ ་ ལ ་ མ ་ དངལ ་ གཏང ་ ཤགས ་ སར ་ བས ་ ཆ ་ ར ་ གཏང ་ དག ས ། [SEP]', 'score': 0.003296012757346034, 'token': 77, 'token_str': 'ས'}, {'sequence': '[CLS] 1. ཞང ་ ལས ་ དང ་ གང ་ གསབ ་ ལ ་ མ ་ དངལ ་ གཏང ་ ཤགས ་ སར ་ བས ་ ཆ ་ ར ་ གཏང ་ དགའ ། [SEP]', 'score': 0.003292103298008442, 'token': 119, 'token_str': '##འ'}]

# test, not tok
# reference https://huggingface.co/transformers/usage.html#masked-language-modeling
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-botok")
model = AutoModelWithLMHead.from_pretrained("bert-botok")

sequence = f'1. ཞིང་ལས་དང་གྲོང་གསེབ་ལ་མ་དངུལ་གཏོང་ཤུགས་སྔར་བས་ཆེ་རུ་གཏོང་དགོ{tokenizer.mask_token}།'

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
# result
# 1. ཞིང་ལས་དང་གྲོང་གསེབ་ལ་མ་དངུལ་གཏོང་ཤུགས་སྔར་བས་ཆེ་རུ་གཏོང་དགོ[PAD]།
# 1. ཞིང་ལས་དང་གྲོང་གསེབ་ལ་མ་དངུལ་གཏོང་ཤུགས་སྔར་བས་ཆེ་རུ་གཏོང་དགོ་།
# 1. ཞིང་ལས་དང་གྲོང་གསེབ་ལ་མ་དངུལ་གཏོང་ཤུགས་སྔར་བས་ཆེ་རུ་གཏོང་དགོཔ།
# 1. ཞིང་ལས་དང་གྲོང་གསེབ་ལ་མ་དངུལ་གཏོང་ཤུགས་སྔར་བས་ཆེ་རུ་གཏོང་དགོས།
# 1. ཞིང་ལས་དང་གྲོང་གསེབ་ལ་མ་དངུལ་གཏོང་ཤུགས་སྔར་བས་ཆེ་རུ་གཏོང་དགོ##འ།

