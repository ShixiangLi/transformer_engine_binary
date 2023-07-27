import torch
from transformers import BertTokenizer, BertModel
from bertviz import head_view
model_name = 'bert-base-chinese'  # 使用中文预训练的BERT模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)
text = "我喜欢使用BertViz来可视化注意力头。"
tokens = tokenizer.tokenize(text)
input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
outputs = model(input_ids)
attention = outputs[-1]  # 获取注意力头的输出
head_view(attention, tokens)
