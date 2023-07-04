import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.en_dropout)
        self.d_model = config.hidden_size
        self.max_len = config.max_position_embeddings
        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(self.max_len, self.d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, self.max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term_1 = torch.exp(
            torch.arange(0, self.d_model, 2) * -(math.log(float(config.time_steps)) / self.d_model)
        )
        div_term_2 = torch.exp(
            torch.arange(1, self.d_model, 2) * -(math.log(float(config.time_steps)) / self.d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term_1)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term_2)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
