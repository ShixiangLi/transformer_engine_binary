# 训练一个简单的copy任务
import math

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns

from CMAPSSDataset import get_data, get_features_dim
from train_utils import run_epoch, data_gen, SimpleLossCompute, run_distill_epoch
from transformer.binary.b_model import Model
from transformer.binary.config import ModelConfig


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_random_seed(42)  # 替换为你想要的随机种子值

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FD = '1'
batch_size = 1024
epochs = 100
patient = 50
p = 0
rmse_test = float("inf")
seq_len, feature_columns, d_model = get_features_dim(FD=FD)
mode = 'eval'
# 32bits
# Dataset   |   RMSE    |   hidden_dropout_prob
# FD001     |   13.11   |   0.1
# FD002     |   19.67   |   0.5
# FD003     |   11.85   |   0.1
# FD004     |   19.50   |   0.5

# 1bit_no_kd bwn    0.005/0.1
# Dataset   |   RMSE    |   hidden_dropout_prob
# FD001     |   17.35   |   0.1
# FD002     |   27.19   |   0.5
# FD003     |   16.70   |   0.1
# FD004     |   22.92   |   0.5

# 1bit_no_kd elastic    0.005/0.1
# Dataset   |   RMSE    |   hidden_dropout_prob
# FD001     |   16.24   |   0.1
# FD002     |   22.75   |   0.5
# FD003     |   15.72   |   0.1
# FD004     |   21.83   |   0.5

if mode == 'train':

    config = ModelConfig(
        num_attention_heads=4,
        hidden_size=16,
        input_bits=32,
        clip_init_val=2.5,
        quantize_act=True,
        weight_bits=32,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,  ####
        intermediate_size=64,
        num_hidden_layers=6,
        time_steps=30,
        max_position_embeddings=30,
        en_dropout=0.1,
        initializer_range=0.02,
        device=device,
        weight_layerwise=True,
        input_layerwise=True,
        weight_quant_method='bwn',
        input_quant_method='elastic',
        learnable_scaling=True,
        sym_quant_qkvo=True,
    )
    model = Model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optimizer, step_size=1200, gamma=0.1)

    for epoch in range(epochs):

        # 训练一个Batch
        run_epoch(
            data_gen(FD=FD, feature_columns=feature_columns,
                     sequence_length=seq_len, batch_size=batch_size, label='train'),
            model,
            SimpleLossCompute(criterion),
            optimizer,
            device=device,
            scheduler=lr_scheduler,
            mode="train",
            epoch=epoch
        )

        model.eval()
        feature, labels = get_data(FD=FD, feature_columns=feature_columns,
                                   sequence_length=seq_len, batch_size=batch_size, label='test')
        feature = torch.tensor(feature)
        feature_mask = torch.ones(size=(feature.size(0), 1, seq_len))
        labels = torch.tensor(labels)
        with torch.no_grad():
            encoded_layers, layer_atts, output = model.forward(feature.to(device), feature_mask.to(device))
        out = torch.clamp(output, max=1, min=0)
        rmse = math.sqrt(nn.MSELoss()(out.to(device), labels.to(device))) * 125

        if rmse < rmse_test:
            rmse_test = rmse
            p = 0
            if config.weight_bits == 1:
                print("FD00" + FD + " Epoch " + str(epoch) + " RMSE: " + str(rmse_test) + " Saving Model")
                torch.save(model, "./checkpoint/bin/" + str(config.input_quant_method) + "/FD00" + FD + "/model.pt")
            else:
                print("FD00" + FD + " Epoch " + str(epoch) + " RMSE: " + str(rmse_test) + " Saving Model")
                torch.save(model, "./checkpoint/full" + "/FD00" + FD + "/model.pt")
        else:
            p += 1
        if p > patient:
            break

    if config.weight_bits == 1:
        model = torch.load("./checkpoint/bin/" + str(config.input_quant_method) + "/FD00" + FD + "/model.pt")
    else:
        model = torch.load("./checkpoint/full" + "/FD00" + FD + "/model.pt")

    model.eval()
    feature, labels = get_data(FD=FD,
                               feature_columns=feature_columns,
                               sequence_length=seq_len,
                               batch_size=batch_size,
                               label='test')
    feature = torch.tensor(feature)
    feature_mask = torch.ones(size=(feature.size(0), 1, seq_len))
    labels = torch.tensor(labels)
    with torch.no_grad():
        encoded_layers, layer_atts, output = model.forward(feature.to(device), feature_mask.to(device))
    out = torch.clamp(output, max=1, min=0)
    print(math.sqrt(nn.MSELoss()(out.to(device), labels.to(device))) * 125)
    plt.plot(out.to('cpu').detach().numpy())
    plt.plot(labels.to("cpu").detach().numpy())
    plt.show()
else:
    config = ModelConfig(
        num_attention_heads=4,
        hidden_size=16,
        input_bits=1,
        clip_init_val=2.5,
        quantize_act=True,
        weight_bits=1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,  ####
        intermediate_size=64,
        num_hidden_layers=6,
        time_steps=30,
        max_position_embeddings=30,
        en_dropout=0.1,
        initializer_range=0.02,
        device=device,
        weight_layerwise=True,
        input_layerwise=True,
        weight_quant_method='bwn',
        input_quant_method='elastic',
        learnable_scaling=True,
        sym_quant_qkvo=True,
    )
    if config.weight_bits == 1:
        model = torch.load("./checkpoint/bin/" + str(config.input_quant_method) + "/FD00" + FD + "/model.pt")
    else:
        model = torch.load("./checkpoint/full/" + "/FD00" + FD + "/model.pt")
    feature, labels = get_data(FD=FD,
                               feature_columns=feature_columns,
                               sequence_length=seq_len,
                               batch_size=batch_size,
                               label='test')
    feature = torch.tensor(feature)
    feature_mask = torch.ones(size=(feature.size(0), 1, seq_len))
    labels = torch.tensor(labels)
    with torch.no_grad():
        encoded_layers, layer_atts, output, layer_atto = model.forward(feature.to(device), feature_mask.to(device))

    sns.set(style="whitegrid")
    for i, activation in enumerate(layer_atts):
        activation = activation.to('cpu').detach().numpy()
        np.save(f"atto_layer_{i + 1}.npy", activation)
        plt.figure(figsize=(8, 6))  # 设置绘图大小
        sns.histplot(activation.flatten(), bins=50, kde=True, color='b')  # 使用seaborn绘制直方图和核密度估计曲线
        plt.title(f"Activation Histogram - Layer  {i + 1}")  # 设置标题
        plt.xlabel("Activation Value")  # 设置X轴标签
        plt.ylabel("Density")  # 设置Y轴标签
        plt.tight_layout()  # 调整布局，防止文本重叠
        plt.show()

    out = torch.clamp(output, max=1, min=0)
    print(math.sqrt(nn.MSELoss()(out.to(device), labels.to(device))) * 125)
    plt.plot(out.to('cpu').detach().numpy())
    plt.plot(labels.to("cpu").detach().numpy())
    plt.show()
