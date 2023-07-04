# 训练一个简单的copy任务
import math

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from CMAPSSDataset import get_data, get_features_dim
from train_utils import run_epoch, data_gen, SimpleLossCompute, run_distill_epoch
from transformer.binary.b_model import Model
from transformer.binary.config import ModelConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FD = '1'
batch_size = 1024
epochs = 2000
patient = 100
p = 0
rmse_test = float("inf")
seq_len, feature_columns, d_model = get_features_dim(FD=FD)

# 32bits
# Dataset   |   RMSE    |   Dropout
# FD001     |   13.96   |   0.1
# FD002     |   19.16   |   0.5
# FD003     |   11.12   |   0.1
# FD004     |   17.77   |   0.5

# 1bit_kd
# Dataset   |   RMSE    |   Dropout
# FD001     |   20.49   |   0.1
# FD002     |   29.05   |   0.5
# FD003     |   20.34   |   0.1
# FD004     |   27.85   |   0.5

config = ModelConfig(
    num_attention_heads=4,
    hidden_size=16,
    intermediate_size=64,
    weight_bits=1,
    input_bits=1,
    num_hidden_layers=6,
    hidden_dropout_prob=0.1)
model = Model(config).to(device)
teacher = torch.load("./checkpoint/full/FD00" + FD + "/model.pt")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
lr_scheduler = StepLR(optimizer, step_size=2500, gamma=0.5)

for epoch in range(epochs):
    model.train()
    # 训练一个Batch
    run_distill_epoch(
        data_gen(FD=FD, feature_columns=feature_columns,
                 sequence_length=seq_len, batch_size=batch_size, label='train'),
        teacher,
        model,
        SimpleLossCompute(criterion),
        optimizer,
        device=device,
        scheduler=lr_scheduler,
        mode="train"
    )

    model.eval()
    feature, labels = get_data(FD=FD, feature_columns=feature_columns,
                               sequence_length=seq_len, batch_size=batch_size, label='test')
    feature = torch.tensor(feature)
    feature_mask = torch.ones(size=(feature.size(0), 1, seq_len))
    labels = torch.tensor(labels)
    with torch.no_grad():
        _, _, out, _, _, _, _ = model.forward(feature.to(device), feature_mask.to(device))
    out = torch.clamp(out, max=1, min=0)
    rmse = math.sqrt(nn.MSELoss()(out.to(device), labels.to(device))) * 125

    if rmse < rmse_test:
        rmse_test = rmse
        p = 0
        torch.save(model, "./checkpoint/bin_kd/" + "FD00" + FD + "/model.pt")
    else:
        p += 1
    if p > patient:
        break

model = torch.load("./checkpoint/bin_kd/" + "FD00" + FD + "/model.pt")
model.eval()
# for name, module in model.named_modules():
#     if hasattr(module, 'weight_quantizer'):
#         if module.weight_bits == 1:
#             module.weight.data = module.weight_quantizer.apply(
#                 module.weight)
#         else:
#             module.weight.data = module.weight_quantizer.apply(
#                 module.weight,
#                 module.weight_clip_val,
#                 module.weight_bits, True,
#                 module.type)
feature, labels = get_data(FD=FD, feature_columns=feature_columns,
                           sequence_length=seq_len, batch_size=batch_size, label='test')
feature = torch.tensor(feature)
feature_mask = torch.ones(size=(feature.size(0), 1, seq_len))
labels = torch.tensor(labels)
with torch.no_grad():
    _, _, out, _, _, _, _ = model.forward(feature.to(device), feature_mask.to(device))
out = torch.clamp(out, max=1, min=0)
print(math.sqrt(nn.MSELoss()(out.to(device), labels.to(device))) * 125)
plt.plot(out.to('cpu').detach().numpy())
plt.plot(labels.to("cpu").detach().numpy())
plt.show()
