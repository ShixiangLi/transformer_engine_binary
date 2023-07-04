# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2022.09.25 - Add distill_attn argument for removing attention distillation
#              Meta Platforms, Inc. <zechunliu@fb.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os

import torch
import logging

from torch import nn
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME, MISC_NAME
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from transformer.optimization import BertAdam
from helper import *
from utils_glue import *
import numpy as np
import pickle

from CMAPSSDataset import get_data, get_features_dim

logging.basicConfig(level=logging.INFO)


class Batch:
    """
    定义一个Batch，来存放一个batch的src，tgt，src_mask等对象。
    方便后续的取用
    """

    def __init__(self, src, tgt=None):  # 2 = <blank>
        """
        src: 和EncoderDecoder#forward中的那个src一致。
             未进行word embedding的句子，例如`[[ 0, 5, 4, 6, 1, 2, 2 ]]`
             上例shape为(1, 7)，即batch size为1，句子大小为7。其中0为bos，
             1为eos, 2为pad

        tgt: 和src类似。是目标句子。
        """
        self.src = src

        """
        构造src_mask：就是将src中pad的部分给盖住，因为这些不属于句子成分，不应该参与计算。
                     例如，src为[[ 0, 5, 4, 6, 1, 2, 2 ]]，则src_mask则为：
                     [[[ True, True, True, True, True, False, False ]]]。因为最后两个2(pad)
                     不属于句子成分。（“<bos>”、“<eos>”和“<unk>”是要算作句子成分的）
        这里unsqueeze一下是因为后续是要对Attention中的scores进行mask，而scores的len(shape)=3,
        为了与scores保持一致，所以unsqueeze(-2)一下。具体可参考attention函数中的注释。
        """
        self.src_mask = torch.ones(size=(src.size(0), 1, src.size(1)))
        self.tgt = tgt


class KDLearner(object):
    def __init__(self, config, student_model, teacher_model=None):
        self.config = config
        self.device = config.device
        self.student_model = student_model
        self.teacher_model = teacher_model

    def build(self, lr=None):
        param_optimizer = list(self.student_model.named_parameters())
        self.clip_params = {}
        for k, v in param_optimizer:
            if 'clip_' in k:
                self.clip_params[k] = v

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and not 'clip_' in n)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and not 'clip_' in n)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.clip_params.items()], 'lr': self.config.clip_lr,
             'weight_decay': self.config.clip_wd},
        ]

        learning_rate = self.config.learning_rate if not lr else lr
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        logging.info("Optimizer prepared.")

    def _do_eval(self, model, FD):
        model.eval()
        seq_len, feature_columns, d_model = get_features_dim(FD=FD)
        batch_size = 1024
        feature, labels = get_data(FD=FD, feature_columns=feature_columns,
                                   sequence_length=seq_len, batch_size=batch_size, label='test')
        feature = torch.tensor(feature)
        feature_mask = torch.ones(size=(feature.size(0), 1, seq_len))
        labels = torch.tensor(labels)
        with torch.no_grad():
            _, _, out, _, _, _, _ = model.forward(feature.to(self.config.device), feature_mask.to(self.config.device))
        out = torch.clamp(out, max=1, min=0)
        rmse = math.sqrt(nn.MSELoss()(out.to(self.config.device), labels.to(self.config.device))) * 125

        return rmse

    def evaluate(self, FD):
        """ Evalutaion of checkpoints from models/. directly use config.student_model """

        self.student_model.eval()
        result = self._do_eval(self.student_model, FD)

        logging.info("***** Running evaluation, Task: %s *****" % ("FD00" + FD))
        logging.info("***** Eval results, Task: %s *****" % ("FD00" + FD))
        logging.info("  RMSE = %s" % str(result))

    def train(self, train_examples, task_name, output_mode, eval_labels, num_labels,
              train_dataloader, eval_dataloader, eval_examples, tokenizer, mm_eval_labels, mm_eval_dataloader):
        """ quant-aware pretraining + KD """

        loss_mse = MSELoss()

        self.teacher_model.eval()
        teacher_results = self._do_eval(self.teacher_model, task_name, eval_dataloader, output_mode, eval_labels,
                                        num_labels)
        logging.info("Teacher network evaluation")
        for key in sorted(teacher_results.keys()):
            logging.info("  %s = %s", key, str(teacher_results[key]))

        self.teacher_model.train()

        global_step = self.prev_global_step
        best_dev_acc = 0.0
        output_eval_file = os.path.join(self.config.output_dir, "eval_results.txt")

        logging.info("***** Running training, Task: %s, Job id: %s*****" % (self.config.task_name, self.config.job_id))
        logging.info(" Distill rep: %d, Distill attn: %d, Distill logit: %d" % (
            self.config.distill_rep, self.config.distill_attn, self.config.distill_logit))
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", self.config.batch_size)
        logging.info("  Num steps = %d", self.num_train_optimization_steps)

        global_tr_loss = 0
        for epoch_ in range(self.config.num_train_epochs):

            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):

                self.student_model.train()

                batch = tuple(t.to(self.device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.
                rep_loss_layerwise = []
                att_loss_layerwise = []

                student_logits, student_atts, student_reps = self.student_model(input_ids, segment_ids, input_mask)

                if self.config.distill_logit or self.config.distill_rep or self.config.distill_attn:

                    with torch.no_grad():
                        teacher_logits, teacher_atts, teacher_reps = self.teacher_model(input_ids, segment_ids,
                                                                                        input_mask)

                    loss = 0.
                    if self.config.distill_logit:
                        cls_loss = soft_cross_entropy(student_logits / self.config.temperature,
                                                      teacher_logits / self.config.temperature)
                        loss += cls_loss
                        tr_cls_loss += cls_loss.item()

                    if self.config.distill_rep or self.config.distill_attn:
                        for student_att, teacher_att in zip(student_atts, teacher_atts):
                            student_att = torch.where(student_att <= -1e2,
                                                      torch.zeros_like(student_att).to(self.device),
                                                      student_att)
                            teacher_att = torch.where(teacher_att <= -1e2,
                                                      torch.zeros_like(teacher_att).to(self.device),
                                                      teacher_att)

                            tmp_loss = loss_mse(student_att, teacher_att)
                            att_loss += tmp_loss
                            att_loss_layerwise.append(tmp_loss.item())

                        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                            tmp_loss = loss_mse(student_rep, teacher_rep)
                            rep_loss += tmp_loss
                            rep_loss_layerwise.append(tmp_loss.item())

                        tr_att_loss += att_loss.item()
                        tr_rep_loss += rep_loss.item()

                        if self.config.distill_rep:
                            loss += rep_loss
                        if self.config.distill_attn:
                            loss += att_loss

                else:
                    if output_mode == "classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(student_logits, label_ids.view(-1))
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                global_tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                # evaluation and save model
                if global_step % self.config.eval_step == 0 or \
                        global_step == len(train_dataloader) - 1:

                    logging.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logging.info("  Num examples = %d", len(eval_examples))
                    logging.info(f"  Previous best = {best_dev_acc}")

                    loss = tr_loss / (step + 1)
                    global_avg_loss = global_tr_loss / (global_step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    self.student_model.eval()
                    result = self._do_eval(self.student_model, task_name, eval_dataloader, output_mode, eval_labels,
                                           num_labels)
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss
                    result['global_loss'] = global_avg_loss

                    preds = student_logits.detach().cpu().numpy()
                    train_label = label_ids.cpu().numpy()
                    if output_mode == "classification":
                        preds = np.argmax(preds, axis=1)
                    elif output_mode == "regression":
                        preds = np.squeeze(preds)
                    result['train_batch_acc'] = list(compute_metrics(task_name, preds, train_label).values())[0]

                    if self.config.distill_rep or self.config.distill_attn:
                        logging.info("embedding layer rep_loss: %.8f" % (rep_loss_layerwise[0]))
                        rep_loss_layerwise = rep_loss_layerwise[1:]
                        for lid in range(len(rep_loss_layerwise)):
                            logging.info("layer %d rep_loss: %.8f" % (lid + 1, rep_loss_layerwise[lid]))
                            logging.info("layer %d att_loss: %.8f" % (lid + 1, att_loss_layerwise[lid]))

                    result_to_file(result, output_eval_file)

                    save_model = False

                    if task_name in acc_tasks and result['acc'] > best_dev_acc:
                        best_dev_acc = result['acc']
                        save_model = True

                    if task_name in corr_tasks and result['corr'] > best_dev_acc:
                        best_dev_acc = result['corr']
                        save_model = True

                    if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                        best_dev_acc = result['mcc']
                        save_model = True

                    if save_model:
                        self._save()

                        if task_name == "mnli":
                            logging.info('MNLI-mm Evaluation')
                            result = self._do_eval(self.student_model, 'mnli-mm', mm_eval_dataloader, output_mode,
                                                   mm_eval_labels, num_labels)
                            result['global_step'] = global_step
                            if not os.path.exists(self.output_dir + '-MM'):
                                os.makedirs(self.output_dir + '-MM')
                            tmp_output_eval_file = os.path.join(self.output_dir + '-MM', "eval_results.txt")
                            result_to_file(result, tmp_output_eval_file)

                # if self.config.quantize_weight:
                # self.quanter.restore()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

    def _save(self):
        logging.info("******************** Save model ********************")
        model_to_save = self.student_model.module if hasattr(self.student_model, 'module') else self.student_model
        output_model_file = os.path.join(self.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def check_grad_scale(self):
        logging.info("Check grad scale ratio: grad/w")
        for k, v in self.student_model.named_parameters():
            if v.grad is not None:
                has_grad = True
                ratio = v.grad.norm(p=2) / v.data.norm(p=2)
                # print('%.6e, %s' % (ratio.float(), k))
            else:
                has_grad = False
                logging.info('params: %s has no gradient' % k)
                continue

            # update grad_scale stats
            if 'weight' in k and v.ndimension() == 2:
                key = 'weight'
            elif 'bias' in k and v.ndimension() == 1:
                key = 'bias'
            elif 'LayerNorm' in k and 'weight' in k and v.ndimension() == 1:
                key = 'layer_norm'
            elif 'clip_' in k:
                key = 'step_size/clip_val'
            else:
                key = None

            if key and has_grad:
                if self.grad_scale_stats[key]:
                    self.grad_scale_stats[key] = self.ema_grad * self.grad_scale_stats[key] + (
                            1 - self.ema_grad) * ratio
                else:
                    self.grad_scale_stats[key] = ratio

        for (key, val) in self.grad_scale_stats.items():
            if val is not None:
                logging.info('%.6e, %s' % (val, key))
