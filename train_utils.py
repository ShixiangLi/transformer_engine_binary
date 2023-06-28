import time

import torch

from CMAPSSDataset import get_data


# 学习率调整函数
def rate(step, model_size, factor, warmup):
    # 避免分母为0
    if step == 0:
        step = 1
    # 这里比原公式还多了一个factor，factor默认取1，相当于没有多。
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def att_loss_r2b(Q_s, Q_t):
    Q_s_norm = Q_s / torch.norm(Q_s, p=2)
    Q_t_norm = Q_t / torch.norm(Q_t, p=2)
    tmp = Q_s_norm - Q_t_norm
    loss = torch.norm(tmp, p=2)
    return loss


def direction_matching_distillation(student_scores, teacher_scores, layers_per_block, student_layer_num, device):
    tmp_loss = 0.
    new_teacher_scores = [teacher_scores[i * layers_per_block + layers_per_block - 1] for i in
                          range(student_layer_num)]
    for student_score, teacher_score in zip(student_scores, new_teacher_scores):
        student_score = torch.where(student_score <= -1e2,
                                    torch.zeros_like(student_score).to(device),
                                    student_score)
        teacher_score = torch.where(teacher_score <= -1e2,
                                    torch.zeros_like(teacher_score).to(device),
                                    teacher_score)
        tmp_loss += att_loss_r2b(student_score, teacher_score)
    return tmp_loss


def data_gen(FD='1', feature_columns=[], sequence_length=30, batch_size=20, label='train'):
    """
    生成一组随机数据。（该方法仅用于Demo）
    :param V: 词典的大小
    :param batch_size
    :param nbatches: 生成多少个batch
    :return: yield一个Batch对象
    """
    feature, labels = get_data(FD=FD, feature_columns=feature_columns,
                               sequence_length=sequence_length, batch_size=batch_size, label=label)
    feature = torch.tensor(feature)
    labels = torch.tensor(labels)
    num_batch = len(feature) // batch_size
    for i in range(num_batch):
        src = feature[i * batch_size:(i + 1) * batch_size]
        tgt = labels[i * batch_size:(i + 1) * batch_size]
        src = src.requires_grad_(False).clone().detach()
        tgt = tgt.requires_grad_(False).clone().detach()
        yield Batch(src, tgt)


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


class TrainState:
    """用于保存一些训练状态"""

    # step的次数，但注意是一次loss.backward()算一次，或者说一个batch算一次
    # 并不是一次optimizer.step()算一次。在后面的训练代码中，可能会累计多次loss
    # 然后进行一次optimizer.step()
    step: int = 0

    # 参数更新的次数。这个才是optimizer.step()的次数
    accum_step: int = 0

    samples: int = 0  # 记录训练过的样本数量
    tokens: int = 0  # 记录处理过的token数量（target的）


class SimpleLossCompute:
    """
    一个简单的损失计算和训练函数。
    该类除了包含损失计算外，还包含模型generator部分的前向传递。
    如果你对上面这句话不太理解，可参考这篇文章：
    https://blog.csdn.net/zhaohongfei_358/article/details/125759911
    请参考章节：Pytorch 实现梯度下降与参数更新
    """

    def __init__(self, criterion):
        """
        generator: Generator类对象，用于根据Decoder的输出预测下一个token
        criterion: LabelSmoothing类对象，用于对Label进行平滑和计算损失
        """
        self.criterion = criterion

    def __call__(self, x, y):
        """
        x: EncoderDecoder的输出，也就是Decoder的输出
        y: batch.tgt_y，要被预测的所有token，例如src为`<bos> I love you <eos>`，
           则`tgt_y`则为`我 爱 你 <eos>`
        norm: batch.ntokens, tgt_y中的有效token数。用于对loss进行正则化。
        """

        """
        这里首先使用KLDivLoss进行了损失计算，随后又除以batch.ntokens对loss进行正则化。
        """
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous()
            )
        )

        return sloss


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        device,
        scheduler=None,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """
    进行一个epoch训练

    data_iter: 可迭代对象，一次返回一个Batch对象
    model: Transformer模型，EncoderDecoder类对象
    loss_compute: SimpleLossCompute对象，用于计算损失
    optimizer: Adam优化器。验证时，optimizer是DummyOptimizer
    scheduler：LambdaLR对象，用于调整Adam的学习率，实现WarmUp
               若对调整学习率不熟悉，可参考：https://blog.csdn.net/zhaohongfei_358/article/details/125759911
               验证时，scheduler是DummyScheduler
    accum_iter: 多少个batch更新一次参数，默认为1，也就是每个batch都对参数进行更新
    train_state: TrainState对象，用于保存一些训练状态
    """
    start = time.time()
    total_loss = 0
    n_accum = 0  # 本次epoch更新了多少次模型参数
    for i, batch in enumerate(data_iter):
        # 前向传递。等价于model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        # 但注意，这里的out是Decoder的输出，并不是Generator的输出，因为在EncoderDecoder
        # 的forward中并没有使用generator。generator的调用放在了loss_compute中
        _, _, out, _, _, _, _ = model.forward(batch.src.to(device), batch.src_mask.to(device))

        """
        计算损失，传入的三个参数分别为：
        1. out: EncoderDecoder的输出，该值并没有过最后的线性层，过线性层被集成在了计算损失中
        2. tgt_y: 例如src为`<bos> I love you <eos>`，则`tgt_y`则为
                  `我 爱 你 <eos>`

        返回两个loss，其中loss_node是正则化之后的，所以梯度下降时用这个。
                    而loss是未进行正则化的，用于统计total_loss。
        """
        loss = loss_compute(out, batch.tgt.to(device))
        loss_node = loss
        if mode == "train" or mode == "train+log":
            # 计算梯度
            loss_node.backward()
            # 记录step次数
            train_state.step += 1
            # 记录样本数量。batch.src.shape[0]获取的是Batch size
            train_state.samples += batch.src.shape[0]

            # 如果达到了accum_iter次，就进行一次参数更新
            if i % accum_iter == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # 记录本次epoch的参数更新次数
                n_accum += 1
                # 记录模型的参数更新次数
                train_state.accum_step += 1
            # 更新学习率
            scheduler.step()

        # 累计loss
        total_loss += loss_node
        # 每40个batch打印一次日志。
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            # 打印一下当前的学习率
            lr = optimizer.param_groups[0]["lr"]
            # 记录这40个batch的消耗时间
            elapsed = time.time() - start
            # 打印日志
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %.5f "
                        + "| Sec: %4.2f | Learning Rate: %6.1e"
                )
                # i: 本次epoch的第几个batch
                # n_accum: 本次epoch更新了多少次模型参数
                # lr: 学习率（learning rate），这里打印学习率的目的是看一下warmup下学习率的变化
                % (i, n_accum, torch.sqrt(loss)*125, elapsed, lr)
            )
            # 重置开始时间
            start = time.time()

        del loss_node
        del loss
    # 返回正则化之后的total_loss，返回训练状态
    return total_loss / (i + 1), train_state


def run_distill_epoch(
        data_iter,
        teacher_model,
        student_model,
        loss_compute,
        optimizer,
        device,
        scheduler=None,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    start = time.time()
    total_loss = 0.
    # rep_loss = 0.
    n_accum = 0  # 本次epoch更新了多少次模型参数
    for i, batch in enumerate(data_iter):
        student_logits, student_score, student_reps, student_values, student_context, student_queries, student_keys = student_model.forward(
            batch.src.to(device), batch.src_mask.to(device))
        with torch.no_grad():
            teacher_logits, teacher_score, teacher_reps, teacher_values, teacher_context, teacher_queries, teacher_keys = teacher_model.forward(
                batch.src.to(device), batch.src_mask.to(device))

        teacher_layer_num = len(teacher_values)
        student_layer_num = len(student_values)
        assert teacher_layer_num % student_layer_num == 0

        layers_per_block = int(teacher_layer_num / student_layer_num)

        loss = loss_compute(student_reps, teacher_reps)

        query_loss = direction_matching_distillation(student_queries, teacher_queries, layers_per_block,
                                                     student_layer_num, device)
        loss += query_loss

        key_loss = direction_matching_distillation(student_keys, teacher_keys, layers_per_block,
                                                     student_layer_num, device)
        loss += key_loss

        value_loss = direction_matching_distillation(student_values, teacher_values, layers_per_block,
                                                     student_layer_num, device)
        loss += value_loss

        # new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        # teacher_reps = new_teacher_reps
        # for student_rep, teacher_rep in zip(student_reps, teacher_reps):
        #     rep_loss += att_loss_r2b(student_rep, teacher_rep)
        # loss += rep_loss

        if mode == "train" or mode == "train+log":
            # 计算梯度
            loss.backward()
            # 记录step次数
            train_state.step += 1
            # 记录样本数量。batch.src.shape[0]获取的是Batch size
            train_state.samples += batch.src.shape[0]

            # 如果达到了accum_iter次，就进行一次参数更新
            if i % accum_iter == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # 记录本次epoch的参数更新次数
                n_accum += 1
                # 记录模型的参数更新次数
                train_state.accum_step += 1
            # 更新学习率
            scheduler.step()

        # 累计loss
        total_loss += loss
        # 每40个batch打印一次日志。
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            # 打印一下当前的学习率
            lr = optimizer.param_groups[0]["lr"]
            # 记录这40个batch的消耗时间
            elapsed = time.time() - start
            # 打印日志
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %.5f "
                        + "| Sec: %4.2f | Learning Rate: %6.1e"
                )
                # i: 本次epoch的第几个batch
                # n_accum: 本次epoch更新了多少次模型参数
                # lr: 学习率（learning rate），这里打印学习率的目的是看一下warmup下学习率的变化
                % (i, n_accum, loss, elapsed, lr)
            )
            # 重置开始时间
            start = time.time()

        del loss
        del key_loss
        del query_loss
        del value_loss
        # del rep_loss
        # 返回正则化之后的total_loss，返回训练状态
    return total_loss / (i + 1), train_state
