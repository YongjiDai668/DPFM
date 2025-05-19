# encoding:utf-8
import os
import nni
import math
import time
import json
import torch
import argparse
import torch.nn.functional as F
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from nni.utils import merge_parameter
from transformers import AutoModel, AutoTokenizer

from model import BertEncoder, Classifier
from data_process import FewRelProcessor, tacredProcessor
from utils import collate_fn, save_checkpoint, get_prototypes, memory_select, set_random_seed, compute_cos_sim, \
    get_augmentative_data, Memory_train_get_prototypes, get_static_prototypes, SupConLoss,get_augmentative_memory_data
from torch.utils.data import Dataset, DataLoader

default_print = "\033[0m"
blue_print = "\033[1;34;40m"
yellow_print = "\033[1;33;40m"
green_print = "\033[1;32;40m"
import logging


## SIM-BERT \ 静态平均，动态指导  \ CALOSS保持空间结构    \原型相似度区分不同类空间  \prompt  \（三元组损失）
# 添加层次对比损失(在初始阶段和回忆阶段？？？)
# 焦点知识蒸馏没有MASK新任务

def do_train(args, tokenizer, processor, i_exp):
    memory = []
    memory_len = []
    relations = []
    testset = []
    relations_description = []

    prev_encoder, prev_classifier = None, None
    taskdatas = processor.get()
    rel2id = processor.get_rel2id()  # {"rel": id}
    task_acc, memory_acc = [], []
    prototypes = None
    # Based_knowledge_relation_encoder = BertModel.from_pretrained('bert-base-uncased')
    logging.basicConfig(filename='SIM_training_memory.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Experiment {i_exp}")
    now_task_num = 0
    for i in range(args.task_num):
        now_task_num += 1
        task = taskdatas[i]
        traindata, _, testdata = task['train'], task['val'], task['test']
        train_len = task['train_len']
        testset += testdata
        new_relations = task['relation']
        new_relations_description = task['relations_description']
        new_relations_description_input = encode_relations_description(args, tokenizer, new_relations_description, 512)
        relations += new_relations
        relations_description += new_relations_description
        relations_description_input = encode_relations_description(args, tokenizer, relations_description, 512)

        args.seen_rel_num = len(relations)

        # print some info
        print(f"{yellow_print}Training task {i}, relation set {task['relation']}.{default_print}")
        logging.info(f"Training task {i}, relation set {task['relation']}.")

        # train and val on task data
        current_encoder = BertEncoder(args, tokenizer, encode_style=args.encode_style)
        current_classifier = Classifier(args, args.hidden_dim, args.seen_rel_num, prev_classifier).to(args.device)

        if prev_encoder is not None:
            current_encoder.load_state_dict(prev_encoder.state_dict())
        if args.dataset_name == "FewRel":
            current_encoder = train_val_task(args, current_encoder, current_classifier, traindata, testdata, rel2id,
                                             train_len, len(relations))

        else:
            aug_traindata = get_augmentative_data(args, traindata, train_len)
            # current_encoder = train_val_task(args, current_encoder, current_classifier, aug_traindata, testdata, rel2id,
            #                                  train_len, len(relations))

        # memory select
        print(f'{blue_print}Selecting memory for task {i}...{default_print}')
        new_memory, new_memory_len = memory_select(args, current_encoder, traindata, train_len)
        memory += new_memory
        memory_len += new_memory_len
        ###             记忆搜索，memory拥有所有典型样例，memory_len拥有对应的长度

        # evaluate on task testdata
        current_prototypes, current_proto_features = get_prototypes(args, current_encoder, traindata, train_len,
                                                                    new_relations_description_input)
        acc = evaluate(args, current_encoder, current_classifier, testdata, rel2id)
        print(f'{blue_print}Accuracy of task {i} is {acc}.{default_print}')
        logging.info(f"Accuracy of task {i} is {acc}.")
        task_acc.append(acc)

        # train and val on memory data
        if prev_encoder is not None:
            print(f'{blue_print}Training on memory...{default_print}')
            task_prototypes = torch.cat([task_prototypes, current_prototypes], dim=0)
            task_proto_features = torch.cat([task_proto_features, current_proto_features], dim=0)

            prototypes = torch.cat([prototypes, current_prototypes], dim=0)
            proto_features = torch.cat([proto_features, current_proto_features], dim=0)

            current_model = (current_encoder, current_classifier)
            prev_model = (prev_encoder, prev_classifier)
            aug_memory = get_augmentative_memory_data(args, memory, memory_len,current_encoder)
            # current_encoder = train_val_memory(args, current_model, prev_model, memory, aug_memory, testset, rel2id,
            #                                    memory_len, memory_len, prototypes, proto_features, task_prototypes,
            #                                    task_proto_features, relations_description_input, now_task_num)
        else:
            print(f"{blue_print}Initial task, won't train on memory.{default_print}")
            logging.info(f"Initial task, won't train on memory.")

        # update prototype
        print(f'{blue_print}Updating prototypes...{default_print}')
        if prev_encoder is not None:
            prototypes_replay, proto_features_replay = get_prototypes(args, current_encoder, memory, memory_len,
                                                                      relations_description_input)
            prototypes, proto_features = (1 - args.beta) * task_prototypes + args.beta * prototypes_replay, (
                        1 - args.beta) * task_proto_features + args.beta * proto_features_replay
            prototypes = F.layer_norm(prototypes, [args.hidden_dim])
            proto_features = F.normalize(proto_features, p=2, dim=1)
        else:
            task_prototypes, task_proto_features = current_prototypes, current_proto_features
            prototypes, proto_features = current_prototypes, current_proto_features

        # test
        print(f'{blue_print}Evaluating...{default_print}')
        if prev_encoder is not None:
            acc = evaluate(args, current_encoder, current_classifier, testset, rel2id, proto_features)
        else:
            acc = evaluate(args, current_encoder, current_classifier, testset, rel2id)
        print(f'{green_print}Evaluate finished, final accuracy over task 0-{i} is {acc}.{default_print}')
        logging.info(f"Final accuracy over task 0-{i} is {acc}.")
        memory_acc.append(acc)

        # save checkpoint
        print(f'{blue_print}Saving checkpoint of task {i}...{default_print}')
        save_checkpoint(args, current_encoder, i_exp, i, "encoder")
        save_checkpoint(args, current_classifier, i_exp, i, "classifier")

        prev_encoder = current_encoder
        prev_classifier = current_classifier
        nni.report_intermediate_result(acc)

    return task_acc, memory_acc



def train_val_task(args, encoder, classifier, traindata, valdata, rel2id, train_len, relations_num):
    dataloader = DataLoader(traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn,
                            drop_last=True)

    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': args.encoder_lr},
        {'params': classifier.parameters(), 'lr': args.classifier_lr}
    ], eps=args.adam_epsilon)
    # todo add different learning rate for each layer

    best_acc = 0.0
    for epoch in range(args.epoch_num_task):
        encoder.train()
        classifier.train()
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }
            # hidden, _, mask_hidden, _ = encoder(**inputs)
            hidden, _ = encoder(**inputs)

            inputs = {
                'hidden': hidden,
                # 'hidden': mask_hidden,
                'labels': batch[4].to(args.device)
            }
            loss, _ = classifier(**inputs)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    acc = evaluate(args, encoder, classifier, valdata, rel2id)
    best_acc = max(acc, best_acc)
    print(f'Evaluate on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')

    return encoder


# def train_val_memory(args, model, prev_model, traindata, aug_traindata, testdata, rel2id, memory_len, aug_memory_len, prototypes, proto_features, task_prototypes, task_proto_features, relations,now_task_num):
#     enc, cls = model
#     prev_enc, prev_cls = prev_model

#     class CustomDataset(Dataset):
#         def __init__(self, data):
#             self.data = data

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             # 返回数据和对应的索引
#             return self.data[idx], idx

#     def batched_matrix_multiplication(A, B, batch_size):
#         result = []
#         for i in range(0, A.size(0), batch_size):
#             result.append(torch.mm(A[i:i + batch_size], B))
#         return torch.cat(result, dim=0)

#     # dataloader = DataLoader(aug_traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)
#     dataloader = DataLoader(CustomDataset(aug_traindata), batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn,
#                             drop_last=True)
#     logging.basicConfig(filename='SIM_training_memory.log', level=logging.INFO,
#                         format='%(asctime)s - %(levelname)s - %(message)s')
#     optimizer = AdamW([
#         {'params': enc.parameters(), 'lr': args.encoder_lr},
#         {'params': cls.parameters(), 'lr': args.classifier_lr}
#     ], eps=args.adam_epsilon)


#     if args.dataset_name == "FewRel":
#         rel_per_task = 8
#         alpha_CALOSS = 1.0
#         rel_num = rel_per_task * now_task_num
#     else:
#         rel_per_task = 4
#         alpha_CALOSS = 0.5
#         rel_num = rel_per_task * now_task_num
#     train_set = {}
#     for label_i in range(rel_num):
#         train_set[str(label_i)] = []
#         for item in aug_traindata:
#             if item['label'] == label_i:
#                 train_set[str(label_i)].append(item)
#     prev_enc.eval()
#     prev_cls.eval()
#     best_acc = 0.0
#     # 创建一个全1的矩阵
#     full_ones = torch.ones((len(memory_len), len(memory_len)), dtype=torch.float32).to(device)
#     # 创建一个掩码，下三角的元素为0，其余为1
#     mask = torch.triu(full_ones, diagonal=1).to(device)
#     for epoch in range(args.epoch_num_memory):
#         enc.train()
#         cls.train()
#         for step, batch in enumerate(tqdm(dataloader)):
#             enc.train()
#             cls.train()
#             labels = batch[4].to(args.device)
#             loss_mask = labels < len(memory_len) - rel_per_task
#             enc_inputs = {
#                 'input_ids': batch[0].to(args.device),
#                 'attention_mask': batch[1].to(args.device),
#                 'h_index': batch[2].to(args.device),
#                 't_index': batch[3].to(args.device),
#             }

#             batch_data2 = []
#             for batch_data_index in batch[6]:
#                 # batch_label = batch_data['label']
#                 # no_self = train_set[str(batch_label)].copy()
#                 #
#                 # no_self.remove(batch_data)
#                 # item2 = random.choice(no_self)
#                 # batch_data2.append(item2)
#                 self_data = aug_traindata[batch_data_index]
#                 batch_label = self_data['label']
#                 item_list = train_set[str(batch_label)]
#                 if len(item_list) > 1:
#                     while True:
#                         item2 = random.choice(item_list)
#                         if item2 != self_data:
#                             break
#                 else:
#                     # 处理只有一个元素的情况，或者跳过循环
#                     item2 = item_list[0]
#                 batch_data2.append(item2)
#             batch2 = args.collate_fn(batch_data2)
#             batch2_inputs = {
#                 'input_ids': batch2[0].to(args.device),
#                 'attention_mask': batch2[1].to(args.device),
#                 'h_index': batch2[2].to(args.device),
#                 't_index': batch2[3].to(args.device),
#             }
#             hidden, feature = enc(**enc_inputs)
#             hidden2, feature2 = enc(**batch2_inputs)
#             with torch.no_grad():
#                 prev_hidden, prev_feature = prev_enc(**enc_inputs)
#                 prev_hidden2, prev_feature2 = prev_enc(**batch2_inputs)

#             # # CALoss
#             Kloss = torch.sum(((feature - prev_feature) - (feature2 - prev_feature2)) ** 2, dim=1)  ##CA损失
#             Kloss = 0 if torch.sum(loss_mask) == 0 else torch.sum(Kloss * loss_mask) / torch.sum(loss_mask)
#             # Kloss = 0 if torch.sum(loss_mask) == 0 else torch.sum(Kloss * loss_mask) / torch.sum(loss_mask) * ((now_task_num-1) ** alpha_CALOSS)
#             cos_similarity = F.cosine_similarity(feature, prev_feature, dim=1)
#             # 使用温度调节
#             T = 1.0  # 温度参数
#             CosKloss = 1 - (cos_similarity / T)
#             # 应用掩码
#             CosKloss = torch.sum(Kloss * loss_mask) / torch.sum(loss_mask) if torch.sum(loss_mask) != 0 else 0.0

#             #Con_Loss、fkdloss、++SimilarPrototypeLoss
#             cont_loss,_ = contrastive_loss(args, feature, labels, prototypes, proto_features, prev_feature,loss_mask)
#             # cont_loss = cont_loss + 0.3*similarity_prototypes_loss+0.3*Kloss
#             cont_loss = cont_loss + 0.1 * Kloss + 0.1 * CosKloss
#             cont_loss.backward(retain_graph=True)

#             rep_loss,_ = replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels, prototypes, proto_features,loss_mask)

#             rep_loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             # torch.cuda.empty_cache()

#         #增加原型相似度区分、以及平行学习损失

#         if (epoch+1) % 10 == 0:
#           acc = evaluate(args, enc, cls, testdata, rel2id, proto_features)
#           best_acc = max(best_acc, acc)
#           print(f'Evaluate testset on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
#           logging.info(f'Evaluate testset on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
#           nni.report_intermediate_result(acc)

#           prototypes_replay, proto_features_replay = get_prototypes(args, enc, traindata, memory_len, relations)
#           prototypes, proto_features = (1-args.beta)*task_prototypes + args.beta*prototypes_replay, (1-args.beta)*task_proto_features + args.beta*proto_features_replay
#           prototypes = F.layer_norm(prototypes, [args.hidden_dim])
#           proto_features = F.normalize(proto_features, p=2, dim=1)

#     return enc

def train_val_memory(args, model, prev_model, traindata, aug_traindata, testdata, rel2id, memory_len, aug_memory_len,
                     prototypes, proto_features, task_prototypes, task_proto_features, relations, now_task_num):
    enc, cls = model
    prev_enc, prev_cls = prev_model

    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # 返回数据和对应的索引
            return self.data[idx], idx

    def batched_matrix_multiplication(A, B, batch_size):
        result = []
        for i in range(0, A.size(0), batch_size):
            result.append(torch.mm(A[i:i + batch_size], B))
        return torch.cat(result, dim=0)

    # dataloader = DataLoader(aug_traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)
    dataloader = DataLoader(CustomDataset(aug_traindata), batch_size=args.train_batch_size, shuffle=True,
                            collate_fn=args.collate_fn, drop_last=True)
    con_criterion = SupConLoss(device=args.device)
    logging.basicConfig(filename='SIM_training_memory.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    optimizer = AdamW([
        {'params': enc.parameters(), 'lr': args.encoder_lr},
        {'params': cls.parameters(), 'lr': args.classifier_lr}
    ], eps=args.adam_epsilon)

    if args.dataset_name == "FewRel":
        rel_per_task = 8
        alpha_CALOSS = 1.0
        rel_num = rel_per_task * now_task_num
    else:
        rel_per_task = 4
        alpha_CALOSS = 0.5
        rel_num = rel_per_task * now_task_num
    train_set = {}
    for label_i in range(rel_num):
        train_set[str(label_i)] = []
        for item in aug_traindata:
            if item['label'] == label_i:
                train_set[str(label_i)].append(item)
    prev_enc.eval()
    prev_cls.eval()
    best_acc = 0.0
    # 创建一个全1的矩阵
    full_ones = torch.ones((len(memory_len), len(memory_len)), dtype=torch.float32).to(device)
    # 创建一个掩码，下三角的元素为0，其余为1
    mask = torch.triu(full_ones, diagonal=1).to(device)
    for epoch in range(args.epoch_num_memory):
        enc.train()
        cls.train()
        for step, batch in enumerate(tqdm(dataloader)):
            enc.train()
            cls.train()
            labels = batch[4].to(args.device)
            loss_mask = labels < len(memory_len) - rel_per_task
            enc_inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }

            batch_data2 = []
            for batch_data_index in batch[6]:
                self_data = aug_traindata[batch_data_index]
                batch_label = self_data['label']
                item_list = train_set[str(batch_label)]
                if len(item_list) > 1:
                    while True:
                        item2 = random.choice(item_list)
                        if item2 != self_data:
                            break
                else:
                    # 处理只有一个元素的情况，或者跳过循环
                    item2 = item_list[0]
                batch_data2.append(item2)
            batch2 = args.collate_fn(batch_data2)
            batch2_inputs = {
                'input_ids': batch2[0].to(args.device),
                'attention_mask': batch2[1].to(args.device),
                'h_index': batch2[2].to(args.device),
                't_index': batch2[3].to(args.device),
            }
            hidden, feature = enc(**enc_inputs)
            hidden2, feature2 = enc(**batch2_inputs)
            with torch.no_grad():
                prev_hidden, prev_feature = prev_enc(**enc_inputs)
                prev_hidden2, prev_feature2 = prev_enc(**batch2_inputs)

            # # CALoss
            CAloss = torch.sum(((feature - prev_feature) - (feature2 - prev_feature2)) ** 2, dim=1)  ##CA损失
            CAloss = 0 if torch.sum(loss_mask) == 0 else torch.sum(CAloss * loss_mask) / torch.sum(loss_mask)
            # Kloss = 0 if torch.sum(loss_mask) == 0 else torch.sum(Kloss * loss_mask) / torch.sum(loss_mask) * ((now_task_num-1) ** alpha_CALOSS)

            rep_output = torch.cat([feature, feature2], dim=1)
            rep_feature = torch.reshape(torch.unsqueeze(rep_output, 1).to(args.device), (-1, 2, args.contrasive_size))
            con_loss = torch.sum(con_criterion(rep_feature, labels)) / (args.train_batch_size)
            # con_loss = 0 if args.train_batch_size == torch.sum(loss_mask) else torch.sum(torch.mul(con_criterion(rep_feature, labels), ~loss_mask)) / (args.train_batch_size - torch.sum(loss_mask))

            cos_similarity = F.cosine_similarity(feature, prev_feature, dim=1)
            # 使用温度调节
            T = 1.0  # 温度参数
            CosKloss = 1 - (cos_similarity / 0.5)
            # 应用掩码
            CosKloss = torch.sum(CosKloss * loss_mask) / torch.sum(loss_mask) if torch.sum(loss_mask) != 0 else 0.0

            # Con_Loss、fkdloss、++SimilarPrototypeLoss
            cont_loss, _ = contrastive_loss(args, feature, labels, prototypes, proto_features, prev_feature, loss_mask)
            # cont_loss = cont_loss + 0.3*similarity_prototypes_loss+0.3*Kloss
            cont_loss = cont_loss + 0.8 * CosKloss + 0.5 * CAloss + 0.4 * con_loss
            cont_loss.backward(retain_graph=True)

            rep_loss, _ = replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels,
                                      prototypes, proto_features, loss_mask)

            rep_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # torch.cuda.empty_cache()

        # 增加原型相似度区分、以及平行学习损失

        if (epoch + 1) % 10 == 0:
            acc = evaluate(args, enc, cls, testdata, rel2id, proto_features)
            best_acc = max(best_acc, acc)
            print(f'Evaluate testset on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
            logging.info(f'Evaluate testset on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
            nni.report_intermediate_result(acc)

            prototypes_replay, proto_features_replay = get_prototypes(args, enc, traindata, memory_len, relations)
            prototypes, proto_features = (1 - args.beta) * task_prototypes + args.beta * prototypes_replay, (
                        1 - args.beta) * task_proto_features + args.beta * proto_features_replay
            prototypes = F.layer_norm(prototypes, [args.hidden_dim])
            proto_features = F.normalize(proto_features, p=2, dim=1)

    return enc


def contrastive_loss(args, feature, labels, prototypes, proto_features=None, prev_feature=None, loss_mask=None):
    # supervised contrastive learning loss
    dot_div_temp = torch.mm(feature, proto_features.T) / args.cl_temp  # [batch_size, rel_num]
    dot_div_temp_norm = dot_div_temp - 1.0 / args.cl_temp
    exp_dot_temp = torch.exp(dot_div_temp_norm) + 1e-8  # avoid log(0)

    mask = torch.zeros_like(exp_dot_temp).to(args.device)
    mask.scatter_(1, labels.unsqueeze(1), 1.0)
    cardinalities = torch.sum(mask, dim=1)

    log_prob = -torch.log(exp_dot_temp / torch.sum(exp_dot_temp, dim=1, keepdim=True))
    scloss_per_sample = torch.sum(log_prob * mask, dim=1) / cardinalities
    scloss = torch.mean(scloss_per_sample)

    # focal knowledge distillation loss
    if prev_feature is not None:
        with torch.no_grad():
            prev_proto_features = proto_features[:proto_features.shape[0] - args.relnum_per_task]
            prev_sim = F.softmax(torch.mm(feature, prev_proto_features.T) / args.cl_temp / args.kd_temp, dim=1)

            prob = F.softmax(torch.mm(feature, proto_features.T) / args.cl_temp / args.kd_temp, dim=1)
            focal_weight = 1.0 - torch.gather(prob, dim=1, index=labels.unsqueeze(1)).squeeze()
            focal_weight = focal_weight ** args.gamma

            target = F.softmax(torch.mm(prev_feature, prev_proto_features.T) / args.cl_temp,
                               dim=1)  # [batch_size, prev_rel_num]

        source = F.log_softmax(torch.mm(feature, prev_proto_features.T) / args.cl_temp,
                               dim=1)  # [batch_size, prev_rel_num]
        target = target * prev_sim + 1e-8
        fkdloss = torch.sum(-source * target, dim=1)
        fkdloss = torch.mean(fkdloss * focal_weight)

    else:
        fkdloss = 0.0

    # margin loss
    if proto_features is not None:
        with torch.no_grad():
            sim = torch.mm(feature, proto_features.T)
            neg_sim = torch.scatter(sim, 1, labels.unsqueeze(1), -10.0)
            neg_indices = torch.argmax(neg_sim, dim=1)

        pos_proto = proto_features[labels]
        neg_proto = proto_features[neg_indices]

        positive = torch.sum(feature * pos_proto, dim=1)
        negative = torch.sum(feature * neg_proto, dim=1)

        marginloss = torch.maximum(args.margin - positive + negative, torch.zeros_like(positive).to(args.device))
        marginloss = torch.mean(marginloss)
    else:
        marginloss = 0.0

    loss = scloss + args.cl_lambda * marginloss + args.kd_lambda2 * fkdloss

    return loss, scloss


def replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels, prototypes=None,
                proto_features=None, loss_mask=None):
    # cross entropy
    celoss, logits = cls(hidden, labels)
    with torch.no_grad():
        prev_logits, = prev_cls(prev_hidden)

    if prototypes is None:
        index = prev_logits.shape[1]
        source = F.log_softmax(logits[:, :index], dim=1)
        target = F.softmax(prev_logits, dim=1) + 1e-8
        kdloss = F.kl_div(source, target)
    else:
        # focal knowledge distillation
        with torch.no_grad():
            sim = compute_cos_sim(hidden, prototypes)
            prev_sim = sim[:, :prev_logits.shape[1]]  # [batch_size, prev_rel_num]
            prev_sim = F.softmax(prev_sim / args.kd_temp, dim=1)

            prob = F.softmax(logits, dim=1)
            focal_weight = 1.0 - torch.gather(prob, dim=1, index=labels.unsqueeze(1)).squeeze()
            focal_weight = focal_weight ** args.gamma

        source = logits.narrow(1, 0, prev_logits.shape[1])
        source = F.log_softmax(source, dim=1)
        target = F.softmax(prev_logits, dim=1)
        target = target * prev_sim + 1e-8
        kdloss = torch.sum(-source * target, dim=1)
        kdloss = torch.mean(kdloss * focal_weight)
        # masked_fkdloss = kdloss * focal_weight * loss_mask  # 应用掩码和焦点权重
        # sum_loss_mask = torch.sum(loss_mask)  # 有效样本数量
        # kdloss = 0 if sum_loss_mask == 0 else torch.sum(masked_fkdloss) / sum_loss_mask

    rep_loss = celoss + args.kd_lambda1 * kdloss

    return rep_loss, celoss


# def evaluate(args, model, classifier, valdata, rel2id, proto_features=None):
#     model.eval()
#     dataloader = DataLoader(valdata, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
#     pred_labels, golden_labels = [], []

#     for i, batch in enumerate(tqdm(dataloader)):
#         inputs = {
#             'input_ids': batch[0].to(args.device),
#             'attention_mask': batch[1].to(args.device),
#             'h_index': batch[2].to(args.device),
#             't_index': batch[3].to(args.device),

#         }

#         with torch.no_grad():
#             # hidden, feature ,MASK_hidden, MASK_feature= model(**inputs)
#             hidden, feature = model(**inputs)
#             logits = classifier(hidden)[0]
#             # logits = classifier(MASK_hidden)[0]   ##MASK版本
#             prob_cls = F.softmax(logits, dim=1)
#             if proto_features is not None:
#                 logits = torch.mm(feature, proto_features.T) / args.cl_temp
#                 prob_ncm = F.softmax(logits, dim=1)
#                 final_prob = args.alpha*prob_cls + (1-args.alpha)*prob_ncm
#             else:
#                 final_prob = prob_cls

#         # get pred_labels
#         pred_labels += torch.argmax(final_prob, dim=1).cpu().tolist()
#         golden_labels += batch[4].tolist()

#     pred_labels = torch.tensor(pred_labels, dtype=torch.long)
#     golden_labels = torch.tensor(golden_labels, dtype=torch.long)

#     acc = float(torch.sum(pred_labels==golden_labels).item()) / float(len(golden_labels))
#     return acc

def evaluate(args, model, classifier, valdata, rel2id, proto_features=None):
    model.eval()
    dataloader = DataLoader(valdata, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    pred_labels, golden_labels = [], []
    cont_pred_labels, cls_pred_labels = [], []  # 在这里初始化
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),

        }

        with torch.no_grad():
            # hidden, feature ,MASK_hidden, MASK_feature= model(**inputs)
            hidden, feature = model(**inputs)
            logits = classifier(hidden)[0]
            # logits = classifier(MASK_hidden)[0]   ##MASK版本
            prob_cls = F.softmax(logits, dim=1)
            if proto_features is not None:
                logits = torch.mm(feature, proto_features.T) / args.cl_temp
                prob_ncm = F.softmax(logits, dim=1)
                final_prob = args.alpha * prob_cls + (1 - args.alpha) * prob_ncm
            else:
                final_prob = prob_cls

        # get pred_labels
        pred_labels += torch.argmax(final_prob, dim=1).cpu().tolist()
        if proto_features is not None:
            cont_pred_labels += torch.argmax(prob_ncm, dim=1).cpu().tolist()
            cls_pred_labels += torch.argmax(prob_cls, dim=1).cpu().tolist()
        golden_labels += batch[4].tolist()

    pred_labels = torch.tensor(pred_labels, dtype=torch.long)
    golden_labels = torch.tensor(golden_labels, dtype=torch.long)
    if proto_features is not None:
        cont_pred_labels = torch.tensor(cont_pred_labels, dtype=torch.long)
        cls_pred_labels = torch.tensor(cls_pred_labels, dtype=torch.long)
        ncm_acc = float(torch.sum(cont_pred_labels == golden_labels).item()) / float(len(golden_labels))
        cls_acc = float(torch.sum(cls_pred_labels == golden_labels).item()) / float(len(golden_labels))
        print(f"Classifieracc:{cls_acc}")
        print(f"ConAcc:{ncm_acc}")
    acc = float(torch.sum(pred_labels == golden_labels).item()) / float(len(golden_labels))
    return acc


import torch


def encode_relations_description(args, tokenizer, relations_description, max_length=512):
    # 初始化编码结果列表
    input_ids_list = []
    attention_masks_list = []

    # 遍历关系描述列表
    for description in relations_description:
        # 对每个描述进行编码
        encoded_description = tokenizer(description,
                                        add_special_tokens=True,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors="pt")

        # 将编码结果添加到列表中
        input_ids_list.append(encoded_description['input_ids'].squeeze(1))  # 移除单维度
        attention_masks_list.append(encoded_description['attention_mask'].squeeze(1))  # 移除单维度

    # 使用torch.cat沿着第二个维度拼接编码结果
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)

    # 将input_ids和attention_masks合并为一个字典
    return {
        'input_ids': input_ids.to(args.device),
        'attention_mask': attention_masks.to(args.device)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoint", type=str)
    parser.add_argument("--dataset_name", default="tacred", type=str)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--cuda_device", default=1, type=int)

    parser.add_argument("--plm_name", default="princeton-nlp/sup-simcse-bert-base-uncased", type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--epoch_num_task", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--epoch_num_memory", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--hidden_dim", default=768, type=int, help="Output dimension of encoder.")
    parser.add_argument("--feature_dim", default=64, type=int, help="Output dimension of projection head.")
    parser.add_argument("--encoder_lr", default=1e-5, type=float,
                        help="The initial learning rate of encoder for AdamW.")
    parser.add_argument("--classifier_lr", default=1e-3, type=float,
                        help="The initial learning rate of classifier for AdamW.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--alpha", default=0.6, type=float, help="Bagging Hyperparameter.")
    parser.add_argument("--beta", default=0.2, type=float, help="Prototype weight.")
    parser.add_argument("--cl_temp", default=0.1, type=float, help="Temperature for contrastive learning.")
    parser.add_argument("--cl_lambda", default=0.8, type=float, help="Hyperparameter for contrastive learning.")
    parser.add_argument("--margin", default=0.15, type=float, help="Hyperparameter for margin loss.")
    parser.add_argument("--kd_temp", default=0.5, type=float, help="Temperature for knowledge distillation.")
    parser.add_argument("--kd_lambda1", default=0.7, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--kd_lambda2", default=0.5, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--gamma", default=2.0, type=float, help="Hyperparameter of focal loss.")
    parser.add_argument("--encode_style", default="emarker", type=str, help="Encode style of encoder.")
    parser.add_argument("--contrasive_size", default="64", type=int, help="Encode style of encoder.")

    parser.add_argument("--experiment_num", default=5, type=int)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--set_task_order", default=True, type=bool)
    parser.add_argument("--read_from_task_order", default=True, type=bool)
    parser.add_argument("--task_num", default=10, type=int)
    parser.add_argument("--memory_size", default=10, type=int, help="Memory size for each relation.")
    parser.add_argument("--early_stop_patient", default=10, type=int)

    args = parser.parse_args()
    logging.basicConfig(filename='SIM_training_memory.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    if args.cuda:
        device = "cuda:" + str(args.cuda_device)
    else:
        device = "cpu"

    args.device = device
    args.collate_fn = collate_fn

    tuner_params = nni.get_next_parameter()
    args = merge_parameter(args, tuner_params)
    BERTtokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                  additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased",
                                              additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    s = time.time()
    task_results, memory_results = [], []
    for i in range(args.experiment_num):

        set_random_seed(args)
        if args.dataset_name == "FewRel":
            processor = FewRelProcessor(args, tokenizer, BERTtokenizer)
        else:
            processor = tacredProcessor(args, tokenizer, BERTtokenizer)
        if args.set_task_order:
            processor.set_task_order("task_order.json", i, "relation_description.json")  ###赋值第i个实验的关系，不是实验数据
        if args.read_from_task_order:
            processor.set_read_from_order(i)  ###这个代表的实验i ，赋值第i个实验给processor

        task_acc, memory_acc = do_train(args, tokenizer, processor, i)
        print(f'{green_print}Result of experiment {i}:')
        print(f'task acc: {task_acc}')
        print(f'memory acc: {memory_acc}')
        print(f'Average: {sum(memory_acc) / len(memory_acc)}{default_print}')
        logging.info(f'Result of experiment {i}:')
        logging.info(f'task acc: {task_acc}')
        logging.info(f'memory acc: {memory_acc}')
        logging.info(f'Average: {sum(memory_acc) / len(memory_acc)}')
        task_results.append(task_acc)
        memory_results.append(memory_acc)
    e = time.time()

    task_results = torch.tensor(task_results, dtype=torch.float32)
    memory_results = torch.tensor(memory_results, dtype=torch.float32)
    print(f'All task result: {task_results.tolist()}')
    print(f'All memory result: {memory_results.tolist()}')
    logging.info(f'All task result: {task_results.tolist()}')
    logging.info(f'All memory result: {memory_results.tolist()}')

    task_results = torch.mean(task_results, dim=0).tolist()
    memory_results = torch.mean(memory_results, dim=0)
    final_average = torch.mean(memory_results).item()
    print(f'Final task result: {task_results}')
    print(f'Final memory result: {memory_results.tolist()}')
    print(f'Final average: {final_average}')
    print(f'Time cost: {e - s}s.')
    logging.info(f'Final task result: {task_results}')
    logging.info(f'Final memory result: {memory_results.tolist()}')
    logging.info(f'Final average: {final_average}')
    logging.info(f'Time cost: {e - s}s.')

    nni.report_final_result(final_average)