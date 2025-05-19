import os
import copy
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity


from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import logging

# def set_random_seed(args):
#     # 移除所有设置随机种子的代码
#     # 允许随机性和非确定性行为
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False

# def set_random_seed(args,i):
#     seed = args.seed + i * 101
#     torch.manual_seed(seed)
#     if torch.cuda.is_available() and args.cuda:
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    index = []
    istraindata = 0
    if type(batch[0]) == tuple:

        index = [b[1] for b in batch]
        batch1 = [b[0] for b in batch]
        batch = batch1
        istraindata = 1
    max_len = max([len(sample['input_ids']) for sample in batch])
    # 补零
    input_ids = [sample['input_ids'] + [0] * (max_len - len(sample['input_ids'])) for sample in batch]
    #编码器输入
    attention_mask = [[1.0] * len(sample['input_ids']) + [0.0] * (max_len - len(sample['input_ids'])) for sample in batch]
    h_index = [sample['h_index'] for sample in batch]
    t_index = [sample['t_index'] for sample in batch]
    # MASK_index = [sample.get('MASK_index',sample["input_ids"].index(103)) for sample in batch]  # Use get method with default value
    labels = [sample['label'] for sample in batch]
    relations = [sample['relation'] for sample in batch]


    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    h_index = torch.tensor(h_index, dtype=torch.long)
    t_index = torch.tensor(t_index, dtype=torch.long)
    # MASK_index = torch.tensor(MASK_index, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    if istraindata:
        output = (input_ids, attention_mask, h_index, t_index,  labels, relations , index)
        # 返回索引来寻找no_self的数据
    else:
        output = (input_ids, attention_mask, h_index, t_index, labels, relations)

    return output


def compute_cos_sim(tensor_a, tensor_b):
    """
    tensor_a [k, m]
    tensor_b [n, m]
    """
    norm_a = torch.norm(tensor_a, dim=1).unsqueeze(1)  # [k, 1]
    norm_b = torch.norm(tensor_b, dim=1).unsqueeze(0)  # [1, n]
    cos_sim = torch.mm(tensor_a, tensor_b.T) / torch.mm(norm_a, norm_b)  # [k, n]
    return cos_sim


# def save_checkpoint(args, model, i_exp, i_task, name):
#     if model is None:
#         raise Exception(f'The best model of task {i_task} is None.')
#     torch.save(model.state_dict(),
#                os.path.join(args.checkpoint_dir, args.dataset_name, f"Exp{i_exp}", f"{i_task}_{name}.pkl"))
def save_checkpoint(args, model, i_exp, i_task, name):
    if model is None:
        raise Exception(f'The best model of task {i_task} is None.')

    # 检查目录是否存在，如果不存在则创建
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name, f"Exp{i_exp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{i_task}_{name}.pkl"))

def get_static_prototypes(args, model, data, reldata_len):
    model.eval()
    dataloader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)

    hiddens, features = [], []
    for i, batch in enumerate(dataloader):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }

        with torch.no_grad():
            hidden, _ = model(**inputs)
            hiddens.append(hidden)

    with torch.no_grad():
        hiddens = torch.cat(hiddens, dim=0)
        hidden_tensors = []
        current_idx = 0
        for i in range(len(reldata_len)):
            rel_len = reldata_len[i]
            rel_hiddens = torch.narrow(hiddens, 0, current_idx, rel_len)
            hidden_proto = torch.mean(rel_hiddens, dim=0)
            hidden_tensors.append(hidden_proto)
            current_idx += rel_len
        hidden_tensors = torch.stack(hidden_tensors, dim=0)
        hidden_tensors = torch.nn.LayerNorm([args.hidden_dim]).to(args.device)(hidden_tensors)
        feature_tensors = model.get_low_dim_feature(hidden_tensors)
    return hidden_tensors, feature_tensors
def get_prototypes(args, model, data, reldata_len, relations):
    model.eval()
    dataloader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    hiddens, features,MASK_hiddens = [], [],[]
    for i, batch in enumerate(dataloader):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }
        with torch.no_grad():
            hidden, _= model(**inputs)
            hiddens.append(hidden)
    # 不使用梯度计算
    with torch.no_grad():
        Rel_descriptioncls_hiddens= model(**relations)
        Rel_descriptioncls_hiddens = torch.split(Rel_descriptioncls_hiddens, 1, dim=0)
    with torch.no_grad():
        hiddens = torch.cat(hiddens, dim=0)
        hidden_tensors = []
        current_idx = 0
        j = 0
        for i in range(len(reldata_len)):
            rel_len = reldata_len[i]
            rel_hiddens = torch.narrow(hiddens, 0, current_idx, rel_len)

            cls_representations = Rel_descriptioncls_hiddens[j].to(args.device)
            Multi_Head_input = torch.cat([rel_hiddens, cls_representations], dim=0)
            j += 1
            hidden_proto = multi_head(Multi_Head_input, True, 12, args)  # 增加
            # hidden_proto = torch.mean(rel_hiddens, dim=0)
            hidden_tensors.append(hidden_proto)
            current_idx += rel_len
        # hidden_tensors = torch.stack(hidden_tensors, dim=0)
        hidden_tensors = torch.cat(hidden_tensors, dim=0)
        hidden_tensors = torch.nn.LayerNorm([args.hidden_dim]).to(args.device)(hidden_tensors)
        feature_tensors = model.get_low_dim_feature(hidden_tensors)  ##不降维度


    return hidden_tensors, feature_tensors

def Memory_train_get_prototypes(args, model, data, reldata_len, relations):
    model.train()
    dataloader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    hiddens, features,MASK_hiddens = [], [],[]
    for i, batch in enumerate(dataloader):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }
        hidden, _= model(**inputs)
        hiddens.append(hidden)
    # 不使用梯度计算
    with torch.no_grad():
        Rel_descriptioncls_hiddens= model(**relations)
        Rel_descriptioncls_hiddens = torch.split(Rel_descriptioncls_hiddens, 1, dim=0)

    hiddens = torch.cat(hiddens, dim=0)
    hidden_tensors = []
    current_idx = 0
    j = 0
    for i in range(len(reldata_len)):
        rel_len = reldata_len[i]
        rel_hiddens = torch.narrow(hiddens, 0, current_idx, rel_len)

        cls_representations = Rel_descriptioncls_hiddens[j].to(args.device)
        Multi_Head_input = torch.cat([rel_hiddens, cls_representations], dim=0)
        j += 1
        hidden_proto = multi_head(Multi_Head_input, True, 12, args)  # 增加
        # hidden_proto = torch.mean(rel_hiddens, dim=0)
        hidden_tensors.append(hidden_proto)
        current_idx += rel_len
    # hidden_tensors = torch.stack(hidden_tensors, dim=0)
    hidden_tensors = torch.cat(hidden_tensors, dim=0)
    hidden_tensors = torch.nn.LayerNorm([args.hidden_dim]).to(args.device)(hidden_tensors)
    feature_tensors = model.get_low_dim_feature(hidden_tensors)


    return hidden_tensors, feature_tensors


def multi_head(input, mask_last_one=True, num_head=12, args=None):
    input = input.view(1, -1, num_head, int(args.hidden_dim / num_head))
    # (batch_size, seq_Len, head, dim)
    q_transpose = input.permute(0, 2, 1, 3)
    k_transpose = input.permute(0, 2, 1, 3)
    v_transpose = input.permute(0, 2, 1, 3) ## [1, 12, seq, hidden/12]
    q_transpose *= (float(args.hidden_dim / num_head) ** -0.5)
    # make it [B, H, N, N]
    dot_product = torch.matmul(q_transpose, k_transpose.permute(0, 1, 3, 2))  ##[1, 12, seq, seq]
    if mask_last_one:
        dot_product[:, :, :, -1] = -1e7
    weights = F.softmax(dot_product, dim=-1)
    # output is [B, H, N, V]
    weighted_output = torch.matmul(weights, v_transpose)
    output_transpose = weighted_output.permute(0, 2, 1, 3).contiguous()
    output_transpose = output_transpose.view(-1, args.hidden_dim)
    return output_transpose[-1:, ]


def memory_select(args, model, data, data_len):
    model.eval()
    dataloader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False,
                            shuffle=False)
    hiddens, memory, memory_len,mask_hiddens = [], [], [],[]
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }

        with torch.no_grad():
            hidden, _ = model(**inputs)
            hiddens.append(hidden.cpu())

    hiddens = np.concatenate(hiddens, axis=0)                              ####原本的####
    current_len = 0
    for i in range(args.relnum_per_task):
        rel_len = data_len[i]
        kmdata = hiddens[current_len: current_len + rel_len]
        k = min(args.memory_size, rel_len)
        kmeans = KMeans(n_clusters=k, random_state=0)
        distances = kmeans.fit_transform(kmdata)  # n_sample* n_clusters

        rel_data = data[current_len: current_len + rel_len]
        for j in range(k):
            select_idx = np.argmin(distances[:, j])  # [k]
            memory.append(rel_data[select_idx])

        current_len += rel_len
        memory_len.append(k)
    return memory, memory_len

    #
    # mask_hiddens = np.concatenate(mask_hiddens, axis=0)                          #####MASK更换的
    # current_len = 0
    # for i in range(args.relnum_per_task):
    #     rel_len = data_len[i]
    #     kmdata = mask_hiddens[current_len: current_len + rel_len]
    #     k = min(args.memory_size, rel_len)
    #     kmeans = KMeans(n_clusters=k, random_state=0)
    #     distances = kmeans.fit_transform(kmdata)
    #
    #     rel_data = data[current_len: current_len + rel_len]
    #     for j in range(k):
    #         select_idx = np.argmin(distances[:, j])  # [k]
    #         memory.append(rel_data[select_idx])
    #
    #     current_len += rel_len
    #     memory_len.append(k)
    # return memory, memory_len


def get_augmentative_data(args, data, data_len):
    logging.basicConfig(filename='SIM_training_memory.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    index = 0
    data_double = copy.deepcopy(data)
    for i in range(len(data_len)):
        rel_data = data[index: index + data_len[i]]
        index += data_len[i]

        rel_data_temp = copy.deepcopy(rel_data)
        random.shuffle(rel_data_temp)

        for j in range(data_len[i]):
            sample1, sample2 = rel_data[j], rel_data_temp[j]
            input_ids1 = sample1['input_ids'][1:-1]
            input_ids2 = sample2['input_ids'][1:-1]
            h_tokens = input_ids2[input_ids2.index(30522) + 1:input_ids2.index(30523)]
            t_tokens = input_ids2[input_ids2.index(30524) + 1:input_ids2.index(30525)]
            input_ids1[input_ids1.index(30522) + 1: input_ids1.index(30523)] = h_tokens
            input_ids1[input_ids1.index(30524) + 1: input_ids1.index(30525)] = t_tokens
            input_ids1[0:input_ids1.index(103)] = h_tokens
            input_ids1[input_ids1.index(103) + 1:input_ids1.index(102, input_ids1.index(103))] = t_tokens

            input_ids = [101] + input_ids1 + [102]
            if(input_ids[input_ids.index(101)+1:input_ids.index(103)] != h_tokens or
               input_ids[input_ids.index(103)+1:input_ids.index(102, input_ids.index(101))] != t_tokens):
                print("get_augmentative_data：error")
                logging.info("get_augmentative_data：error")
            h_index = input_ids.index(30522)
            t_index = input_ids.index(30524)
            data_double.append({
                "input_ids": input_ids,
                'h_index': h_index,
                't_index': t_index,
                'label': sample1['label'],
                'relation': sample1['relation']
            })

    aug_data = []
    add_data1 = copy.deepcopy(data_double)
    random.shuffle(add_data1)
    aug_data1 = rel_data_augment(args, data_double, add_data1)
    aug_data += data_double
    aug_data += aug_data1
    return aug_data

def get_augmentative_memory_data(args, data, data_len,enc):
    logging.basicConfig(filename='SIM_training_memory.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    index = 0
    data_double = copy.deepcopy(data)
    for i in range(len(data_len)):
        rel_data = data[index: index + data_len[i]]
        batch2 = args.collate_fn(rel_data)
        batch2_inputs = {
            'input_ids': batch2[0].to(args.device),
            'attention_mask': batch2[1].to(args.device),
            'h_index': batch2[2].to(args.device),
            't_index': batch2[3].to(args.device),
        }
        with torch.no_grad():
            # Get hidden2 from encoder
            hidden2, _ = enc(**batch2_inputs)

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(hidden2.unsqueeze(1), hidden2.unsqueeze(0), dim=2)
        rel_data_temp = []

        for idx in range(similarity_matrix.size(0)):
            # Get cosine similarities for the current row
            similarities = similarity_matrix[idx]
            # Exclude self-similarity by setting diagonal to a very low value
            similarities[idx] = -float('inf')
            # Get indices of top 3 most similar vectors
            top_k_indices = torch.topk(similarities, k=3).indices
            # Randomly select one among the top 3
            selected_index = top_k_indices[torch.randint(0, 3, (1,)).item()]
            rel_data_temp.append(copy.deepcopy(rel_data[selected_index]))
        index += data_len[i]
        for j in range(data_len[i]):
            sample1, sample2 = rel_data[j], rel_data_temp[j]
            input_ids1 = sample1['input_ids'][1:-1]
            input_ids2 = sample2['input_ids'][1:-1]
            h_tokens = input_ids2[input_ids2.index(30522) + 1:input_ids2.index(30523)]
            t_tokens = input_ids2[input_ids2.index(30524) + 1:input_ids2.index(30525)]
            input_ids1[input_ids1.index(30522) + 1: input_ids1.index(30523)] = h_tokens
            input_ids1[input_ids1.index(30524) + 1: input_ids1.index(30525)] = t_tokens
            input_ids1[0:input_ids1.index(103)] = h_tokens
            input_ids1[input_ids1.index(103) + 1:input_ids1.index(102, input_ids1.index(103))] = t_tokens

            input_ids = [101] + input_ids1 + [102]
            if(input_ids[input_ids.index(101)+1:input_ids.index(103)] != h_tokens or
               input_ids[input_ids.index(103)+1:input_ids.index(102, input_ids.index(101))] != t_tokens):
                print("get_augmentative_data：error")
                logging.info("get_augmentative_data：error")
            h_index = input_ids.index(30522)
            t_index = input_ids.index(30524)
            data_double.append({
                "input_ids": input_ids,
                'h_index': h_index,
                't_index': t_index,
                'label': sample1['label'],
                'relation': sample1['relation']
            })

    aug_data = []
    add_data1 = copy.deepcopy(data_double)
    random.shuffle(add_data1)
    aug_data1 = rel_data_augment(args, data_double, add_data1)
    aug_data += data_double
    aug_data += aug_data1
    return aug_data

# def get_augmentative_data(args, data, data_len):
#     index = 0
#     data_double = copy.deepcopy(data)
#     for i in range(len(data_len)):
#         rel_data = data[index: index+data_len[i]]
#         index += data_len[i]
#
#         rel_data_temp = copy.deepcopy(rel_data)
#         random.shuffle(rel_data_temp)
#
#         for j in range(data_len[i]):
#             sample1, sample2 = rel_data[j], rel_data_temp[j]
#             input_ids1 = sample1['input_ids'][1:-1]
#             input_ids2 = sample2['input_ids'][1:-1]
#             h_tokens = input_ids2[input_ids2.index(30522)+1:input_ids2.index(30523)]
#             t_tokens = input_ids2[input_ids2.index(30524)+1:input_ids2.index(30525)]
#             input_ids1[input_ids1.index(30522)+1: input_ids1.index(30523)] = h_tokens
#             input_ids1[input_ids1.index(30524)+1: input_ids1.index(30525)] = t_tokens
#             input_ids = [101] + input_ids1 + [102]
#             h_index = input_ids.index(30522)
#             t_index = input_ids.index(30524)
#             data_double.append({
#                 "input_ids": input_ids,
#                 'h_index': h_index,
#                 't_index': t_index,
#                 'label': sample1['label'],
#                 'relation': sample1['relation']
#             })
#
#
#     aug_data = []
#     add_data1 = copy.deepcopy(data_double)
#     random.shuffle(add_data1)
#     aug_data1 = rel_data_augment(args, data_double, add_data1)
#     aug_data += data_double
#     aug_data += aug_data1
#     return aug_data

def rel_data_augment(args, rel_data1, rel_data2):
    aug_data = []
    length = min(len(rel_data1), len(rel_data2))
    logging.basicConfig(filename='SIM_training_memory.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    for i in range(length):
        sample1, sample2 = rel_data1[i], rel_data2[i]
        input_ids1 = sample1['input_ids'][1:-1]
        #input_ids2 = sample2['input_ids'][1:-1]
        input_ids2 = sample2['input_ids'][0:-1]
        input_ids2.remove(30522)
        input_ids2.remove(30523)
        input_ids2.remove(30524)
        input_ids2.remove(30525)
        # 找到"101"到"103"对应位置的所有字符串并删除
        start_idx = input_ids2.index(101)  # 找到"101"的位置
        end_idx = input_ids2.index(102, start_idx)  # 找到"102"的位置

        # 从列表中删除"101"到"103"对应位置的所有元素
        del input_ids2[start_idx:end_idx + 1]
        if args.dataset_name == "FewRel":
            length = 512 - 6 - len(input_ids1)
            input_ids2 = input_ids2[:length]
        if i % 2 == 0:
            input_ids = [101] + input_ids1 + input_ids2 + [102]
            h_index = sample1['h_index']
            t_index = sample1['t_index']
            if(h_index != input_ids.index(30522) or t_index != input_ids.index(30524)):
                print(h_index, input_ids.index(30522))
                print(t_index, input_ids.index(30524))
                print("rel_data_augment：error")
                h_index = input_ids.index(30522)
                t_index = input_ids.index(30524)
                logging.info(f"h_index={h_index}, t_index={t_index},rel_data_augment：error")
        else:
            end_idx = input_ids1.index(102, start_idx)  # 找到"102"的位置；因为id1被删了cls；不需要找101

            # 从列表中删除"101"到"103"对应位置的所有元素
            prompt = input_ids1[0:end_idx + 1]
            del input_ids1[0:end_idx + 1]
            input_ids = [101] + prompt + input_ids2 + input_ids1 + [102]
            h_index = sample1['h_index'] + len(input_ids2)
            t_index = sample1['t_index'] + len(input_ids2)
            if(h_index != input_ids.index(30522) or t_index != input_ids.index(30524)):
                print(h_index, input_ids.index(30522))
                print(t_index, input_ids.index(30524))
                print("rel_data_augment：error")
                h_index = input_ids.index(30522)
                t_index = input_ids.index(30524)
                logging.info(f"h_index={h_index}, t_index={t_index},rel_data_augment：error")
        aug_data.append({
            "input_ids": input_ids,
            'h_index': h_index,
            't_index': t_index,
            'label': sample1['label'],
            'relation': sample1['relation']
        })
    return aug_data

# def rel_data_augment(args, rel_data1, rel_data2):
#     aug_data = []
#     length = min(len(rel_data1), len(rel_data2))
#     for i in range(length):
#         sample1, sample2 = rel_data1[i], rel_data2[i]
#         input_ids1 = sample1['input_ids'][1:-1]
#         input_ids2 = sample2['input_ids'][1:-1]
#         input_ids2.remove(30522)
#         input_ids2.remove(30523)
#         input_ids2.remove(30524)
#         input_ids2.remove(30525)
#         if args.dataset_name == "FewRel":
#             length = 512-2-len(input_ids1)
#             input_ids2 = input_ids2[:length]
#         if i % 2 == 0:
#             input_ids = [101] + input_ids1 + input_ids2 + [102]
#             h_index = sample1['h_index']
#             t_index = sample1['t_index']
#         else:
#             input_ids = [101] + input_ids2 + input_ids1 + [102]
#             h_index = sample1['h_index'] + len(input_ids2)
#             t_index = sample1['t_index'] + len(input_ids2)
#         aug_data.append({
#             "input_ids": input_ids,
#             'h_index': h_index,
#             't_index': t_index,
#             'label': sample1['label'],
#             'relation': sample1['relation']
#         })
#     return aug_data


def get_aca_data(args, data, data_len, current_relations):
    index = 0
    rel_datas = []
    for i in range(len(data_len)):
        rel_data = data[index: index + data_len[i]]
        rel_datas.append(rel_data)
        index += data_len[i]

    rel_id = args.seen_rel_num
    aca_data = copy.deepcopy(data)
    idx = args.relnum_per_task // 2
    for i in range(args.relnum_per_task // 2):
        j = i + idx

        datas1 = rel_datas[i]
        datas2 = rel_datas[j]
        L = 5
        for data1, data2 in zip(datas1, datas2):
            input_ids1 = data1['input_ids'][1:-1]
            e11 = input_ids1.index(30522);
            e12 = input_ids1.index(30523)
            e21 = input_ids1.index(30524);
            e22 = input_ids1.index(30525)
            if e21 <= e11:
                continue
            input_ids1_sub = input_ids1[max(0, e11 - L): min(e12 + L + 1, e21)]

            input_ids2 = data2['tokens'][1:-1]
            e11 = input_ids2.index(30522);
            e12 = input_ids2.index(30523)
            e21 = input_ids2.index(30524);
            e22 = input_ids2.index(30525)
            if e21 <= e11:
                continue

            token2_sub = input_ids2[max(e12 + 1, e21 - L): min(e22 + L + 1, len(input_ids2))]

            input_ids = [101] + input_ids1_sub + token2_sub + [102]
            aca_data.append({
                'input_ids': input_ids,
                'h_index': input_ids.index(30522),
                't_index': input_ids.index(30524),
                'label': rel_id,
                'relation': data1['relation'] + '-' + data2['relation']
            })

            for index in [30522, 30523, 30524, 30525]:
                assert index in input_ids and input_ids.count(index) == 1

        rel_id += 1

    for i in range(len(current_relations)):
        if current_relations[i] in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spous',
                                    'per:alternate_names', 'per:other_family']:
            continue

        for data in rel_datas[i]:
            input_ids = data['input_ids']
            e11 = input_ids.index(30522);
            e12 = input_ids.index(30523)
            e21 = input_ids.index(30524);
            e22 = input_ids.index(30525)
            input_ids[e11] = 30524;
            input_ids[e12] = 30525
            input_ids[e21] = 30522;
            input_ids[e22] = 30523

            aca_data.append({
                'input_ids': input_ids,
                'h_index': input_ids.index(30522),
                't_index': input_ids.index(30524),
                'label': rel_id,
                'relation': data1['relation'] + '-reverse'
            })

            for index in [30522, 30523, 30524, 30525]:
                assert index in input_ids and input_ids.count(index) == 1
        rel_id += 1
    return aca_data
class SupConLoss(torch.nn.Module):
    def __init__(self, device, temperature=0.1, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        labels = torch.reshape(labels, (-1, 1))
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        # compute log_prob  #############################
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive and loss
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        loss = loss.view(anchor_count, batch_size).mean(dim=0)
        return loss