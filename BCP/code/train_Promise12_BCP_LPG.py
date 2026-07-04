import argparse
import math
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks1.net_factory import BCP_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d


from val_2D import test_single_volume
from dataloaders.dataset import (BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment, RandomGenerator)
from networks1.net_factory import BCP_net
from utils import ramps, losses,val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/lnn/zzy/BCP-main/data/Promise12', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--image_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
# label and unlabel

parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')
# patch size
parser.add_argument('--patch_size', type=int, default=128, help='patch_size')
parser.add_argument('--h_size', type=int, default=2, help='h_size')
parser.add_argument('--w_size', type=int, default=2, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
args = parser.parse_args()
dice_loss = losses.DiceLoss(n_classes=2)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_Promise12_LargestCC(segmentation):
    class_list = []
    for i in range(1, 2):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    HPH55_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(HPH55_largestCC).cuda()


def get_Promise12_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 2):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_Promise12_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_Promise12_2DLargestCC(probs)
    return probs


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "HPH55" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Promise12":
        ref_dict = {"3": 93, "7": 191 ,"11": 306, "14": 391, "18": 478, "35": 940}
    elif "MSD":
        ref_dict = {"3": 50, "5": 90 ,"7": 121, "22": 406}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]






# ===================== 优化后的工具函数 =====================

@torch.jit.script
def js_divergence_batch(patch: torch.Tensor, pool_patches: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    向量化批量计算JS散度（替代单样本循环）
    :param patch: 单个目标补丁 [d]
    :param pool_patches: hcpool补丁集合 [n, d]
    :param epsilon: 防止log(0)
    :return: 每个pool_patch与目标patch的JS散度 [n]
    """
    # 批量归一化为概率分布
    p = patch.float() / (patch.sum() + epsilon)
    q = pool_patches.float() / (pool_patches.sum(dim=1, keepdim=True) + epsilon)

    # 批量计算平均分布
    m = 0.5 * (p.unsqueeze(0) + q)

    # 批量计算KL散度
    kl_p = (p.unsqueeze(0) * torch.log((p.unsqueeze(0) + epsilon) / (m + epsilon))).sum(dim=1)
    kl_q = (q * torch.log((q + epsilon) / (m + epsilon))).sum(dim=1)

    # 批量计算JS散度
    js = 0.5 * (kl_p + kl_q)
    return js


@torch.jit.script
def l2_distance_batch(patch: torch.Tensor, pool_patches: torch.Tensor) -> torch.Tensor:
    """
    向量化批量计算L2距离（欧氏距离）
    :param patch: 单个目标补丁 [d]
    :param pool_patches: hcpool补丁集合 [n, d]
    :return: 每个pool_patch与目标patch的L2距离 [n]
    """
    # 批量计算L2距离 [n]
    l2_dist = torch.norm(pool_patches.float() - patch.float().unsqueeze(0), dim=1)
    return l2_dist


@torch.jit.script
def cosine_similarity_batch(patch: torch.Tensor, pool_patches: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    向量化批量计算余弦相似度（返回相似度值，范围[-1,1]）
    :param patch: 单个目标补丁 [d]
    :param pool_patches: hcpool补丁集合 [n, d]
    :param epsilon: 防止除以0
    :return: 每个pool_patch与目标patch的余弦相似度 [n]
    """
    # 归一化向量
    patch_norm = patch.float() / (torch.norm(patch.float()) + epsilon)
    pool_norm = pool_patches.float() / (torch.norm(pool_patches.float(), dim=1, keepdim=True) + epsilon)

    # 批量计算余弦相似度 [n]
    cos_sim = torch.sum(patch_norm.unsqueeze(0) * pool_norm, dim=1)
    return cos_sim


def js_divergence(patch1, patch2, epsilon=1e-8):
    """单样本JS散度（兼容原有逻辑，内部复用批量计算函数）"""
    pool = patch2.unsqueeze(0)
    return js_divergence_batch(patch1, pool, epsilon).item()


def l2_distance(patch1, patch2):
    """单样本L2距离"""
    pool = patch2.unsqueeze(0)
    return l2_distance_batch(patch1, pool).item()


def cosine_similarity(patch1, patch2, epsilon=1e-8):
    """单样本余弦相似度"""
    pool = patch2.unsqueeze(0)
    return cosine_similarity_batch(patch1, pool, epsilon).item()

# ===================== 优化后的LPMC函数 =====================
def LPMC(image_patch_1, image_patch_2,
         label_patch_1, label_patch_2,
         prediction_patch_1, prediction_patch_2,
         labeled_indices, args):
    """
    优化点：
    1. 减少张量克隆操作（仅必要时clone）
    2. 预计算有效索引（避免重复nonzero）
    3. 批量收集被替换补丁（减少hcpool.append循环）
    """
    hcpool = []
    replaced_patches = []  # 批量收集，减少append开销

    for idx in labeled_indices:
        if idx >= image_patch_1.shape[0]:
            continue

        # 预计算有效标签索引（合并维度计算）
        label_mean_1 = label_patch_1[idx].float().mean(dim=1)
        label_mean_2 = label_patch_2[idx].float().mean(dim=1)
        valid_indices_1 = torch.where(label_mean_1 != 0)[0]
        valid_indices_2 = torch.where(label_mean_2 != 0)[0]

        if len(valid_indices_1) < 2 or len(valid_indices_2) < 2:
            continue

        # 计算补丁均值并排序（优化排序逻辑）
        patch_means_1 = prediction_patch_1.float()[idx][valid_indices_1].mean(dim=1)
        patch_means_2 = prediction_patch_2.float()[idx][valid_indices_2].mean(dim=1)
        sorted_idx_1 = valid_indices_1[torch.argsort(patch_means_1)]
        sorted_idx_2 = valid_indices_2[torch.argsort(patch_means_2)]

        # 双向交换机制（优化交换逻辑）
        exchange_num = min(len(sorted_idx_1) // 2, len(sorted_idx_2) // 2)
        if exchange_num == 0:
            continue

        # 批量收集被替换补丁
        replaced_patches.append(image_patch_2[idx][sorted_idx_2[:exchange_num]].clone())
        replaced_patches.append(image_patch_1[idx][sorted_idx_1[:exchange_num]].clone())

        # 批量替换（减少循环内赋值）
        image_patch_2[idx][sorted_idx_2[:exchange_num]] = image_patch_1[idx][sorted_idx_1[-exchange_num:]].clone()
        image_patch_1[idx][sorted_idx_1[:exchange_num]] = image_patch_2[idx][sorted_idx_2[-exchange_num:]].clone()

    # 合并收集的补丁到hcpool
    if replaced_patches:
        hcpool = torch.cat(replaced_patches, dim=0).split(1)  # 恢复为单个补丁列表
        hcpool = [p.squeeze(0) for p in hcpool]

    return image_patch_1, image_patch_2, label_patch_1, label_patch_2, hcpool


# ===================== 优化后的UPCL函数 =====================
def UPCL(image_patch_1, image_patch_2,
         prediction_patch_1, prediction_patch_2,
         label_patch_1, label_patch_2,
         labeled_indices, args, hcpool):
    """
    核心优化点：
    1. 向量化JS散度计算（批量匹配hcpool补丁）
    2. 预计算标注样本的有效标签（避免重复遍历）
    3. 过滤空hcpool（提前返回）
    4. 减少张量克隆/复制操作
    5. 合并重复的索引边界检查
    """
    # 提前过滤空hcpool
    if not hcpool:
        return image_patch_1, image_patch_2, label_patch_1, label_patch_2

    # 预转换hcpool为张量（批量计算）
    hcpool_tensor = torch.stack(hcpool, dim=0)  # [n, d]
    num_patches = prediction_patch_1.shape[1]
    top_k = min(int(math.sqrt(num_patches)), num_patches // 2)
    if top_k < 1:
        return image_patch_1, image_patch_2, label_patch_1, label_patch_2

    # 预计算标注样本的有效标签（避免重复遍历）
    labeled_valid_labels = {}
    for l_idx in labeled_indices:
        if l_idx >= label_patch_1.shape[0]:
            continue
        valid_mask = label_patch_1[l_idx].float().mean(dim=1) != 0
        valid_l_indices = torch.where(valid_mask)[0]
        if len(valid_l_indices) > 0:
            labeled_valid_labels[l_idx] = label_patch_1[l_idx][valid_l_indices[0]].clone()

    # 无标注样本索引集合
    unlabeled = sorted(list(set(range(args.batch_size)) - labeled_indices))
    # 过滤无效索引
    unlabeled = [idx for idx in unlabeled if idx < image_patch_1.shape[0]]

    for idx in unlabeled:
        # 计算当前无标注样本所有补丁的预测均值（批量计算）
        means_1 = prediction_patch_1.float()[idx].mean(dim=1)
        means_2 = prediction_patch_2.float()[idx].mean(dim=1)

        # 提取均值最小的补丁索引（优化索引计算）
        min_idx_1 = torch.argmin(means_1)
        min_idx_2 = torch.argmin(means_2)

        # 处理image_patch_1
        current_patch_1 = image_patch_1[idx][min_idx_1]

        # ========== 批量计算散度匹配最相似补丁 ==========
        js_scores_1 = js_divergence_batch(current_patch_1, hcpool_tensor)
        best_idx_1 = torch.argmin(js_scores_1)

        # l2_scores_1 = l2_distance_batch(current_patch_1,hcpool_tensor)
        # best_idx_1 = torch.argmin(l2_scores_1)

        # cos_scores_1 = cosine_similarity_batch(current_patch_1,hcpool_tensor)
        # best_idx_1 = torch.argmax(cos_scores_1)

        best_match_patch_1 = hcpool_tensor[best_idx_1]
        # 替换（直接赋值，减少clone）
        image_patch_1[idx][min_idx_1] = best_match_patch_1

        # 同步替换标签（使用预计算的标注标签）
        if labeled_valid_labels:
            first_valid_label = next(iter(labeled_valid_labels.values()))
            label_patch_1[idx][min_idx_1] = first_valid_label


        # 处理image_patch_2
        current_patch_2 = image_patch_2[idx][min_idx_2]
        # ========== 批量计算散度匹配最相似补丁 ==========
        js_scores_2 = js_divergence_batch(current_patch_2, hcpool_tensor)
        best_idx_2 = torch.argmin(js_scores_2)

        # l2_scores_2 = l2_distance_batch(current_patch_2,hcpool_tensor)
        # best_idx_2 = torch.argmin(l2_scores_2)

        # cos_scores_2 = cosine_similarity_batch(current_patch_2, hcpool_tensor)
        # best_idx_2 = torch.argmax(cos_scores_2)

        best_match_patch_2 = hcpool_tensor[best_idx_2]
        # 替换（直接赋值）
        image_patch_2[idx][min_idx_2] = best_match_patch_2

        # 同步替换标签
        if labeled_valid_labels:
            first_valid_label = next(iter(labeled_valid_labels.values()))
            label_patch_2[idx][min_idx_2] = first_valid_label

    return image_patch_1, image_patch_2, label_patch_1, label_patch_2


# ===================== 优化后的主函数LPG =====================
def LPG(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args,
        label_idx):
    """
    优化点：
    1. 合并重复的维度检查
    2. 预计算索引集合（避免重复转换）
    3. 减少不必要的张量拼接/拆分
    4. 优化rearrange操作（减少维度变换次数）
    """
    # 1. 预计算索引集合（优化）
    labeled_indices = set(label_idx)
    total_indices = set(range(args.batch_size))
    unlabeled_indices = total_indices - labeled_indices

    # 2. 维度校验（合并逻辑）
    img_shape = volume_batch.shape
    assert img_shape[2] == args.h_size * args.patch_size and img_shape[3] == args.w_size * args.patch_size, \
        f"图像尺寸不匹配：期望({args.h_size * args.patch_size}, {args.w_size * args.patch_size})，实际({img_shape[2]}, {img_shape[3]})"

    # 3. 批量重排维度（减少重复调用）
    rearrange_kwargs = {'p1': args.patch_size, 'p2': args.patch_size, 'h': args.h_size, 'w': args.w_size}
    # 预测结果重排
    prediction_patch_1 = rearrange(outputs1_max, 'b (h p1) (w p2) -> b (h w) (p1 p2)', **rearrange_kwargs)
    prediction_patch_2 = rearrange(outputs2_max, 'b (h p1) (w p2) -> b (h w) (p1 p2)', **rearrange_kwargs)
    # 图像重排
    image_patch_1 = rearrange(volume_batch.squeeze(1), 'b (h p1) (w p2) -> b (h w) (p1 p2)', **rearrange_kwargs)
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b (h p1) (w p2) -> b (h w) (p1 p2)', **rearrange_kwargs)
    # 标签重排
    label_patch_1 = rearrange(label_batch, 'b (h p1) (w p2) -> b (h w) (p1 p2)', **rearrange_kwargs)
    label_patch_2 = rearrange(label_batch_strong, 'b (h p1) (w p2) -> b (h w) (p1 p2)', **rearrange_kwargs)

    # 4. LPMC处理（优化后）
    print("LPMC")
    image_patch_1, image_patch_2, label_patch_1, label_patch_2, hcpool = LPMC(
        image_patch_1, image_patch_2,
        label_patch_1, label_patch_2,
        prediction_patch_1, prediction_patch_2,
        labeled_indices, args
    )

    # 5. UPCL处理（优化后）
    print("UPCL")
    image_patch_1, image_patch_2, label_patch_1, label_patch_2 = UPCL(
        image_patch_1, image_patch_2,
        prediction_patch_1, prediction_patch_2,
        label_patch_1, label_patch_2,
        labeled_indices, args, hcpool
    )

    # 6. 还原维度（优化rearrange参数复用）
    image_patch_last = rearrange(image_patch_1, 'b (h w) (p1 p2) -> b (h p1) (w p2)', **rearrange_kwargs)
    label_patch_last = rearrange(label_patch_1, 'b (h w) (p1 p2) -> b (h p1) (w p2)', **rearrange_kwargs)

    return image_patch_last, label_patch_last
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            # writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                # writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                # writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                # writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes - 1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                # writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    # writer.close()


def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model_1 = BCP_net(in_chns=1, class_num=num_classes)
    model_2 = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(ema_model, pre_trained_model)
    load_net_opt(model_1, optimizer1, pre_trained_model)
    load_net_opt(model_2, optimizer2, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    # writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    times =[]
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:

        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[
                                 args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[
                                                                                      args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_Promise12_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_Promise12_masks(pre_b, nms=1)
                pre_a_s = ema_model(uimg_a_s)
                pre_b_s = ema_model(uimg_b_s)
                plab_a_s = get_Promise12_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_Promise12_masks(pre_b_s, nms=1)
                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0)

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)
            ########################33

            # Model1 Loss
            out_unl_1 = model_1(net_input_unl_1)
            out_l_1 = model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False)
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2 = model_2(net_input_unl_2)
            out_l_2 = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False)
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            # Model1 & Model2 Cross Pseudo Supervision
            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))


            # 构造LPG所需的标签数据
            epoch_start = time.time()
            label_batch = torch.cat([lab_b, plab_a], dim=0)  # 弱增强标签流
            label_batch_strong = torch.cat([lab_b_s, plab_a_s], dim=0)  # 强增强标签流
            # 构造标注索引 (假设当前batch前一半为标注数据)
            label_idx = list(range(args.labeled_bs))
            # image_patch_last = ABD_R_BCP(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)
            # image_patch_last = SRM(out_max_1, out_max_2,net_input_1, net_input_2,label_batch,label_batch_strong,args,label_idx)

            image_patch_last = LPG(out_max_1, out_max_2, net_input_1, net_input_2, label_batch, label_batch_strong,
                                   args, label_idx)

            image_output_1 = model_1(image_patch_last[0].unsqueeze(1))
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_output_2 = model_2(image_patch_last[0].unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            # Model1 & Model2 Second Step Cross Pseudo Supervision
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4
            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()


            iter_num += 1
            epoch_time = time.time() - epoch_start
            times.append(epoch_time)
            update_model_ema(model_1, ema_model, 0.99)

            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/model1_loss', loss_1, iter_num)
            # writer.add_scalar('info/model2_loss', loss_2, iter_num)
            # writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            # writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            # writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            # writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            # writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_1,
                                                  classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes - 1):
                #     writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                # writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(
                        best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)

                logging.info('iteration %d : model1_mean_dice : %f' % (iter_num, performance1))
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_2,
                                                  classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes - 1):
                #     writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                # writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(
                        best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode_path)
                    torch.save(model_2.state_dict(), save_best_path)

                logging.info('iteration %d : model2_mean_dice : %f' % (iter_num, performance2))
                model_2.train()

            if iter_num >= max_iterations:
                break
        print('avg time per epoch:', sum(times))
        times.clear()
        if iter_num >= max_iterations:
            iterator.close()
            break


    # writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "/home/lnn/zzy/BCP-main/code/model/BCP_LPP/Promise12/Promise12_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "/home/lnn/zzy/BCP-main/code/model/BCP_LPP/Promise12/Promise12_{}_{}_labeled/self_train_JS".format(args.exp,args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    # shutil.copy('/home/lnn/zzy/BCP-main/code/model/BCP/Promise12_{}_{}_labeled//train_Promise12_BCP.py', self_snapshot_path)

    # # Pre_trainPromise12_BCP_LPG_train.py
    # logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)





