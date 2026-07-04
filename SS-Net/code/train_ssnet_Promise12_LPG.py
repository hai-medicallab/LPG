import argparse
import logging
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from thop import profile, clever_format
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment)
from networks1.net_factory import net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/lnn/zzy/SS-Net-main/data/Promise12', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SS_LPP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
#LPP Module
parser.add_argument('--patch_size', type=int, default=64, help='patch_size')
parser.add_argument('--h_size', type=int, default=4, help='h_size')
parser.add_argument('--w_size', type=int, default=4, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
parser.add_argument('--image_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')

args = parser.parse_args()



import torch
import math
from einops import rearrange  # 确保einops已安装


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


def js_divergence(patch1, patch2, epsilon=1e-8):
    """单样本JS散度（兼容原有逻辑，内部复用批量计算函数）"""
    pool = patch2.unsqueeze(0)
    return js_divergence_batch(patch1, pool, epsilon).item()


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

        # ========== 批量计算JS散度匹配最相似补丁 ==========
        # 处理image_patch_1
        current_patch_1 = image_patch_1[idx][min_idx_1]
        js_scores_1 = js_divergence_batch(current_patch_1, hcpool_tensor)
        best_idx_1 = torch.argmin(js_scores_1)
        best_match_patch_1 = hcpool_tensor[best_idx_1]
        # 替换（直接赋值，减少clone）
        image_patch_1[idx][min_idx_1] = best_match_patch_1

        # 同步替换标签（使用预计算的标注标签）
        if labeled_valid_labels:
            first_valid_label = next(iter(labeled_valid_labels.values()))
            label_patch_1[idx][min_idx_1] = first_valid_label

        # 处理image_patch_2
        current_patch_2 = image_patch_2[idx][min_idx_2]
        js_scores_2 = js_divergence_batch(current_patch_2, hcpool_tensor)
        best_idx_2 = torch.argmin(js_scores_2)
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

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Promise12":
        ref_dict = {"3": 93, "7": 191 ,"11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model = model.cuda()
    input_size = (1, *[256, 256])
    params_m, gflops = computer_complexity_with_thop(model, input_size)
    # model2 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                WeakStrongAugment(args.image_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    # model2.train()
    prototype_memory = feature_memory.FeatureMemory(elements_per_class=32, n_classes=num_classes)
    prototype_memory2 = feature_memory.FeatureMemory(elements_per_class=32, n_classes=num_classes)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    adv_loss = losses.VAT2d(epi=args.magnitude,num_classes=num_classes)

    # writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for epoch_idx in iterator:

        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            with torch.no_grad():
                outputs, embedding= model(volume_batch)
                outputs_soft = F.softmax(outputs, dim=1)
                outputs_s, embedding_s = model(volume_batch_strong)
                outputs_soft_s = F.softmax(outputs_s, dim=1)


            #use LPP reectify===============================================
            epoch_start = time.time()
            label_idx = list(range(args.labeled_bs))
            out_max_1 = torch.max(outputs_soft.detach(),dim= 1)[0]
            out_max_2 = torch.max(outputs_soft_s.detach(), dim=1)[0]
            del outputs,embedding,outputs_soft,outputs_s,embedding_s,outputs_soft_s
            torch.cuda.empty_cache()
            # image_patch_last = SRM(out_max_1, out_max_2, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args, label_idx)
            image_patch_last,label_patch_last = LPG(out_max_1, out_max_2, volume_batch, volume_batch_strong, label_batch,
                                    label_batch_strong, args, label_idx)
            epoch_time = time.time() - epoch_start
            print("time is:", epoch_time)
            times.append(epoch_time)
            del out_max_1,out_max_2,volume_batch_strong,label_batch_strong
            torch.cuda.empty_cache()
            outputs, embedding = model(image_patch_last.unsqueeze(1))
            outputs_soft = F.softmax(outputs, dim=1)
            del image_patch_last
            torch.cuda.empty_cache()
            # outputs_s, embedding_s = model(image_patch_last.unsqueeze(1))
            # outputs_soft_s = F.softmax(outputs_s, dim=1)

            labeled_features = embedding[:args.labeled_bs,...]
            unlabeled_features = embedding[args.labeled_bs:,...]
            y = outputs_soft[:args.labeled_bs]
            true_labels = label_patch_last[:args.labeled_bs]
            # true_labels = label_batch[:args.labeled_bs]
            del embedding
            torch.cuda.empty_cache()
            # labeled_features_s = embedding_s[:args.labeled_bs, ...]
            # unlabeled_features_s = embedding_s[args.labeled_bs:, ...]
            # y_s = outputs_soft_s[:args.labeled_bs]
            # true_labels_s = label_batch_strong[:args.labeled_bs]


            _, prediction_label = torch.max(y, dim=1)
            _, pseudo_label = torch.max(outputs_soft[args.labeled_bs:], dim=1)  # Get pseudolabels
            # _, prediction_label_s = torch.max(y_s, dim=1)
            # _, pseudo_label_s = torch.max(outputs_soft_s[args.labeled_bs:], dim=1)  # Get pseudolabels
            del outputs_soft
            torch.cuda.empty_cache()
            mask_prediction_correctly = ((prediction_label == true_labels).float() * (prediction_label > 0).float()).bool()
            # mask_prediction_correctly_s = ((prediction_label_s == true_labels_s).float() * (prediction_label_s > 0).float()).bool()
            ### select the correct predictions and ignore the background class
            del prediction_label
            torch.cuda.empty_cache()
            # Apply the filter mask to the features and its labels
            labeled_features = labeled_features.permute(0, 2, 3, 1)
            labels_correct = true_labels[mask_prediction_correctly]
            labeled_features_correct = labeled_features[mask_prediction_correctly, ...]
            # labeled_features_s = labeled_features_s.permute(0, 2, 3, 1)
            # labels_correct_s = true_labels_s[mask_prediction_correctly_s]
            # labeled_features_correct_s = labeled_features_s[mask_prediction_correctly_s, ...]
            del mask_prediction_correctly
            torch.cuda.empty_cache()
            # get projected features
            with torch.no_grad():
                model.eval()
                proj_labeled_features_correct = model.projection_head(labeled_features_correct)
                model.train()
                # model2.eval()
                # proj_labeled_features_correct_s = model2.projection_head(labeled_features_correct_s)
                # model2.train()

            # updated memory bank
            prototype_memory.add_features_from_sample_learned(model, proj_labeled_features_correct, labels_correct)
            labeled_features_all = labeled_features.reshape(-1, labeled_features.size()[-1])
            labeled_labels = true_labels.reshape(-1)
            # prototype_memory2.add_features_from_sample_learned(model2, proj_labeled_features_correct_s, labels_correct_s)
            # labeled_features_all_s = labeled_features_s.reshape(-1, labeled_features_s.size()[-1])
            # labeled_labels_s = true_labels_s.reshape(-1)
            # get predicted features
            proj_labeled_features_all = model.projection_head(labeled_features_all)
            pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)
            # proj_labeled_features_all2 = model2.projection_head(labeled_features_all_s)
            # pred_labeled_features_all2 = model2.prediction_head(proj_labeled_features_all2)

            # Apply contrastive learning loss
            loss_contr_labeled = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_labeled_features_all, labeled_labels, num_classes, prototype_memory.memory)
            # loss_contr_labeled2 = contrastive_losses.contrastive_class_to_class_learned_memory(model2,
            #                                                                                   pred_labeled_features_all2,
            #                                                                                   labeled_labels_s,
            #                                                                                   num_classes,
            #                                                                                   prototype_memory2.memory)
            unlabeled_features = unlabeled_features.permute(0, 2, 3, 1).reshape(-1, labeled_features.size()[-1])
            pseudo_label = pseudo_label.reshape(-1)
            # unlabeled_features2 = unlabeled_features_s.permute(0, 2, 3, 1).reshape(-1, labeled_features.size()[-1])
            # pseudo_label2 = pseudo_label_s.reshape(-1)

            # get predicted features
            proj_feat_unlabeled = model.projection_head(unlabeled_features)
            pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)
            # proj_feat_unlabeled2 = model2.projection_head(unlabeled_features2)
            # pred_feat_unlabeled2 = model2.prediction_head(proj_feat_unlabeled2)

            # Apply contrastive learning loss
            loss_contr_unlabeled = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_feat_unlabeled, pseudo_label, num_classes, prototype_memory.memory)

            loss_seg_ce = ce_loss(outputs[:args.labeled_bs], true_labels[:].long())

            loss_seg_dice = dice_loss(y, true_labels.unsqueeze(1))

            loss_lds =adv_loss(model, volume_batch)
            # loss_contr_unlabeled2 = contrastive_losses.contrastive_class_to_class_learned_memory(model2,
            #                                                                                     pred_feat_unlabeled2,
            #                                                                                     pseudo_label2,
            #                                                                                     num_classes,
            #                                                                                     prototype_memory2.memory)
            #
            # loss_seg_ce2 = ce_loss(outputs_s[:args.labeled_bs], true_labels_s[:].long())
            #
            # loss_seg_dice2 = dice_loss(y_s, true_labels_s.unsqueeze(1))
            #
            # loss_lds2 = adv_loss(model2, volume_batch_strong)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            loss1 = loss_seg_dice + consistency_weight * (loss_lds + 0.1 * (loss_contr_labeled + loss_contr_unlabeled))
            # loss2 = loss_seg_dice2 + consistency_weight * (loss_lds2 + 0.1 * (loss_contr_labeled2 + loss_contr_unlabeled2))
            # loss = loss1+loss2
            optimizer.zero_grad()
            # optimizer2.zero_grad()
            loss1.backward()
            optimizer.step()
            # optimizer2.step()

            iter_num = iter_num + 1
            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_seg_ce, iter_num)
            # writer.add_scalar('info/loss_dice', loss_seg_dice, iter_num)
            # writer.add_scalar('info/loss_vat', loss_lds, iter_num)
            # writer.add_scalar('info/loss_cl_l', loss_contr_labeled, iter_num)
            # writer.add_scalar('info/loss_cl_u', loss_contr_unlabeled, iter_num)
            # writer.add_scalar('info/consistency_weight',    consistency_weight, iter_num)
            logging.info('iteration %d : loss : %f' %(iter_num, loss1))
            monitor_gpu_memory(iter_num, model, volume_batch)
            # if iter_num % 20 == 0:
            #     image = volume_batch[1, 0:1, :, :]
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                # writer.add_scalar('info/val_mean_dice', performance, iter_num)
                # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'model_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

                # model2.eval()
                # metric_list = 0.0
                # for _, sampled_batch in enumerate(valloader):
                #     metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model2,
                #                                          classes=num_classes)
                #     metric_list += np.array(metric_i)
                # metric_list = metric_list / len(db_val)
                # # for class_i in range(num_classes-1):
                # #     writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                # #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                #
                # performance = np.mean(metric_list, axis=0)[0]
                #
                # mean_hd95_2 = np.mean(metric_list, axis=0)[1]
                # # writer.add_scalar('info/val_mean_dice', performance, iter_num)
                # # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                #
                # if performance > best_performance2:
                #     best_performance2 = performance
                #     save_mode_path = os.path.join(snapshot_path,
                #                                   'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                #     save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                #     torch.save(model2.state_dict(), save_mode_path)
                #     torch.save(model2.state_dict(), save_best_path)
                #
                # logging.info('iteration %d : mean_dice2 : %f mean_hd952 : %f' % (iter_num, performance, mean_hd95_2))
                # model2.train()

            if iter_num >= max_iterations:
                break
        print('avg time per epoch:', sum(times))
        times.clear()
        if iter_num >= max_iterations:
            iterator.close()
            break
    # writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/Promise12_{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('./code/', snapshot_path + '/code',shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
