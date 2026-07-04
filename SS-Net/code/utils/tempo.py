from random import random

import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import cripser as cr


def extract_patch_topology(patch_predictions, patch_images, threshold=0.7):
    """
    提取单个块的拓扑特征

    参数：
        patch_predictions: 块的预测输出 [c, p1, p2]
        patch_images: 块的原始图像 [p1, p2]
        threshold: 持久性阈值

    返回：
        topo_features: 拓扑特征字典
    """
    # 将预测转换为似然（假设是多分类softmax输出）
    if len(patch_predictions.shape) == 3:
        # 多分类情况，取最大类概率作为似然
        likelihood = torch.softmax(patch_predictions, dim=0).max(dim=0)[0]
    else:
        likelihood = patch_predictions

    # 转换为numpy并计算拓扑
    likelihood_np = likelihood.cpu().detach().numpy()
    lh = 1 - likelihood_np  # 转换为非似然

    # 计算持久同调
    pd = cr.computePH(lh, maxdim=1, location="birth")
    pd_arr = pd[pd[:, 0] == 0]  # 0维拓扑特征

    if pd_arr.shape[0] == 0:
        return {
            'has_topology': False,
            'critical_points': [],
            'persistence': [],
            'birth_points': [],
            'death_points': []
        }

    # 提取出生和死亡点坐标
    birth_points = pd_arr[:, 3:5]  # 出生临界点坐标
    death_points = pd_arr[:, 6:8]  # 死亡临界点坐标
    persistence = abs(pd_arr[:, 2] - pd_arr[:, 1])  # 持久性

    # 过滤低持久性的噪声
    valid_idx = persistence > threshold

    return {
        'has_topology': True,
        'critical_points': {
            'birth': birth_points[valid_idx],
            'death': death_points[valid_idx]
        },
        'persistence': persistence[valid_idx],
        'num_features': valid_idx.sum(),
        'mean_persistence': persistence[valid_idx].mean() if valid_idx.sum() > 0 else 0,
        'total_persistence': persistence[valid_idx].sum() if valid_idx.sum() > 0 else 0
    }


def compute_topology_similarity(topo1, topo2):
    """
    计算两个块之间的拓扑相似度

    返回：
        similarity: 相似度分数（越小越相似）
    """
    if not topo1['has_topology'] or not topo2['has_topology']:
        return float('inf') if topo1['has_topology'] != topo2['has_topology'] else 0

    # 基于持久性分布的相似度
    pers1 = topo1['persistence']
    pers2 = topo2['persistence']

    if len(pers1) == 0 or len(pers2) == 0:
        return abs(len(pers1) - len(pers2)) * 100

    # 计算持久性直方图相似度
    # 归一化持久性
    pers1_norm = pers1 / (pers1.sum() + 1e-8)
    pers2_norm = pers2 / (pers2.sum() + 1e-8)

    # 使用Wasserstein距离比较持久性分布
    # 简化版：使用均值差异
    mean_diff = abs(pers1.mean() - pers2.mean())
    count_diff = abs(len(pers1) - len(pers2)) / max(len(pers1), len(pers2))

    return mean_diff + count_diff


def find_topology_matching_patches(patches_topo, target_topo, top_k=4):
    """
    基于拓扑特征找到匹配的块

    参数：
        patches_topo: 所有块的拓扑特征列表
        target_topo: 目标拓扑特征
        top_k: 返回最相似的k个块索引

    返回：
        matched_indices: 匹配的块索引（按相似度排序）
        similarities: 对应的相似度分数
    """
    similarities = []
    for idx, patch_topo in enumerate(patches_topo):
        sim = compute_topology_similarity(patch_topo, target_topo)
        similarities.append(sim)

    similarities = torch.tensor(similarities)
    top_similarities, top_indices = similarities.topk(min(top_k, len(similarities)), largest=False)

    return top_indices.numpy(), top_similarities.numpy()


def ABD_R_Topological(outputs1_max, outputs2_max, volume_batch, volume_batch_strong,
                      outputs1_unlabel, outputs2_unlabel, args):
    """
    基于拓扑的ABD-R（双向位移块）- 改进版

    核心改进：
    1. 用拓扑特征替代简单的块均值
    2. 基于临界点匹配而非简单的值比较
    3. 考虑拓扑结构的相似性
    """
    B = args.labeled_bs
    unlabeled_bs = outputs1_max.shape[0] - B

    # 1. 准备未标记数据的块
    patches_1 = rearrange(outputs1_max[B:], 'b (h p1) (w p2)->b (h w) (p1 p2)',
                          p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[B:], 'b (h p1) (w p2)->b (h w) (p1 p2)',
                          p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch.squeeze(1)[B:], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[B:], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)

    # 2. 准备未标记数据的输出块（用于拓扑计算）
    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)

    # 3. 计算每个块的拓扑特征
    all_patches_topo_1 = []
    all_patches_topo_2 = []

    for i in range(unlabeled_bs):
        n_patches = patches_1.shape[1]
        batch_topo_1 = []
        batch_topo_2 = []

        for j in range(n_patches):
            # 提取块的预测和图像
            patch_pred_1 = patches_outputs_1[i, :, j, :].reshape(args.num_classes, args.patch_size, args.patch_size)
            patch_pred_2 = patches_outputs_2[i, :, j, :].reshape(args.num_classes, args.patch_size, args.patch_size)
            patch_img_1 = image_patch_1[i, j].reshape(args.patch_size, args.patch_size)
            patch_img_2 = image_patch_2[i, j].reshape(args.patch_size, args.patch_size)

            # 提取拓扑特征
            topo_1 = extract_patch_topology(patch_pred_1, patch_img_1, args.topo_threshold)
            topo_2 = extract_patch_topology(patch_pred_2, patch_img_2, args.topo_threshold)

            batch_topo_1.append(topo_1)
            batch_topo_2.append(topo_2)

        all_patches_topo_1.append(batch_topo_1)
        all_patches_topo_2.append(batch_topo_2)

    # 4. 基于拓扑特征进行双向匹配
    for i in range(unlabeled_bs):
        # 计算每个块的拓扑重要性分数
        topo_importance_1 = [t['total_persistence'] if t['has_topology'] else 0
                             for t in all_patches_topo_1[i]]
        topo_importance_2 = [t['total_persistence'] if t['has_topology'] else 0
                             for t in all_patches_topo_2[i]]

        # 找到最重要和最不重要的块（基于拓扑复杂度）
        if random.random() < 0.5:
            # 策略1：基于拓扑特征的智能匹配
            # 找到拓扑最丰富的块（源）
            src_idx_1 = np.argmax(topo_importance_1)
            src_idx_2 = np.argmax(topo_importance_2)

            # 找到拓扑最贫乏的块（目标）
            tgt_idx_1 = np.argmin(topo_importance_1)
            tgt_idx_2 = np.argmin(topo_importance_2)

            # 使用拓扑匹配确保交换的合理性
            # 找到与源块拓扑相似的目标块
            matched_indices_1, _ = find_topology_matching_patches(
                all_patches_topo_2[i], all_patches_topo_1[i][src_idx_1], top_k=1
            )
            matched_indices_2, _ = find_topology_matching_patches(
                all_patches_topo_1[i], all_patches_topo_2[i][src_idx_2], top_k=1
            )

            if len(matched_indices_1) > 0:
                tgt_idx_2 = matched_indices_1[0]
            if len(matched_indices_2) > 0:
                tgt_idx_1 = matched_indices_2[0]

        else:
            # 策略2：基于拓扑差异的对比增强
            # 找到拓扑差异最大的块对
            topo_scores_1 = torch.tensor(topo_importance_1)
            topo_scores_2 = torch.tensor(topo_importance_2)

            # 最高拓扑复杂度 vs 最低拓扑复杂度
            src_idx_1 = topo_scores_1.argmax().item()
            tgt_idx_1 = topo_scores_1.argmin().item()
            src_idx_2 = topo_scores_2.argmax().item()
            tgt_idx_2 = topo_scores_2.argmin().item()

        # 执行块交换
        # 从view2交换到view1
        image_patch_1[i][tgt_idx_1] = image_patch_2[i][src_idx_2]
        # 从view1交换到view2
        image_patch_2[i][tgt_idx_2] = image_patch_1[i][src_idx_1]

    # 5. 重组图像
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b (h p1) (w p2)',
                                 h=args.h_size, w=args.w_size,
                                 p1=args.patch_size, p2=args.patch_size)

    return image_patch_last


def ABD_R_BCP_Topological(out_max_1, out_max_2, net_input_1, net_input_2,
                          out_1, out_2, args):
    """
    基于拓扑的ABD-R BCP版本
    """
    B = out_max_1.shape[0]

    # 准备块数据
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)',
                          p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)',
                          p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(net_input_2.squeeze(1), 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)

    # 准备输出块
    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)

    # 计算拓扑特征
    all_patches_topo_1 = []
    all_patches_topo_2 = []

    for i in range(B):
        n_patches = patches_1.shape[1]
        batch_topo_1 = []
        batch_topo_2 = []

        for j in range(n_patches):
            patch_pred_1 = patches_outputs_1[i, :, j, :].reshape(args.num_classes, args.patch_size, args.patch_size)
            patch_pred_2 = patches_outputs_2[i, :, j, :].reshape(args.num_classes, args.patch_size, args.patch_size)
            patch_img_1 = image_patch_1[i, j].reshape(args.patch_size, args.patch_size)
            patch_img_2 = image_patch_2[i, j].reshape(args.patch_size, args.patch_size)

            topo_1 = extract_patch_topology(patch_pred_1, patch_img_1, args.topo_threshold)
            topo_2 = extract_patch_topology(patch_pred_2, patch_img_2, args.topo_threshold)

            batch_topo_1.append(topo_1)
            batch_topo_2.append(topo_2)

        all_patches_topo_1.append(batch_topo_1)
        all_patches_topo_2.append(batch_topo_2)

    # 执行拓扑引导的块交换
    for i in range(B):
        topo_importance_1 = [t['total_persistence'] if t['has_topology'] else 0
                             for t in all_patches_topo_1[i]]
        topo_importance_2 = [t['total_persistence'] if t['has_topology'] else 0
                             for t in all_patches_topo_2[i]]

        if random.random() < 0.5:
            # 拓扑引导的智能匹配
            # 找到拓扑复杂度最高的块
            max_idx_1 = np.argmax(topo_importance_1)
            max_idx_2 = np.argmax(topo_importance_2)

            # 找到拓扑复杂度最低的块
            min_idx_1 = np.argmin(topo_importance_1)
            min_idx_2 = np.argmin(topo_importance_2)

            # 基于拓扑相似度找到最佳匹配
            matched_for_min_1, _ = find_topology_matching_patches(
                all_patches_topo_2[i], all_patches_topo_1[i][min_idx_1], top_k=1
            )
            matched_for_min_2, _ = find_topology_matching_patches(
                all_patches_topo_1[i], all_patches_topo_2[i][min_idx_2], top_k=1
            )

            if len(matched_for_min_1) > 0:
                max_idx_2 = matched_for_min_1[0]
            if len(matched_for_min_2) > 0:
                max_idx_1 = matched_for_min_2[0]

        else:
            # 基于拓扑对比增强
            topo_scores_1 = torch.tensor(topo_importance_1)
            topo_scores_2 = torch.tensor(topo_importance_2)

            max_idx_1 = topo_scores_1.argmax().item()
            min_idx_1 = topo_scores_1.argmin().item()
            max_idx_2 = topo_scores_2.argmax().item()
            min_idx_2 = topo_scores_2.argmin().item()

        # 执行交换
        image_patch_1[i][min_idx_1] = image_patch_2[i][max_idx_2]
        image_patch_2[i][min_idx_2] = image_patch_1[i][max_idx_1]

    # 重组图像
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b (h p1) (w p2)',
                                 h=args.h_size, w=args.w_size,
                                 p1=args.patch_size, p2=args.patch_size)

    return image_patch_last


def ABD_I_Topological(outputs1_max, outputs2_max, volume_batch, volume_batch_strong,
                      label_batch, label_batch_strong, args):
    """
    基于拓扑的ABD-I（监督数据的双向位移）
    """
    B = args.labeled_bs

    # 准备监督数据的块
    patches_sup_1 = rearrange(outputs1_max[:B], 'b (h p1) (w p2)->b (h w) (p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)
    patches_sup_2 = rearrange(outputs2_max[:B], 'b (h p1) (w p2)->b (h w) (p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)
    image_patch_sup_1 = rearrange(volume_batch.squeeze(1)[:B], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)
    image_patch_sup_2 = rearrange(volume_batch_strong.squeeze(1)[:B], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)
    label_patch_sup_1 = rearrange(label_batch[:B], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)
    label_patch_sup_2 = rearrange(label_batch_strong[:B], 'b (h p1) (w p2) -> b (h w)(p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)

    # 计算拓扑特征（监督数据）
    all_patches_topo_1 = []
    all_patches_topo_2 = []

    for i in range(B):
        n_patches = patches_sup_1.shape[1]
        batch_topo_1 = []
        batch_topo_2 = []

        for j in range(n_patches):
            # 对于监督数据，使用原始图像而不是预测
            patch_img_1 = image_patch_sup_1[i, j].reshape(args.patch_size, args.patch_size)
            patch_img_2 = image_patch_sup_2[i, j].reshape(args.patch_size, args.patch_size)

            # 对图像本身进行拓扑分析
            topo_1 = extract_patch_topology(patch_img_1, patch_img_1, args.topo_threshold)
            topo_2 = extract_patch_topology(patch_img_2, patch_img_2, args.topo_threshold)

            batch_topo_1.append(topo_1)
            batch_topo_2.append(topo_2)

        all_patches_topo_1.append(batch_topo_1)
        all_patches_topo_2.append(batch_topo_2)

    # 执行拓扑引导的块交换
    for i in range(B):
        topo_importance_1 = [t['total_persistence'] if t['has_topology'] else 0
                             for t in all_patches_topo_1[i]]
        topo_importance_2 = [t['total_persistence'] if t['has_topology'] else 0
                             for t in all_patches_topo_2[i]]

        if random.random() < 0.5:
            # 找到拓扑特征最丰富的块
            max_idx_1 = np.argmax(topo_importance_1)
            max_idx_2 = np.argmax(topo_importance_2)

            # 找到拓扑特征最贫乏的块
            min_idx_1 = np.argmin(topo_importance_1)
            min_idx_2 = np.argmin(topo_importance_2)

            # 基于拓扑相似度匹配
            matched_for_min_1, _ = find_topology_matching_patches(
                all_patches_topo_2[i], all_patches_topo_1[i][min_idx_1], top_k=1
            )
            matched_for_min_2, _ = find_topology_matching_patches(
                all_patches_topo_1[i], all_patches_topo_2[i][min_idx_2], top_k=1
            )

            if len(matched_for_min_1) > 0:
                max_idx_2 = matched_for_min_1[0]
            if len(matched_for_min_2) > 0:
                max_idx_1 = matched_for_min_2[0]

            # 执行交换（图像和标签）
            image_patch_sup_1[i][min_idx_1] = image_patch_sup_2[i][max_idx_2]
            image_patch_sup_2[i][min_idx_2] = image_patch_sup_1[i][max_idx_1]

            label_patch_sup_1[i][min_idx_1] = label_patch_sup_2[i][max_idx_2]
            label_patch_sup_2[i][min_idx_2] = label_patch_sup_1[i][max_idx_1]

    # 重组图像和标签
    image_patch_sup = torch.cat([image_patch_sup_1, image_patch_sup_2], dim=0)
    image_patch_sup_last = rearrange(image_patch_sup, 'b (h w)(p1 p2) -> b (h p1) (w p2)',
                                     h=args.h_size, w=args.w_size,
                                     p1=args.patch_size, p2=args.patch_size)

    label_patch_sup = torch.cat([label_patch_sup_1, label_patch_sup_2], dim=0)
    label_patch_sup_last = rearrange(label_patch_sup, 'b (h w)(p1 p2) -> b (h p1) (w p2)',
                                     h=args.h_size, w=args.w_size,
                                     p1=args.patch_size, p2=args.patch_size)

    return image_patch_sup_last, label_patch_sup_last