import argparse
import os
import shutil
import cv2
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib import pyplot as plt
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from dataloaders.dataset import MRSEG19Normalization
from networks1.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/lnn/zzy/DiffRect-main/datasets/Promise12', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--ckpt', type=str,
                    default='/home/lnn/zzy/DiffRect-main/logs/ACDC/diffrect_7_labeled/unet/unet_best_model.pth',
                    help='checkpoint_name', required=True)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        jc = metric.binary.jc(pred, gt)
        return dice, jc, hd95, asd
    else:
        return 0, 0, 50, 10.


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(os.path.join(FLAGS.root_path, "data", f"{case}.h5"), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    # 注册钩子用于Grad-CAM
    target_layer = net.decoder.out_conv  # 根据实际模型结构调整
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['layer'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['layer'] = grad_output[0].detach()

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # 创建叠加图像保存目录
    overlay_dir = os.path.join(test_save_path, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    with torch.enable_grad():
        for ind in range(image.shape[0]):
            slice_data = image[ind, :, :]
            orig_h, orig_w = slice_data.shape

            # 预处理图像
            slice_resized = zoom(slice_data, (256 / orig_h, 256 / orig_w), order=0)
            input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
            input_tensor.requires_grad = True

            # 前向传播
            net.eval()
            out_main = net(input_tensor)
            if len(out_main) > 1:
                out_main = out_main[0]

            # 获取预测结果
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out_cpu = out.cpu().detach().numpy()
            pred = zoom(out_cpu, (orig_h / 256, orig_w / 256), order=0)
            prediction[ind] = pred

            # Grad-CAM计算
            net.zero_grad()
            out_main[:, 1].mean().backward()  # 假设类别1为目标区域

            # 获取激活值和梯度
            acts = activations['layer'].squeeze(0).cpu().numpy()
            grads = gradients['layer'].squeeze(0).cpu().numpy()

            # 计算权重和CAM
            weights = np.mean(grads, axis=(1, 2))
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
            cam = np.maximum(cam, 0)  # ReLU激活
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            # 调整到原始尺寸
            cam = zoom(cam, (orig_h / cam.shape[0], orig_w / cam.shape[1]), order=1)

            # 生成叠加图像
            original = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            original = (original * 255).astype(np.uint8)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

            heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(original_rgb, 1 - 0.3, heatmap, 0.3, 0)

            # 保存叠加图像
            plt.imsave(
                os.path.join(overlay_dir, f"{case}_slice{ind:03d}.png"),
                cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            )

    # 移除钩子
    handle_forward.remove()
    handle_backward.remove()

    # 保存NIfTI图像
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

    for data, suffix in zip([img_itk, prd_itk, lab_itk], ["_img", "_pred", "_gt"]):
        data.SetSpacing((1, 1, 10))
        sitk.WriteImage(data, os.path.join(test_save_path, f"{case}{suffix}.nii.gz"))

    # 计算评估指标
    metrics = []
    for class_id in range(1, FLAGS.num_classes):
        if np.sum(prediction == class_id) == 0 or np.sum(label == class_id) == 0:
            metrics.append((0, 0, 50, 10))  # 空值时设为默认值
        else:
            metrics.append(calculate_metric_percase(prediction == class_id, label == class_id))

    return metrics  # 返回所有类别的指标


def Inference(FLAGS, list=None):
    with open(FLAGS.root_path + f'/{list}.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])

    test_save_path = FLAGS.ckpt.split('/unet/')[0] + '/predictions/'
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    # 加载模型
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    if 'state_dict' in torch.load(FLAGS.ckpt).keys():
        info = net.load_state_dict(torch.load(FLAGS.ckpt)['state_dict'])
    else:
        info = net.load_state_dict(torch.load(FLAGS.ckpt))
    print(f"init weight from {FLAGS.ckpt}")
    print(info)
    net.eval()

    # 初始化存储所有样本的指标
    all_class_metrics = [[] for _ in range(FLAGS.num_classes - 1)]  # 每个类别一个列表

    for case in tqdm(image_list):
        case_metrics = test_single_volume(case, net, test_save_path, FLAGS)
        for class_idx, metrics in enumerate(case_metrics):
            all_class_metrics[class_idx].append(metrics)

    # 计算均值和标准差
    avg_metrics = []
    std_metrics = []
    for class_metrics in all_class_metrics:
        if not class_metrics:
            avg_metrics.append(np.zeros(4))
            std_metrics.append(np.zeros(4))
            continue

        class_metrics = np.array(class_metrics)
        avg_metrics.append(np.mean(class_metrics, axis=0))
        std_metrics.append(np.std(class_metrics, axis=0))

    return avg_metrics, std_metrics


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    list_list = ['test'] if 'Promise12' in FLAGS.root_path else ['val']

    for list_name in list_list:
        avg_metric, std_metric = Inference(FLAGS, list_name)
        print(f"\n=== Evaluation Results for {list_name} Set ===")

        # 定义指标名称和类别名称
        metric_names = ["Dice", "Jaccard", "HD95", "ASD"]
        class_names = [f"Class {i + 1}" for i in range(FLAGS.num_classes - 1)]

        # 打印每类的指标（均值±标准差）
        print("\nResults per class:")
        for class_idx, (class_name, avg, std) in enumerate(zip(class_names, avg_metric, std_metric)):
            print(f"\n{class_name}:")
            for metric_idx, (metric_name, avg_val, std_val) in enumerate(zip(metric_names, avg, std)):
                print(f"  {metric_name}: {avg_val:.4f} ± {std_val:.4f}")

        # 计算所有类别的平均指标
        print("\nOverall Average:")
        overall_avg = np.mean(avg_metric, axis=0)
        overall_std = np.mean(std_metric, axis=0)
        for metric_idx, (metric_name, avg_val, std_val) in enumerate(zip(metric_names, overall_avg, overall_std)):
            print(f"  {metric_name}: {avg_val:.4f} ± {std_val:.4f}")