import argparse
import os
import shutil

import cv2
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib import pyplot as plt
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from networks1.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/lnn/zzy/SS-Net-main/data/Promise12', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SSNet', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


# def test_single_volume(case, net, test_save_path, FLAGS):
#     h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (256 / x, 256 / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out_main = net(input)
#             if len(out_main)>1:
#                 out_main=out_main[0]
#             out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
#             prediction[ind] = pred
#     if np.sum(prediction == 1)==0:
#         first_metric = 0,0,0,0
#     else:
#         first_metric = calculate_metric_percase(prediction == 1, label == 1)
#
#     # if np.sum(prediction == 2)==0:
#     #     second_metric = 0,0,0,0
#     # else:
#     #     second_metric = calculate_metric_percase(prediction == 2, label == 2)
#     #
#     # if np.sum(prediction == 3)==0:
#     #     third_metric = 0,0,0,0
#     # else:
#     #     third_metric = calculate_metric_percase(prediction == 3, label == 3)
#
#     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     img_itk.SetSpacing((1, 1, 10))
#     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     prd_itk.SetSpacing((1, 1, 10))
#     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     lab_itk.SetSpacing((1, 1, 10))
#     sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric
def test_single_volume(case, net, test_save_path, FLAGS):
    # 加载数据
    h5f = h5py.File(os.path.join(FLAGS.root_path, "data", f"{case}.h5"), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    # 注册钩子
    target_layer = net.decoder.out_conv  # 根据实际模型结构调整
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['layer'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['layer'] = grad_output[0].detach()

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # 创建输出目录
    overlay_dir = os.path.join(test_save_path, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    with torch.enable_grad():
        for ind in range(image.shape[0]):
            # 处理每个切片
            slice_data = image[ind, :, :]
            orig_h, orig_w = slice_data.shape

            # 预处理
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

            # --- Grad-CAM 计算 ---
            net.zero_grad()
            out_main[:, 1].mean().backward()  # 假设类别1为目标

            # 获取激活和梯度
            acts = activations['layer'].squeeze(0).cpu().numpy()
            grads = gradients['layer'].squeeze(0).cpu().numpy()

            # 计算权重和CAM
            weights = np.mean(grads, axis=(1, 2))
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
            cam = np.maximum(cam, 0)  # ReLU激活
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            # 调整到原始尺寸
            cam = zoom(cam, (orig_h / cam.shape[0], orig_w / cam.shape[1]), order=1)

            # --- 生成叠加图像 ---
            # 归一化原始图像
            original = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            original = (original * 255).astype(np.uint8)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

            # 生成热力图（JET颜色映射）
            heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # heatmap = cv2.flip(heatmap,flipCode=1)

            # 叠加图像（调整透明度）
            alpha = 0.3  # 热力图透明度
            superimposed = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)

            # 保存叠加图像
            plt.imsave(
                os.path.join(overlay_dir, f"{case}_slice{ind:03d}.png"),
                cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            )

    # 移除钩子
    handle_forward.remove()
    handle_backward.remove()

    # 保存原始预测结果
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

    for data, suffix in zip([img_itk, prd_itk, lab_itk], ["_img", "_pred", "_gt"]):
        data.SetSpacing((1, 1, 10))
        sitk.WriteImage(data, os.path.join(test_save_path, f"{case}{suffix}.nii.gz"))

    # 计算指标
    if np.sum(prediction == 1) == 0:
        return (0, 0, 0, 0)
    else:
        return calculate_metric_percase(prediction == 1, label == 1)

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/Promise12_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    test_save_path = "./model/Promise12_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    # second_total = 0.0
    # third_total = 0.0
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        # second_total += np.asarray(second_metric)
        # third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list)]
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))

