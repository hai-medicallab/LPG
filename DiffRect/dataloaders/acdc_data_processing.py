import glob
import os
import h5py
import numpy as np

def split_h5_slices(input_dir, output_dir):
    """
    将输入的H5文件中的3D图像和标签按切片保存为独立的H5文件
    切片索引从1开始，命名规则：
      - 1~9 显示为 1,2,...,9
      - 10及以上显示为 10,11,...
    :param input_dir: 输入H5文件路径（如 "/path/to/*.h5"）
    :param output_dir: 输出切片保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_files = glob.glob(input_dir)
    total_slices = 0

    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            image = f['image'][:]  # [slice_num, H, W]
            label = f['label'][:]

        if image.shape != label.shape:
            print(f"Error: {os.path.basename(file_path)} 图像标签形状不匹配")
            continue

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # 从1开始遍历切片索引
        for slice_idx in range(1, image.shape[0] + 1):  # 关键修改点
            output_path = os.path.join(
                output_dir,
                f"{base_name}_slice_{slice_idx}.h5"  # 直接使用自然数，不补零
            )

            with h5py.File(output_path, 'w') as f:
                # 注意：image和label的索引需要调整为 slice_idx-1
                f.create_dataset('image', data=image[slice_idx-1], compression="gzip")
                f.create_dataset('label', data=label[slice_idx-1], compression="gzip")

            total_slices += 1

    print(f"处理完成！共生成 {total_slices} 个切片文件。")

# 使用示例
split_h5_slices(
    input_dir="/home/lnn/zzy/DiffRect-main/datasets/ACDC/data/*.h5",
    output_dir="/home/lnn/zzy/DiffRect-main/datasets/ACDC/data/slices"
)
