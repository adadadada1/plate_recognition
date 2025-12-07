import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2
import matplotlib.pyplot as plt


# 数据增强方法
def random_affine_transform(image):
    rows, cols = image.shape[:2]

    # 随机生成仿射变换的参数
    random_scale_x = random.uniform(0.8, 1.2)
    random_scale_y = random.uniform(0.8, 1.2)
    random_rotation = random.uniform(-30, 30)
    random_translation_x = random.randint(-30, 30)
    random_translation_y = random.randint(-30, 30)

    # 构建仿射变换矩阵
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), random_rotation, random_scale_x)
    matrix[:, 2] += [random_translation_x, random_translation_y]

    # 执行仿射变换
    transformed_image = cv2.warpAffine(image, matrix, (cols, rows))

    return transformed_image


def random_perspective_transform(image):
    height, width, _ = image.shape

    # 随机生成透视变换的四个点
    perspective_points = np.array([
        [random.randint(0, width // 4), random.randint(0, height // 4)],
        [random.randint(3 * width // 4, width), random.randint(0, height // 4)],
        [random.randint(0, width // 4), random.randint(3 * height // 4, height)],
        [random.randint(3 * width // 4, width), random.randint(3 * height // 4, height)]
    ], dtype=np.float32)

    # 随机生成透视变换后的四个目标点
    target_points = np.array([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(perspective_points, target_points)

    # 执行透视变换
    transformed_image = cv2.warpPerspective(image, perspective_matrix, (width, height))

    return transformed_image


def process_image(image_path, output_folder):
    original_image = Image.open(image_path)

    # 将图像转为NumPy数组用于一些增强操作
    original_image_array = np.array(original_image)

    # 创建图像列表，用于保存每种增强后的图像
    augmented_images = []

    # 1. 原图
    augmented_images.append(original_image)

    # 2. 旋转
    rotated_image = original_image.rotate(45)
    augmented_images.append(rotated_image)

    # 3. 水平翻转
    flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images.append(flipped_image)

    # 4. 亮度调整
    enhancer = ImageEnhance.Brightness(original_image)
    brightened_image = enhancer.enhance(1.5)
    augmented_images.append(brightened_image)

    # 5. 裁剪
    cropped_image = original_image.crop((100, 100, 400, 400))
    augmented_images.append(cropped_image)

    # 6. 色彩增强
    color_enhancer = ImageEnhance.Color(original_image)
    enhanced_color_image = color_enhancer.enhance(2.0)
    augmented_images.append(enhanced_color_image)

    # 7. 添加噪声
    noise = np.random.normal(0, 15, original_image_array.shape).astype(np.uint8)
    noisy_array = cv2.add(original_image_array, noise)
    noisy_image = Image.fromarray(noisy_array)
    augmented_images.append(noisy_image)

    # 8. 饱和度调整
    saturation_enhancer = ImageEnhance.Color(original_image)
    saturated_image = saturation_enhancer.enhance(1.5)
    augmented_images.append(saturated_image)

    # 9. 高斯模糊
    blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius=7))
    augmented_images.append(blurred_image)

    # 10. 锐化
    sharpness_enhancer = ImageEnhance.Sharpness(original_image)
    sharpened_image = sharpness_enhancer.enhance(2.0)
    augmented_images.append(sharpened_image)

    # 11. 马赛克
    def mosaic(image, block_size=10):
        image = image.copy()
        width, height = image.size

        # 确保在最后一个块的时候不会超出边界
        for i in range(0, width, block_size):
            for j in range(0, height, block_size):
                # 修正超出边界的问题
                block_width = min(block_size, width - i)
                block_height = min(block_size, height - j)

                box = (i, j, i + block_width, j + block_height)
                region = image.crop(box)
                average_color = region.resize((1, 1), Image.ANTIALIAS).getpixel((0, 0))

                # 填充马赛克块
                for x in range(i, i + block_width):
                    for y in range(j, j + block_height):
                        image.putpixel((x, y), average_color)

        return image

    mosaiced_image = mosaic(original_image)
    augmented_images.append(mosaiced_image)

    # 12. 仿射变换
    transformed_image = random_affine_transform(original_image_array)
    transformed_image = Image.fromarray(transformed_image)
    augmented_images.append(transformed_image)

    # 13. 透视变换
    transformed_image = random_perspective_transform(original_image_array)
    augmented_images.append(Image.fromarray(transformed_image))

    # 保存增强后的图像
    for i, aug_img in enumerate(augmented_images):
        # 生成文件名
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        augmented_image_path = os.path.join(output_folder, f"{name}_augmented_{i + 1}{ext}")

        # 保存图像
        aug_img.save(augmented_image_path)


# 读取文件夹中的所有图像并进行数据增强
def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        # 只处理图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing {filename}...")
            process_image(image_path, output_folder)


# 设置输入和输出文件夹路径
input_folder = "train_15P"  # 替换为你的输入图像文件夹路径
output_folder = "train_enhance"  # 替换为你希望保存增强图像的文件夹路径

# 处理文件夹中的所有图像
process_images_in_folder(input_folder, output_folder)

print("Data augmentation completed.")
