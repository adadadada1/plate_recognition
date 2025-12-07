import os
import cv2
import Levenshtein
from read_plate import ReadPlate

# ----------------------------
# 读取 ground truth 标签
# ----------------------------
def load_gt(gt_file):
    gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:5])
            text = parts[5]
            gt[img_name.lower()] = [text, [x1, y1, x2, y2]]
    return gt


# ----------------------------
# IoU 计算
# ----------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    if union <= 0:
        return 0
    return inter / union


# ----------------------------
# 执行评估
# ----------------------------
def evaluate(test_folder, gt_file):
    gt_data = load_gt(gt_file)
    reader = ReadPlate()

    TP = FP = FN = 0
    correct_plate = 0
    total_plate = len(gt_data)

    total_chars = 0
    correct_chars = 0
    edit_distance_sum = 0

    for img_name in os.listdir(test_folder):
        if not img_name.lower().endswith((".bmp", ".jpg", ".png")):
            continue

        img_path = os.path.join(test_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取：{img_path}")
            continue

        img_key = img_name.lower()
        if img_key not in gt_data:
            print(f"标签缺失：{img_name}")
            continue

        gt_text, gt_box = gt_data[img_key]

        # ----------------------------
        # 模型预测
        # ----------------------------
        preds = reader(image)   # [[plate_string, box], ...]

        if len(preds) == 0:
            FN += 1
            continue

        # 默认取第一个（你的模型只输出一个车牌）
        pred_text, pred_box = preds[0]

        # ----------------------------
        # 评估 Detection（检测）
        # ----------------------------
        IoU = iou(pred_box, gt_box)
        if IoU >= 0.4:
            TP += 1
        else:
            FN += 1
            continue

        # ----------------------------
        # 评估 OCR（识别）
        # ----------------------------
        # 1) 整牌正确
        if pred_text == gt_text:
            correct_plate += 1

        # 2) 字符级准确率
        l = max(len(pred_text), len(gt_text))
        total_chars += l
        correct_chars += sum(a == b for a, b in zip(pred_text, gt_text))

        # 3) 编辑距离（越小越好）
        edit_distance_sum += Levenshtein.distance(pred_text, gt_text)

    # ----------------------------
    # 计算指标
    # ----------------------------
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Plate Accuracy": correct_plate / total_plate,
        "Character Accuracy": correct_chars / total_chars,
        "Average Edit Distance": edit_distance_sum / total_plate
    }


# ----------------------------
# 主函数
# ----------------------------
if __name__ == "__main__":
    test_folder = "test/"      # 测试集图片
    gt_file = "test_label.txt"         # 标签 txt 文件

    metrics = evaluate(test_folder, gt_file)
    print("\n===== 评估结果 =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
