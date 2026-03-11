```python
# =========================================================
# 学習なしで final model を使って hold-out test 135枚だけ再評価
# 実行時間も計測
# =========================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import train_test_split


# =========================================================
# 0. 設定
# =========================================================
BASE_DIR = Path('/content/drive/MyDrive/dataset')  # 必要ならここだけ変更
RESULT_DIR = Path('/content/drive/MyDrive/defect_cv_with_final_test')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0
RANDOM_STATE = 42
FINAL_TEST_SIZE = 0.10
ROTATION_DEGREES = 30

BETA = 2.0
TARGET_RECALL = 0.90

FINAL_MODEL_NAME = 'final_resnet18_512_with_holdout_test.pth'
FINAL_TEST_RESULT_NAME = 'final_test_result.json'
AGGREGATE_RESULT_NAME = 'aggregate_summary.json'


# =========================================================
# 1. utility
# =========================================================
def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path: Path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_image_files(folder: Path):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
        files.extend(folder.glob(ext.upper()))
    return sorted(files)


# =========================================================
# 2. データ一覧取得 + hold-out split 再現
# =========================================================
good_dir = BASE_DIR / 'good'
bad_dir = BASE_DIR / 'bad'

good_files = get_image_files(good_dir)
bad_files = get_image_files(bad_dir)

all_paths = np.array([str(p) for p in bad_files + good_files])
all_labels = np.array([0] * len(bad_files) + [1] * len(good_files))  # bad=0, good=1

print("Total bad :", len(bad_files))
print("Total good:", len(good_files))
print("Total all :", len(all_paths))

trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=FINAL_TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=all_labels
)

print("\n===== Hold-out split =====")
print("trainval size:", len(trainval_paths))
print("final test size:", len(test_paths))
print("trainval bad/good:", int(np.sum(trainval_labels == 0)), int(np.sum(trainval_labels == 1)))
print("final test bad/good:", int(np.sum(test_labels == 0)), int(np.sum(test_labels == 1)))


# =========================================================
# 3. Dataset
# =========================================================
class DefectDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = int(self.labels[idx])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label, img_path


# =========================================================
# 4. Transform
# =========================================================
valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 5. model / metrics
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\nDevice:", device)

criterion = nn.CrossEntropyLoss()

def create_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    return model

def collect_probs(model, loader, criterion, device):
    model.eval()

    all_labels = []
    all_bad_probs = []
    all_paths = []
    all_images_cpu = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            bad_probs = probs[:, 0]  # bad=0

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_bad_probs.extend(bad_probs.cpu().numpy().tolist())
            all_paths.extend(list(paths))
            all_images_cpu.extend(images.cpu())

    return {
        'loss': total_loss / max(total_count, 1),
        'labels': np.array(all_labels, dtype=np.int64),
        'bad_probs': np.array(all_bad_probs, dtype=np.float32),
        'paths': all_paths,
        'images_cpu': all_images_cpu
    }

def metrics_from_probs(labels, bad_probs, threshold=0.5, beta=2.0):
    labels = np.asarray(labels).astype(np.int64)          # bad=0, good=1
    bad_probs = np.asarray(bad_probs).astype(np.float32)

    preds = np.where(bad_probs >= threshold, 0, 1)

    tp = int(np.sum((preds == 0) & (labels == 0)))  # badをbadと当てた
    tn = int(np.sum((preds == 1) & (labels == 1)))  # goodをgoodと当てた
    fp = int(np.sum((preds == 0) & (labels == 1)))  # goodをbadと誤判定
    fn = int(np.sum((preds == 1) & (labels == 0)))  # badをgoodと誤判定

    acc = (tp + tn) / max(len(labels), 1)

    bad_recall = tp / max(tp + fn, 1)
    bad_precision = tp / max(tp + fp, 1)

    beta2 = beta ** 2
    if bad_precision == 0 and bad_recall == 0:
        bad_fbeta = 0.0
    else:
        bad_fbeta = (1 + beta2) * bad_precision * bad_recall / max(beta2 * bad_precision + bad_recall, 1e-12)

    return {
        'acc': float(acc),
        'bad_recall': float(bad_recall),
        'bad_precision': float(bad_precision),
        'bad_fbeta': float(bad_fbeta),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'preds': preds
    }

def save_failure_images(images_cpu, labels, bad_probs, preds, paths, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    saved_count = 0
    for i, (img_tensor, label, pred, bad_prob, img_path) in enumerate(zip(images_cpu, labels, preds, bad_probs, paths)):
        if int(label) == int(pred):
            continue

        img = img_tensor.clone()
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        pil_img = to_pil_image(img)

        stem = Path(img_path).stem
        true_name = 'bad' if int(label) == 0 else 'good'
        pred_name = 'bad' if int(pred) == 0 else 'good'
        out_name = f"{i:03d}_true-{true_name}_pred-{pred_name}_badprob-{float(bad_prob):.4f}_{stem}.png"
        pil_img.save(save_dir / out_name)
        saved_count += 1

    print(f"Saved {saved_count} failure images to: {save_dir}")


# =========================================================
# 6. final test 用 loader
# =========================================================
test_dataset = DefectDataset(test_paths, test_labels, transform=valid_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# =========================================================
# 7. 保存済みモデル・しきい値を読む
# =========================================================
aggregate_summary_path = RESULT_DIR / AGGREGATE_RESULT_NAME
final_model_path = RESULT_DIR / FINAL_MODEL_NAME

if not final_model_path.exists():
    raise FileNotFoundError(f"Saved model not found: {final_model_path}")

# aggregate_summary.json があればそこから読む
# なければ前回ログの値 (FINAL_EPOCHS=10, FINAL_THRESHOLD=0.15) を使う
if aggregate_summary_path.exists():
    aggregate_summary = load_json(aggregate_summary_path)
    FINAL_EPOCHS = int(aggregate_summary['final_epochs'])
    FINAL_THRESHOLD = float(aggregate_summary['final_threshold'])
    print("\nLoaded aggregate summary:", aggregate_summary_path)
else:
    FINAL_EPOCHS = 10
    FINAL_THRESHOLD = 0.15
    print("\naggregate_summary.json not found.")
    print("Fallback values are used:")
    print("FINAL_EPOCHS   = 10")
    print("FINAL_THRESHOLD= 0.15")

print("final_model_path =", final_model_path)
print("FINAL_EPOCHS    =", FINAL_EPOCHS)
print("FINAL_THRESHOLD =", FINAL_THRESHOLD)


# =========================================================
# 8. hold-out final test 実行 + 時間計測
# =========================================================
final_model = create_model(device)
state_dict = torch.load(final_model_path, map_location=device)
final_model.load_state_dict(state_dict)
final_model.eval()

start_time = time.time()

test_out = collect_probs(final_model, test_loader, criterion, device)
test_metrics = metrics_from_probs(
    labels=test_out['labels'],
    bad_probs=test_out['bad_probs'],
    threshold=FINAL_THRESHOLD,
    beta=BETA
)

failure_dir = RESULT_DIR / 'final_test_failure_images'
save_failure_images(
    images_cpu=test_out['images_cpu'],
    labels=test_out['labels'],
    bad_probs=test_out['bad_probs'],
    preds=test_metrics['preds'],
    paths=test_out['paths'],
    save_dir=failure_dir
)

elapsed_sec = time.time() - start_time


# =========================================================
# 9. 結果保存
# =========================================================
result_obj = {
    'final_epochs': int(FINAL_EPOCHS),
    'final_threshold': float(FINAL_THRESHOLD),
    'acc': float(test_metrics['acc']),
    'bad_recall': float(test_metrics['bad_recall']),
    'bad_precision': float(test_metrics['bad_precision']),
    'bad_fbeta': float(test_metrics['bad_fbeta']),
    'tp': int(test_metrics['tp']),
    'tn': int(test_metrics['tn']),
    'fp': int(test_metrics['fp']),
    'fn': int(test_metrics['fn']),
    'beta': float(BETA),
    'target_recall': float(TARGET_RECALL),
    'model_path': str(final_model_path),
    'elapsed_sec': float(elapsed_sec),
    'num_test_samples': int(len(test_dataset))
}
save_json(RESULT_DIR / FINAL_TEST_RESULT_NAME, result_obj)


# =========================================================
# 10. 表示
# =========================================================
print("\n===== Final Test Result =====")
print(f"Final Test Samples  : {len(test_dataset)}")
print(f"Final Test Threshold: {FINAL_THRESHOLD:.2f}")
print(f"Final Test Acc      : {test_metrics['acc']:.4f}")
print(f"Final Test Recall   : {test_metrics['bad_recall']:.4f}")
print(f"Final Test Precision: {test_metrics['bad_precision']:.4f}")
print(f"Final Test F{BETA:.0f}       : {test_metrics['bad_fbeta']:.4f}")
print(f"TP={test_metrics['tp']} TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']}")
print(f"Elapsed time (sec)  : {elapsed_sec:.2f}")

print("\nSaved final test result to:", RESULT_DIR / FINAL_TEST_RESULT_NAME)
print("Saved failure images to  :", failure_dir)
print("Done.")
```

    Mounted at /content/drive
    Total bad : 350
    Total good: 1000
    Total all : 1350
    
    ===== Hold-out split =====
    trainval size: 1215
    final test size: 135
    trainval bad/good: 315 900
    final test bad/good: 35 100
    
    Device: cuda
    
    Loaded aggregate summary: /content/drive/MyDrive/defect_cv_with_final_test/aggregate_summary.json
    final_model_path = /content/drive/MyDrive/defect_cv_with_final_test/final_resnet18_512_with_holdout_test.pth
    FINAL_EPOCHS    = 10
    FINAL_THRESHOLD = 0.15
    Saved 1 failure images to: /content/drive/MyDrive/defect_cv_with_final_test/final_test_failure_images
    
    ===== Final Test Result =====
    Final Test Samples  : 135
    Final Test Threshold: 0.15
    Final Test Acc      : 0.9926
    Final Test Recall   : 0.9714
    Final Test Precision: 1.0000
    Final Test F2       : 0.9770
    TP=34 TN=100 FP=0 FN=1
    Elapsed time (sec)  : 77.17
    
    Saved final test result to: /content/drive/MyDrive/defect_cv_with_final_test/final_test_result.json
    Saved failure images to  : /content/drive/MyDrive/defect_cv_with_final_test/final_test_failure_images
    Done.

