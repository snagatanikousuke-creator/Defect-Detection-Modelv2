```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
import os
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms.functional import to_pil_image

from sklearn.model_selection import train_test_split, StratifiedKFold
```


```python
# =========================================================
# 0. 設定
# =========================================================
BASE_DIR = Path('/content/drive/MyDrive/dataset')  # 必要ならここだけ変更
RESULT_DIR = Path('/content/drive/MyDrive/defect_cv_with_final_test')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0   # Colab安定性重視
NUM_EPOCHS = 20
NUM_FOLDS = 5
RANDOM_STATE = 42

FINAL_TEST_SIZE = 0.10
ROTATION_DEGREES = 30
LEARNING_RATE = 1e-4

BETA = 2.0
TARGET_RECALL = 0.90
THRESHOLDS = np.linspace(0.05, 0.95, 19)

FINAL_MODEL_NAME = 'final_resnet18_512_with_holdout_test.pth'
FINAL_TRAIN_HISTORY_NAME = 'final_train_history.csv'
FINAL_TEST_RESULT_NAME = 'final_test_result.json'
AGGREGATE_RESULT_NAME = 'aggregate_summary.json'
```


```python
# =========================================================
# 1. 画像一覧取得
# =========================================================
def get_image_files(folder: Path):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
        files.extend(folder.glob(ext.upper()))
    return sorted(files)

good_dir = BASE_DIR / 'good'
bad_dir = BASE_DIR / 'bad'

good_files = get_image_files(good_dir)
bad_files = get_image_files(bad_dir)

all_paths = np.array([str(p) for p in bad_files + good_files])
all_labels = np.array([0] * len(bad_files) + [1] * len(good_files))  # bad=0, good=1

print("Total bad :", len(bad_files))
print("Total good:", len(good_files))
print("Total all :", len(all_paths))

# =========================================================
# 2. 最初に final test を分離
# =========================================================
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

pd.DataFrame({
    'path': trainval_paths,
    'label': trainval_labels
}).to_csv(RESULT_DIR / 'trainval_split.csv', index=False)

pd.DataFrame({
    'path': test_paths,
    'label': test_labels
}).to_csv(RESULT_DIR / 'final_test_split.csv', index=False)
```

    Total bad : 350
    Total good: 1000
    Total all : 1350
    
    ===== Hold-out split =====
    trainval size: 1215
    final test size: 135
    trainval bad/good: 315 900
    final test bad/good: 35 100



```python
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
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label, img_path

# =========================================================
# 4. Transform
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=ROTATION_DEGREES),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# 5. モデル作成
# =========================================================
def create_model(device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model.to(device)
```


```python
# =========================================================
# 6. 指標計算
# =========================================================
def calculate_metrics_from_confusion(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    bad_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bad_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return acc, bad_recall, bad_precision

def fbeta_score(precision, recall, beta=2.0):
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom

def predict_with_threshold(bad_probs, threshold):
    return np.where(bad_probs >= threshold, 0, 1)

def confusion_from_preds(labels, preds):
    tp = int(np.sum((labels == 0) & (preds == 0)))
    tn = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 1) & (preds == 0)))
    fn = int(np.sum((labels == 0) & (preds == 1)))
    return tp, tn, fp, fn

def metrics_from_probs(labels, bad_probs, threshold, beta=2.0):
    preds = predict_with_threshold(bad_probs, threshold)
    tp, tn, fp, fn = confusion_from_preds(labels, preds)
    acc, bad_recall, bad_precision = calculate_metrics_from_confusion(tp, tn, fp, fn)
    bad_fbeta = fbeta_score(bad_precision, bad_recall, beta=beta)

    return {
        'threshold': float(threshold),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'acc': float(acc),
        'bad_recall': float(bad_recall),
        'bad_precision': float(bad_precision),
        'bad_fbeta': float(bad_fbeta),
        'preds': preds
    }

def find_best_threshold(labels, bad_probs, thresholds, beta=2.0, target_recall=None):
    results = []
    for t in thresholds:
        m = metrics_from_probs(labels, bad_probs, t, beta=beta)
        results.append(m)

    if target_recall is not None:
        candidates = [r for r in results if r['bad_recall'] >= target_recall]
        if len(candidates) > 0:
            best = max(
                candidates,
                key=lambda x: (x['bad_fbeta'], x['bad_recall'], x['bad_precision'], x['acc'])
            )
            best['selection_mode'] = f"target_recall>={target_recall}"
            return best, results

    best = max(
        results,
        key=lambda x: (x['bad_fbeta'], x['bad_recall'], x['bad_precision'], x['acc'])
    )
    best['selection_mode'] = "max_fbeta"
    return best, results
```


```python
# =========================================================
# 7. 学習 / 推論
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, beta=2.0):
    model.train()

    running_loss = 0.0
    total = 0
    tp = tn = fp = fn = 0

    for images, labels, _paths in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        tp += ((labels == 0) & (preds == 0)).sum().item()
        tn += ((labels == 1) & (preds == 1)).sum().item()
        fp += ((labels == 1) & (preds == 0)).sum().item()
        fn += ((labels == 0) & (preds == 1)).sum().item()

    epoch_loss = running_loss / total
    epoch_acc, epoch_bad_recall, epoch_bad_precision = calculate_metrics_from_confusion(tp, tn, fp, fn)
    epoch_bad_fbeta = fbeta_score(epoch_bad_precision, epoch_bad_recall, beta=beta)

    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'bad_recall': epoch_bad_recall,
        'bad_precision': epoch_bad_precision,
        'bad_fbeta': epoch_bad_fbeta,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def collect_probs(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_bad_probs = []
    all_paths = []
    all_images_cpu = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images_gpu = images.to(device, non_blocking=True)
            labels_gpu = labels.to(device, non_blocking=True)

            outputs = model(images_gpu)
            loss = criterion(outputs, labels_gpu)

            probs = F.softmax(outputs, dim=1)
            bad_probs = probs[:, 0].detach().cpu().numpy()

            running_loss += loss.item() * images_gpu.size(0)
            total += labels_gpu.size(0)

            all_labels.extend(labels.numpy())
            all_bad_probs.extend(bad_probs.tolist())
            all_paths.extend(paths)
            all_images_cpu.extend([img.cpu() for img in images])

    return {
        'loss': running_loss / total,
        'labels': np.array(all_labels),
        'bad_probs': np.array(all_bad_probs),
        'paths': all_paths,
        'images_cpu': all_images_cpu
    }
```


```python
# =========================================================
# 8. 失敗画像保存
# =========================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def denormalize_tensor(img_tensor):
    img = img_tensor.numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return torch.tensor(img)

def save_failure_images(images_cpu, labels, bad_probs, preds, paths, save_dir: Path):
    fn_dir = save_dir / 'FN_bad_to_good'
    fp_dir = save_dir / 'FP_good_to_bad'
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i, (img_tensor, label, prob, pred, path_str) in enumerate(zip(images_cpu, labels, bad_probs, preds, paths)):
        label = int(label)
        pred = int(pred)
        path_str = str(path_str)
        base_name = Path(path_str).stem

        denorm = denormalize_tensor(img_tensor)
        pil_img = to_pil_image(denorm)

        if label == 0 and pred == 1:
            save_path = fn_dir / f'{i:04d}_{base_name}_badprob_{prob:.4f}.png'
            pil_img.save(save_path)
            records.append({
                'type': 'FN',
                'original_path': path_str,
                'saved_path': str(save_path),
                'label': label,
                'pred': pred,
                'bad_prob': float(prob)
            })

        elif label == 1 and pred == 0:
            save_path = fp_dir / f'{i:04d}_{base_name}_badprob_{prob:.4f}.png'
            pil_img.save(save_path)
            records.append({
                'type': 'FP',
                'original_path': path_str,
                'saved_path': str(save_path),
                'label': label,
                'pred': pred,
                'bad_prob': float(prob)
            })

    if len(records) > 0:
        pd.DataFrame(records).to_csv(save_dir / 'failure_records.csv', index=False)
```


```python
# =========================================================
# 9. device / loss / utility
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

cv_dir = RESULT_DIR / 'cv'
cv_dir.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_splits = list(skf.split(trainval_paths, trainval_labels))

print(f"Prepared {len(fold_splits)} folds.")
```

    device: cuda
    Prepared 5 folds.



```python
# =========================================================
# 10. 1 fold 実行関数
# =========================================================
def run_fold(fold_number: int):
    assert 1 <= fold_number <= NUM_FOLDS, f"fold_number must be 1..{NUM_FOLDS}"

    train_idx, valid_idx = fold_splits[fold_number - 1]

    fold_dir = cv_dir / f'fold_{fold_number}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    summary_path = fold_dir / 'summary.json'
    history_path = fold_dir / 'history.csv'
    valid_pred_path = fold_dir / 'valid_predictions.csv'
    model_path = fold_dir / f'best_model_fold_{fold_number}.pth'

    fold_train_paths = trainval_paths[train_idx]
    fold_train_labels = trainval_labels[train_idx]
    fold_valid_paths = trainval_paths[valid_idx]
    fold_valid_labels = trainval_labels[valid_idx]

    print("\n" + "=" * 100)
    print(f"Fold {fold_number}/{NUM_FOLDS}")
    print("=" * 100)
    print("train size:", len(fold_train_paths))
    print("valid size:", len(fold_valid_paths))
    print("train bad/good:", int(np.sum(fold_train_labels == 0)), int(np.sum(fold_train_labels == 1)))
    print("valid bad/good:", int(np.sum(fold_valid_labels == 0)), int(np.sum(fold_valid_labels == 1)))

    if summary_path.exists() and history_path.exists() and valid_pred_path.exists() and model_path.exists():
        print(f"[Fold {fold_number}] already completed. Skip.")
        return

    train_dataset = DefectDataset(fold_train_paths, fold_train_labels, transform=train_transform)
    valid_dataset = DefectDataset(fold_valid_paths, fold_valid_labels, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = create_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_score = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_summary = None

    history = []

    for epoch in range(NUM_EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, beta=BETA)
        valid_out = collect_probs(model, valid_loader, criterion, device)

        best_thr_info, thr_table = find_best_threshold(
            labels=valid_out['labels'],
            bad_probs=valid_out['bad_probs'],
            thresholds=THRESHOLDS,
            beta=BETA,
            target_recall=TARGET_RECALL
        )

        valid_metrics = best_thr_info
        score = valid_metrics['bad_fbeta']

        is_better = False
        if score > best_score:
            is_better = True
        elif np.isclose(score, best_score):
            if best_epoch_summary is None or valid_metrics['bad_recall'] > best_epoch_summary['bad_recall']:
                is_better = True

        if is_better:
            best_score = score
            best_epoch = epoch + 1
            best_threshold = valid_metrics['threshold']
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch_summary = {
                'epoch': best_epoch,
                'threshold': float(best_threshold),
                'acc': float(valid_metrics['acc']),
                'bad_recall': float(valid_metrics['bad_recall']),
                'bad_precision': float(valid_metrics['bad_precision']),
                'bad_fbeta': float(valid_metrics['bad_fbeta']),
                'tp': int(valid_metrics['tp']),
                'tn': int(valid_metrics['tn']),
                'fp': int(valid_metrics['fp']),
                'fn': int(valid_metrics['fn']),
                'selection_mode': valid_metrics['selection_mode']
            }

        epoch_record = {
            'fold': fold_number,
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'train_bad_recall': train_metrics['bad_recall'],
            'train_bad_precision': train_metrics['bad_precision'],
            'train_bad_fbeta': train_metrics['bad_fbeta'],
            'valid_loss': float(valid_out['loss']),
            'valid_acc': float(valid_metrics['acc']),
            'valid_bad_recall': float(valid_metrics['bad_recall']),
            'valid_bad_precision': float(valid_metrics['bad_precision']),
            'valid_bad_fbeta': float(valid_metrics['bad_fbeta']),
            'valid_threshold': float(valid_metrics['threshold']),
            'tp': int(valid_metrics['tp']),
            'tn': int(valid_metrics['tn']),
            'fp': int(valid_metrics['fp']),
            'fn': int(valid_metrics['fn']),
            'selection_mode': valid_metrics['selection_mode']
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f} | Train Bad Recall: {train_metrics['bad_recall']:.4f} | Train Bad Precision: {train_metrics['bad_precision']:.4f} | Train F{BETA:.0f}: {train_metrics['bad_fbeta']:.4f}")
        print(f"  Valid Loss: {valid_out['loss']:.4f} | Valid Acc: {valid_metrics['acc']:.4f} | Valid Bad Recall: {valid_metrics['bad_recall']:.4f} | Valid Bad Precision: {valid_metrics['bad_precision']:.4f} | Valid F{BETA:.0f}: {valid_metrics['bad_fbeta']:.4f}")
        print(f"  Valid Threshold: {valid_metrics['threshold']:.2f} | Selection: {valid_metrics['selection_mode']}")
        print(f"  TP={valid_metrics['tp']} TN={valid_metrics['tn']} FP={valid_metrics['fp']} FN={valid_metrics['fn']}")
        print("-" * 100)

    model.load_state_dict(best_model_wts)

    final_valid_out = collect_probs(model, valid_loader, criterion, device)
    final_metrics = metrics_from_probs(
        labels=final_valid_out['labels'],
        bad_probs=final_valid_out['bad_probs'],
        threshold=best_threshold,
        beta=BETA
    )

    save_failure_images(
        images_cpu=final_valid_out['images_cpu'],
        labels=final_valid_out['labels'],
        bad_probs=final_valid_out['bad_probs'],
        preds=final_metrics['preds'],
        paths=final_valid_out['paths'],
        save_dir=fold_dir / 'failure_images'
    )

    valid_pred_df = pd.DataFrame({
        'path': final_valid_out['paths'],
        'label': final_valid_out['labels'],
        'bad_prob': final_valid_out['bad_probs'],
        'pred': final_metrics['preds']
    })
    valid_pred_df.to_csv(valid_pred_path, index=False)

    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        'fold': fold_number,
        'best_epoch': int(best_epoch),
        'best_threshold': float(best_threshold),
        'acc': float(final_metrics['acc']),
        'bad_recall': float(final_metrics['bad_recall']),
        'bad_precision': float(final_metrics['bad_precision']),
        'bad_fbeta': float(final_metrics['bad_fbeta']),
        'tp': int(final_metrics['tp']),
        'tn': int(final_metrics['tn']),
        'fp': int(final_metrics['fp']),
        'fn': int(final_metrics['fn']),
        'model_path': str(model_path)
    }
    save_json(summary_path, summary)

    print(f"[Fold {fold_number}] Best Epoch: {best_epoch}")
    print(f"[Fold {fold_number}] Best Threshold: {best_threshold:.2f}")
    print(f"[Fold {fold_number}] Acc={final_metrics['acc']:.4f} Recall={final_metrics['bad_recall']:.4f} Precision={final_metrics['bad_precision']:.4f} F{BETA:.0f}={final_metrics['bad_fbeta']:.4f}")
    print(f"[Fold {fold_number}] saved to: {fold_dir}")
```


```python
run_fold(1)
```

    
    ====================================================================================================
    Fold 1/5
    ====================================================================================================
    train size: 972
    valid size: 243
    train bad/good: 252 720
    valid bad/good: 63 180
    Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth


    100%|██████████| 44.7M/44.7M [00:00<00:00, 219MB/s]


    Epoch [1/20]
      Train Loss: 0.2971 | Train Acc: 0.8745 | Train Bad Recall: 0.6944 | Train Bad Precision: 0.7955 | Train F2: 0.7125
      Valid Loss: 0.1949 | Valid Acc: 0.9506 | Valid Bad Recall: 0.9048 | Valid Bad Precision: 0.9048 | Valid F2: 0.9048
      Valid Threshold: 0.45 | Selection: target_recall>=0.9
      TP=57 TN=174 FP=6 FN=6
    ----------------------------------------------------------------------------------------------------
    Epoch [2/20]
      Train Loss: 0.1865 | Train Acc: 0.9259 | Train Bad Recall: 0.8214 | Train Bad Precision: 0.8846 | Train F2: 0.8333
      Valid Loss: 0.0827 | Valid Acc: 0.9712 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9118 | Valid F2: 0.9687
      Valid Threshold: 0.40 | Selection: target_recall>=0.9
      TP=62 TN=174 FP=6 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [3/20]
      Train Loss: 0.1511 | Train Acc: 0.9352 | Train Bad Recall: 0.8651 | Train Bad Precision: 0.8826 | Train F2: 0.8685
      Valid Loss: 0.1103 | Valid Acc: 0.9506 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.8592 | Valid F2: 0.9443
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=170 FP=10 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [4/20]
      Train Loss: 0.1042 | Train Acc: 0.9640 | Train Bad Recall: 0.9246 | Train Bad Precision: 0.9357 | Train F2: 0.9268
      Valid Loss: 0.0711 | Valid Acc: 0.9588 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.8630 | Valid F2: 0.9692
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=170 FP=10 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [5/20]
      Train Loss: 0.1026 | Train Acc: 0.9630 | Train Bad Recall: 0.9246 | Train Bad Precision: 0.9320 | Train F2: 0.9261
      Valid Loss: 0.0336 | Valid Acc: 0.9835 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9403 | Valid F2: 0.9875
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=63 TN=176 FP=4 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [6/20]
      Train Loss: 0.0749 | Train Acc: 0.9794 | Train Bad Recall: 0.9405 | Train Bad Precision: 0.9793 | Train F2: 0.9480
      Valid Loss: 0.0307 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [7/20]
      Train Loss: 0.0925 | Train Acc: 0.9671 | Train Bad Recall: 0.9206 | Train Bad Precision: 0.9508 | Train F2: 0.9265
      Valid Loss: 0.0451 | Valid Acc: 0.9918 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9841 | Valid F2: 0.9841
      Valid Threshold: 0.65 | Selection: target_recall>=0.9
      TP=62 TN=179 FP=1 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [8/20]
      Train Loss: 0.0728 | Train Acc: 0.9681 | Train Bad Recall: 0.9286 | Train Bad Precision: 0.9474 | Train F2: 0.9323
      Valid Loss: 0.0789 | Valid Acc: 0.9835 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9538 | Valid F2: 0.9779
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=62 TN=177 FP=3 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [9/20]
      Train Loss: 0.0610 | Train Acc: 0.9774 | Train Bad Recall: 0.9405 | Train Bad Precision: 0.9713 | Train F2: 0.9465
      Valid Loss: 0.2144 | Valid Acc: 0.9712 | Valid Bad Recall: 0.8889 | Valid Bad Precision: 1.0000 | Valid F2: 0.9091
      Valid Threshold: 0.05 | Selection: max_fbeta
      TP=56 TN=180 FP=0 FN=7
    ----------------------------------------------------------------------------------------------------
    Epoch [10/20]
      Train Loss: 0.0547 | Train Acc: 0.9846 | Train Bad Recall: 0.9643 | Train Bad Precision: 0.9759 | Train F2: 0.9666
      Valid Loss: 0.0819 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9688 | Valid F2: 0.9810
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=62 TN=178 FP=2 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [11/20]
      Train Loss: 0.0493 | Train Acc: 0.9846 | Train Bad Recall: 0.9563 | Train Bad Precision: 0.9837 | Train F2: 0.9617
      Valid Loss: 0.0883 | Valid Acc: 0.9671 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9104 | Valid F2: 0.9561
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=174 FP=6 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [12/20]
      Train Loss: 0.0406 | Train Acc: 0.9918 | Train Bad Recall: 0.9722 | Train Bad Precision: 0.9959 | Train F2: 0.9769
      Valid Loss: 0.2135 | Valid Acc: 0.9630 | Valid Bad Recall: 0.8571 | Valid Bad Precision: 1.0000 | Valid F2: 0.8824
      Valid Threshold: 0.05 | Selection: max_fbeta
      TP=54 TN=180 FP=0 FN=9
    ----------------------------------------------------------------------------------------------------
    Epoch [13/20]
      Train Loss: 0.0531 | Train Acc: 0.9774 | Train Bad Recall: 0.9484 | Train Bad Precision: 0.9637 | Train F2: 0.9514
      Valid Loss: 0.0628 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9688 | Valid F2: 0.9810
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=62 TN=178 FP=2 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [14/20]
      Train Loss: 0.0774 | Train Acc: 0.9722 | Train Bad Recall: 0.9405 | Train Bad Precision: 0.9518 | Train F2: 0.9427
      Valid Loss: 0.1617 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9048 | Valid Bad Precision: 1.0000 | Valid F2: 0.9223
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=57 TN=180 FP=0 FN=6
    ----------------------------------------------------------------------------------------------------
    Epoch [15/20]
      Train Loss: 0.0426 | Train Acc: 0.9835 | Train Bad Recall: 0.9643 | Train Bad Precision: 0.9720 | Train F2: 0.9658
      Valid Loss: 0.0241 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.25 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [16/20]
      Train Loss: 0.0350 | Train Acc: 0.9897 | Train Bad Recall: 0.9802 | Train Bad Precision: 0.9802 | Train F2: 0.9802
      Valid Loss: 0.0483 | Valid Acc: 0.9712 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9118 | Valid F2: 0.9687
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=62 TN=174 FP=6 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [17/20]
      Train Loss: 0.0329 | Train Acc: 0.9856 | Train Bad Recall: 0.9722 | Train Bad Precision: 0.9722 | Train F2: 0.9722
      Valid Loss: 0.0278 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.40 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [18/20]
      Train Loss: 0.0664 | Train Acc: 0.9784 | Train Bad Recall: 0.9484 | Train Bad Precision: 0.9676 | Train F2: 0.9522
      Valid Loss: 0.0908 | Valid Acc: 0.9630 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.8857 | Valid F2: 0.9627
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=62 TN=172 FP=8 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [19/20]
      Train Loss: 0.0460 | Train Acc: 0.9794 | Train Bad Recall: 0.9524 | Train Bad Precision: 0.9677 | Train F2: 0.9554
      Valid Loss: 0.1535 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9206 | Valid Bad Precision: 0.9831 | Valid F2: 0.9325
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=58 TN=179 FP=1 FN=5
    ----------------------------------------------------------------------------------------------------
    Epoch [20/20]
      Train Loss: 0.0184 | Train Acc: 0.9949 | Train Bad Recall: 0.9881 | Train Bad Precision: 0.9920 | Train F2: 0.9889
      Valid Loss: 0.0464 | Valid Acc: 0.9918 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9841 | Valid F2: 0.9841
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=62 TN=179 FP=1 FN=1
    ----------------------------------------------------------------------------------------------------
    [Fold 1] Best Epoch: 6
    [Fold 1] Best Threshold: 0.15
    [Fold 1] Acc=0.9959 Recall=1.0000 Precision=0.9844 F2=0.9968
    [Fold 1] saved to: /content/drive/MyDrive/defect_cv_with_final_test/cv/fold_1



```python
run_fold(2)
```

    
    ====================================================================================================
    Fold 2/5
    ====================================================================================================
    train size: 972
    valid size: 243
    train bad/good: 252 720
    valid bad/good: 63 180
    Epoch [1/20]
      Train Loss: 0.3340 | Train Acc: 0.8508 | Train Bad Recall: 0.7143 | Train Bad Precision: 0.7115 | Train F2: 0.7137
      Valid Loss: 0.4664 | Valid Acc: 0.9465 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.8472 | Valid F2: 0.9414
      Valid Threshold: 0.90 | Selection: target_recall>=0.9
      TP=61 TN=169 FP=11 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [2/20]
      Train Loss: 0.1913 | Train Acc: 0.9259 | Train Bad Recall: 0.8373 | Train Bad Precision: 0.8719 | Train F2: 0.8440
      Valid Loss: 0.0552 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [3/20]
      Train Loss: 0.1509 | Train Acc: 0.9434 | Train Bad Recall: 0.8611 | Train Bad Precision: 0.9156 | Train F2: 0.8715
      Valid Loss: 0.1177 | Valid Acc: 0.9794 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9394 | Valid F2: 0.9748
      Valid Threshold: 0.75 | Selection: target_recall>=0.9
      TP=62 TN=176 FP=4 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [4/20]
      Train Loss: 0.1261 | Train Acc: 0.9527 | Train Bad Recall: 0.8929 | Train Bad Precision: 0.9221 | Train F2: 0.8986
      Valid Loss: 0.0328 | Valid Acc: 0.9753 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9130 | Valid F2: 0.9813
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=63 TN=174 FP=6 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [5/20]
      Train Loss: 0.1009 | Train Acc: 0.9650 | Train Bad Recall: 0.9246 | Train Bad Precision: 0.9395 | Train F2: 0.9275
      Valid Loss: 0.0395 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [6/20]
      Train Loss: 0.1152 | Train Acc: 0.9640 | Train Bad Recall: 0.9008 | Train Bad Precision: 0.9578 | Train F2: 0.9116
      Valid Loss: 0.0326 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [7/20]
      Train Loss: 0.0689 | Train Acc: 0.9815 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9680 | Train F2: 0.9618
      Valid Loss: 0.0691 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9839 | Valid F2: 0.9713
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=179 FP=1 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [8/20]
      Train Loss: 0.0531 | Train Acc: 0.9815 | Train Bad Recall: 0.9563 | Train Bad Precision: 0.9718 | Train F2: 0.9594
      Valid Loss: 0.0294 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [9/20]
      Train Loss: 0.0792 | Train Acc: 0.9743 | Train Bad Recall: 0.9325 | Train Bad Precision: 0.9671 | Train F2: 0.9392
      Valid Loss: 0.0203 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.25 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [10/20]
      Train Loss: 0.0738 | Train Acc: 0.9774 | Train Bad Recall: 0.9484 | Train Bad Precision: 0.9637 | Train F2: 0.9514
      Valid Loss: 0.0337 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.40 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [11/20]
      Train Loss: 0.0351 | Train Acc: 0.9897 | Train Bad Recall: 0.9722 | Train Bad Precision: 0.9879 | Train F2: 0.9753
      Valid Loss: 0.0430 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [12/20]
      Train Loss: 0.0564 | Train Acc: 0.9794 | Train Bad Recall: 0.9524 | Train Bad Precision: 0.9677 | Train F2: 0.9554
      Valid Loss: 0.0234 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.35 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [13/20]
      Train Loss: 0.0449 | Train Acc: 0.9825 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9719 | Train F2: 0.9626
      Valid Loss: 0.0665 | Valid Acc: 0.9712 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9118 | Valid F2: 0.9687
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=62 TN=174 FP=6 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [14/20]
      Train Loss: 0.0474 | Train Acc: 0.9794 | Train Bad Recall: 0.9524 | Train Bad Precision: 0.9677 | Train F2: 0.9554
      Valid Loss: 0.0385 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.70 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [15/20]
      Train Loss: 0.0393 | Train Acc: 0.9856 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9837 | Train F2: 0.9649
      Valid Loss: 0.1241 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9524 | Valid Bad Precision: 1.0000 | Valid F2: 0.9615
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=60 TN=180 FP=0 FN=3
    ----------------------------------------------------------------------------------------------------
    Epoch [16/20]
      Train Loss: 0.0453 | Train Acc: 0.9815 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9680 | Train F2: 0.9618
      Valid Loss: 0.0254 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.25 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [17/20]
      Train Loss: 0.0335 | Train Acc: 0.9877 | Train Bad Recall: 0.9683 | Train Bad Precision: 0.9839 | Train F2: 0.9713
      Valid Loss: 0.0313 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [18/20]
      Train Loss: 0.0305 | Train Acc: 0.9887 | Train Bad Recall: 0.9722 | Train Bad Precision: 0.9839 | Train F2: 0.9745
      Valid Loss: 0.0406 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.95 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [19/20]
      Train Loss: 0.0422 | Train Acc: 0.9907 | Train Bad Recall: 0.9643 | Train Bad Precision: 1.0000 | Train F2: 0.9712
      Valid Loss: 0.0255 | Valid Acc: 0.9918 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9692 | Valid F2: 0.9937
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=178 FP=2 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [20/20]
      Train Loss: 0.0272 | Train Acc: 0.9907 | Train Bad Recall: 0.9762 | Train Bad Precision: 0.9880 | Train F2: 0.9785
      Valid Loss: 0.0508 | Valid Acc: 0.9918 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9841 | Valid F2: 0.9841
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=62 TN=179 FP=1 FN=1
    ----------------------------------------------------------------------------------------------------
    [Fold 2] Best Epoch: 9
    [Fold 2] Best Threshold: 0.25
    [Fold 2] Acc=0.9959 Recall=1.0000 Precision=0.9844 F2=0.9968
    [Fold 2] saved to: /content/drive/MyDrive/defect_cv_with_final_test/cv/fold_2



```python
run_fold(3)
```

    
    ====================================================================================================
    Fold 3/5
    ====================================================================================================
    train size: 972
    valid size: 243
    train bad/good: 252 720
    valid bad/good: 63 180
    Epoch [1/20]
      Train Loss: 0.2996 | Train Acc: 0.8755 | Train Bad Recall: 0.6825 | Train Bad Precision: 0.8075 | Train F2: 0.7043
      Valid Loss: 0.1889 | Valid Acc: 0.9012 | Valid Bad Recall: 0.9048 | Valid Bad Precision: 0.7600 | Valid F2: 0.8716
      Valid Threshold: 0.20 | Selection: target_recall>=0.9
      TP=57 TN=162 FP=18 FN=6
    ----------------------------------------------------------------------------------------------------
    Epoch [2/20]
      Train Loss: 0.1752 | Train Acc: 0.9228 | Train Bad Recall: 0.8214 | Train Bad Precision: 0.8734 | Train F2: 0.8313
      Valid Loss: 0.1576 | Valid Acc: 0.9383 | Valid Bad Recall: 0.9206 | Valid Bad Precision: 0.8529 | Valid F2: 0.9062
      Valid Threshold: 0.20 | Selection: target_recall>=0.9
      TP=58 TN=170 FP=10 FN=5
    ----------------------------------------------------------------------------------------------------
    Epoch [3/20]
      Train Loss: 0.1184 | Train Acc: 0.9558 | Train Bad Recall: 0.9127 | Train Bad Precision: 0.9163 | Train F2: 0.9134
      Valid Loss: 0.0870 | Valid Acc: 0.9835 | Valid Bad Recall: 0.9524 | Valid Bad Precision: 0.9836 | Valid F2: 0.9585
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=60 TN=179 FP=1 FN=3
    ----------------------------------------------------------------------------------------------------
    Epoch [4/20]
      Train Loss: 0.1416 | Train Acc: 0.9465 | Train Bad Recall: 0.8968 | Train Bad Precision: 0.8968 | Train F2: 0.8968
      Valid Loss: 0.1173 | Valid Acc: 0.9465 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.8472 | Valid F2: 0.9414
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=61 TN=169 FP=11 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [5/20]
      Train Loss: 0.0794 | Train Acc: 0.9753 | Train Bad Recall: 0.9405 | Train Bad Precision: 0.9634 | Train F2: 0.9450
      Valid Loss: 0.1252 | Valid Acc: 0.9506 | Valid Bad Recall: 0.9524 | Valid Bad Precision: 0.8696 | Valid F2: 0.9346
      Valid Threshold: 0.50 | Selection: target_recall>=0.9
      TP=60 TN=171 FP=9 FN=3
    ----------------------------------------------------------------------------------------------------
    Epoch [6/20]
      Train Loss: 0.0962 | Train Acc: 0.9722 | Train Bad Recall: 0.9444 | Train Bad Precision: 0.9482 | Train F2: 0.9452
      Valid Loss: 0.1127 | Valid Acc: 0.9671 | Valid Bad Recall: 0.9524 | Valid Bad Precision: 0.9231 | Valid F2: 0.9464
      Valid Threshold: 0.20 | Selection: target_recall>=0.9
      TP=60 TN=175 FP=5 FN=3
    ----------------------------------------------------------------------------------------------------
    Epoch [7/20]
      Train Loss: 0.0915 | Train Acc: 0.9671 | Train Bad Recall: 0.9127 | Train Bad Precision: 0.9583 | Train F2: 0.9215
      Valid Loss: 0.0784 | Valid Acc: 0.9712 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9242 | Valid F2: 0.9591
      Valid Threshold: 0.25 | Selection: target_recall>=0.9
      TP=61 TN=175 FP=5 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [8/20]
      Train Loss: 0.0824 | Train Acc: 0.9671 | Train Bad Recall: 0.9246 | Train Bad Precision: 0.9472 | Train F2: 0.9290
      Valid Loss: 0.1037 | Valid Acc: 0.9712 | Valid Bad Recall: 0.9206 | Valid Bad Precision: 0.9667 | Valid F2: 0.9295
      Valid Threshold: 0.45 | Selection: target_recall>=0.9
      TP=58 TN=178 FP=2 FN=5
    ----------------------------------------------------------------------------------------------------
    Epoch [9/20]
      Train Loss: 0.0608 | Train Acc: 0.9805 | Train Bad Recall: 0.9444 | Train Bad Precision: 0.9794 | Train F2: 0.9512
      Valid Loss: 0.1399 | Valid Acc: 0.9918 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 1.0000 | Valid F2: 0.9744
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=180 FP=0 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [10/20]
      Train Loss: 0.0871 | Train Acc: 0.9691 | Train Bad Recall: 0.9246 | Train Bad Precision: 0.9549 | Train F2: 0.9305
      Valid Loss: 0.0834 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9385 | Valid F2: 0.9621
      Valid Threshold: 0.25 | Selection: target_recall>=0.9
      TP=61 TN=176 FP=4 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [11/20]
      Train Loss: 0.0522 | Train Acc: 0.9794 | Train Bad Recall: 0.9484 | Train Bad Precision: 0.9715 | Train F2: 0.9530
      Valid Loss: 0.0906 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9524 | Valid Bad Precision: 1.0000 | Valid F2: 0.9615
      Valid Threshold: 0.70 | Selection: target_recall>=0.9
      TP=60 TN=180 FP=0 FN=3
    ----------------------------------------------------------------------------------------------------
    Epoch [12/20]
      Train Loss: 0.0549 | Train Acc: 0.9846 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9798 | Train F2: 0.9641
      Valid Loss: 0.1119 | Valid Acc: 0.9794 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9531 | Valid F2: 0.9652
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=177 FP=3 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [13/20]
      Train Loss: 0.0326 | Train Acc: 0.9907 | Train Bad Recall: 0.9722 | Train Bad Precision: 0.9919 | Train F2: 0.9761
      Valid Loss: 0.0826 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9385 | Valid F2: 0.9621
      Valid Threshold: 0.25 | Selection: target_recall>=0.9
      TP=61 TN=176 FP=4 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [14/20]
      Train Loss: 0.0362 | Train Acc: 0.9877 | Train Bad Recall: 0.9722 | Train Bad Precision: 0.9800 | Train F2: 0.9738
      Valid Loss: 0.0900 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9524 | Valid Bad Precision: 0.9524 | Valid F2: 0.9524
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=60 TN=177 FP=3 FN=3
    ----------------------------------------------------------------------------------------------------
    Epoch [15/20]
      Train Loss: 0.0334 | Train Acc: 0.9877 | Train Bad Recall: 0.9683 | Train Bad Precision: 0.9839 | Train F2: 0.9713
      Valid Loss: 0.0791 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9839 | Valid F2: 0.9713
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=61 TN=179 FP=1 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [16/20]
      Train Loss: 0.0340 | Train Acc: 0.9887 | Train Bad Recall: 0.9683 | Train Bad Precision: 0.9879 | Train F2: 0.9721
      Valid Loss: 0.1048 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9839 | Valid F2: 0.9713
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=179 FP=1 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [17/20]
      Train Loss: 0.0232 | Train Acc: 0.9897 | Train Bad Recall: 0.9722 | Train Bad Precision: 0.9879 | Train F2: 0.9753
      Valid Loss: 0.0981 | Valid Acc: 0.9794 | Valid Bad Recall: 0.9365 | Valid Bad Precision: 0.9833 | Valid F2: 0.9455
      Valid Threshold: 0.55 | Selection: target_recall>=0.9
      TP=59 TN=179 FP=1 FN=4
    ----------------------------------------------------------------------------------------------------
    Epoch [18/20]
      Train Loss: 0.0284 | Train Acc: 0.9959 | Train Bad Recall: 0.9921 | Train Bad Precision: 0.9921 | Train F2: 0.9921
      Valid Loss: 0.0627 | Valid Acc: 0.9835 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9683 | Valid F2: 0.9683
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=61 TN=178 FP=2 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [19/20]
      Train Loss: 0.0389 | Train Acc: 0.9928 | Train Bad Recall: 0.9762 | Train Bad Precision: 0.9960 | Train F2: 0.9801
      Valid Loss: 0.0756 | Valid Acc: 0.9671 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9104 | Valid F2: 0.9561
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=174 FP=6 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [20/20]
      Train Loss: 0.0312 | Train Acc: 0.9877 | Train Bad Recall: 0.9643 | Train Bad Precision: 0.9878 | Train F2: 0.9689
      Valid Loss: 0.1061 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9365 | Valid Bad Precision: 0.9672 | Valid F2: 0.9425
      Valid Threshold: 0.40 | Selection: target_recall>=0.9
      TP=59 TN=178 FP=2 FN=4
    ----------------------------------------------------------------------------------------------------
    [Fold 3] Best Epoch: 9
    [Fold 3] Best Threshold: 0.05
    [Fold 3] Acc=0.9918 Recall=0.9683 Precision=1.0000 F2=0.9744
    [Fold 3] saved to: /content/drive/MyDrive/defect_cv_with_final_test/cv/fold_3



```python
run_fold(4)
```

    
    ====================================================================================================
    Fold 4/5
    ====================================================================================================
    train size: 972
    valid size: 243
    train bad/good: 252 720
    valid bad/good: 63 180
    Epoch [1/20]
      Train Loss: 0.3251 | Train Acc: 0.8508 | Train Bad Recall: 0.7063 | Train Bad Precision: 0.7149 | Train F2: 0.7080
      Valid Loss: 0.2136 | Valid Acc: 0.9383 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.8158 | Valid F2: 0.9451
      Valid Threshold: 0.65 | Selection: target_recall>=0.9
      TP=62 TN=166 FP=14 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [2/20]
      Train Loss: 0.1867 | Train Acc: 0.9259 | Train Bad Recall: 0.8135 | Train Bad Precision: 0.8913 | Train F2: 0.8279
      Valid Loss: 0.0632 | Valid Acc: 0.9712 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9242 | Valid F2: 0.9591
      Valid Threshold: 0.20 | Selection: target_recall>=0.9
      TP=61 TN=175 FP=5 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [3/20]
      Train Loss: 0.1599 | Train Acc: 0.9383 | Train Bad Recall: 0.8571 | Train Bad Precision: 0.9000 | Train F2: 0.8654
      Valid Loss: 0.1110 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9254 | Valid F2: 0.9718
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=62 TN=175 FP=5 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [4/20]
      Train Loss: 0.1468 | Train Acc: 0.9424 | Train Bad Recall: 0.8651 | Train Bad Precision: 0.9083 | Train F2: 0.8734
      Valid Loss: 0.1650 | Valid Acc: 0.9794 | Valid Bad Recall: 0.9365 | Valid Bad Precision: 0.9833 | Valid F2: 0.9455
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=59 TN=179 FP=1 FN=4
    ----------------------------------------------------------------------------------------------------
    Epoch [5/20]
      Train Loss: 0.1391 | Train Acc: 0.9527 | Train Bad Recall: 0.8849 | Train Bad Precision: 0.9292 | Train F2: 0.8934
      Valid Loss: 0.0530 | Valid Acc: 0.9835 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9403 | Valid F2: 0.9875
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=176 FP=4 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [6/20]
      Train Loss: 0.0887 | Train Acc: 0.9691 | Train Bad Recall: 0.9325 | Train Bad Precision: 0.9476 | Train F2: 0.9355
      Valid Loss: 0.0388 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.80 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [7/20]
      Train Loss: 0.1089 | Train Acc: 0.9671 | Train Bad Recall: 0.9167 | Train Bad Precision: 0.9545 | Train F2: 0.9240
      Valid Loss: 0.0327 | Valid Acc: 0.9877 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9545 | Valid F2: 0.9906
      Valid Threshold: 0.35 | Selection: target_recall>=0.9
      TP=63 TN=177 FP=3 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [8/20]
      Train Loss: 0.0787 | Train Acc: 0.9774 | Train Bad Recall: 0.9444 | Train Bad Precision: 0.9675 | Train F2: 0.9490
      Valid Loss: 0.0378 | Valid Acc: 0.9835 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9403 | Valid F2: 0.9875
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=63 TN=176 FP=4 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [9/20]
      Train Loss: 0.0669 | Train Acc: 0.9774 | Train Bad Recall: 0.9444 | Train Bad Precision: 0.9675 | Train F2: 0.9490
      Valid Loss: 0.0285 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.50 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [10/20]
      Train Loss: 0.0591 | Train Acc: 0.9794 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9603 | Train F2: 0.9603
      Valid Loss: 0.0430 | Valid Acc: 0.9753 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9130 | Valid F2: 0.9813
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=174 FP=6 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [11/20]
      Train Loss: 0.0573 | Train Acc: 0.9743 | Train Bad Recall: 0.9444 | Train Bad Precision: 0.9558 | Train F2: 0.9467
      Valid Loss: 0.0422 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [12/20]
      Train Loss: 0.0438 | Train Acc: 0.9846 | Train Bad Recall: 0.9643 | Train Bad Precision: 0.9759 | Train F2: 0.9666
      Valid Loss: 0.0238 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.65 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [13/20]
      Train Loss: 0.0431 | Train Acc: 0.9887 | Train Bad Recall: 0.9643 | Train Bad Precision: 0.9918 | Train F2: 0.9697
      Valid Loss: 0.0299 | Valid Acc: 0.9918 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9841 | Valid F2: 0.9841
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=62 TN=179 FP=1 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [14/20]
      Train Loss: 0.0659 | Train Acc: 0.9815 | Train Bad Recall: 0.9484 | Train Bad Precision: 0.9795 | Train F2: 0.9545
      Valid Loss: 0.0810 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.95 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [15/20]
      Train Loss: 0.0604 | Train Acc: 0.9774 | Train Bad Recall: 0.9484 | Train Bad Precision: 0.9637 | Train F2: 0.9514
      Valid Loss: 0.0410 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [16/20]
      Train Loss: 0.0354 | Train Acc: 0.9866 | Train Bad Recall: 0.9683 | Train Bad Precision: 0.9799 | Train F2: 0.9706
      Valid Loss: 0.0765 | Valid Acc: 0.9918 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 1.0000 | Valid F2: 0.9744
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=61 TN=180 FP=0 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [17/20]
      Train Loss: 0.0314 | Train Acc: 0.9918 | Train Bad Recall: 0.9762 | Train Bad Precision: 0.9919 | Train F2: 0.9793
      Valid Loss: 0.0728 | Valid Acc: 0.9835 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9683 | Valid F2: 0.9683
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=61 TN=178 FP=2 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [18/20]
      Train Loss: 0.0239 | Train Acc: 0.9928 | Train Bad Recall: 0.9841 | Train Bad Precision: 0.9880 | Train F2: 0.9849
      Valid Loss: 0.0366 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 0.9688 | Valid F2: 0.9810
      Valid Threshold: 0.55 | Selection: target_recall>=0.9
      TP=62 TN=178 FP=2 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [19/20]
      Train Loss: 0.0394 | Train Acc: 0.9866 | Train Bad Recall: 0.9683 | Train Bad Precision: 0.9799 | Train F2: 0.9706
      Valid Loss: 0.0837 | Valid Acc: 0.9877 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9839 | Valid F2: 0.9713
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=61 TN=179 FP=1 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [20/20]
      Train Loss: 0.0368 | Train Acc: 0.9846 | Train Bad Recall: 0.9643 | Train Bad Precision: 0.9759 | Train F2: 0.9666
      Valid Loss: 0.0462 | Valid Acc: 0.9877 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9545 | Valid F2: 0.9906
      Valid Threshold: 0.20 | Selection: target_recall>=0.9
      TP=63 TN=177 FP=3 FN=0
    ----------------------------------------------------------------------------------------------------
    [Fold 4] Best Epoch: 11
    [Fold 4] Best Threshold: 0.15
    [Fold 4] Acc=0.9959 Recall=1.0000 Precision=0.9844 F2=0.9968
    [Fold 4] saved to: /content/drive/MyDrive/defect_cv_with_final_test/cv/fold_4



```python
run_fold(5)
```

    
    ====================================================================================================
    Fold 5/5
    ====================================================================================================
    train size: 972
    valid size: 243
    train bad/good: 252 720
    valid bad/good: 63 180
    Epoch [1/20]
      Train Loss: 0.3172 | Train Acc: 0.8642 | Train Bad Recall: 0.6389 | Train Bad Precision: 0.7970 | Train F2: 0.6653
      Valid Loss: 0.1300 | Valid Acc: 0.9671 | Valid Bad Recall: 0.9524 | Valid Bad Precision: 0.9231 | Valid F2: 0.9464
      Valid Threshold: 0.55 | Selection: target_recall>=0.9
      TP=60 TN=175 FP=5 FN=3
    ----------------------------------------------------------------------------------------------------
    Epoch [2/20]
      Train Loss: 0.2079 | Train Acc: 0.9126 | Train Bad Recall: 0.7937 | Train Bad Precision: 0.8584 | Train F2: 0.8058
      Valid Loss: 0.1697 | Valid Acc: 0.9753 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 0.9385 | Valid F2: 0.9621
      Valid Threshold: 0.75 | Selection: target_recall>=0.9
      TP=61 TN=176 FP=4 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [3/20]
      Train Loss: 0.1646 | Train Acc: 0.9372 | Train Bad Recall: 0.8492 | Train Bad Precision: 0.9030 | Train F2: 0.8594
      Valid Loss: 0.0462 | Valid Acc: 0.9959 | Valid Bad Recall: 0.9841 | Valid Bad Precision: 1.0000 | Valid F2: 0.9873
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=62 TN=180 FP=0 FN=1
    ----------------------------------------------------------------------------------------------------
    Epoch [4/20]
      Train Loss: 0.1024 | Train Acc: 0.9619 | Train Bad Recall: 0.9127 | Train Bad Precision: 0.9388 | Train F2: 0.9178
      Valid Loss: 0.0255 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [5/20]
      Train Loss: 0.1223 | Train Acc: 0.9547 | Train Bad Recall: 0.8889 | Train Bad Precision: 0.9333 | Train F2: 0.8974
      Valid Loss: 0.0206 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [6/20]
      Train Loss: 0.1039 | Train Acc: 0.9671 | Train Bad Recall: 0.9325 | Train Bad Precision: 0.9400 | Train F2: 0.9340
      Valid Loss: 0.0310 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.30 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [7/20]
      Train Loss: 0.0738 | Train Acc: 0.9722 | Train Bad Recall: 0.9365 | Train Bad Precision: 0.9555 | Train F2: 0.9402
      Valid Loss: 0.0305 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [8/20]
      Train Loss: 0.0787 | Train Acc: 0.9763 | Train Bad Recall: 0.9325 | Train Bad Precision: 0.9751 | Train F2: 0.9408
      Valid Loss: 0.0469 | Valid Acc: 0.9918 | Valid Bad Recall: 0.9683 | Valid Bad Precision: 1.0000 | Valid F2: 0.9744
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=61 TN=180 FP=0 FN=2
    ----------------------------------------------------------------------------------------------------
    Epoch [9/20]
      Train Loss: 0.0778 | Train Acc: 0.9712 | Train Bad Recall: 0.9325 | Train Bad Precision: 0.9553 | Train F2: 0.9370
      Valid Loss: 0.0734 | Valid Acc: 0.9918 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9692 | Valid F2: 0.9937
      Valid Threshold: 0.70 | Selection: target_recall>=0.9
      TP=63 TN=178 FP=2 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [10/20]
      Train Loss: 0.0686 | Train Acc: 0.9794 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9603 | Train F2: 0.9603
      Valid Loss: 0.0128 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.20 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [11/20]
      Train Loss: 0.0912 | Train Acc: 0.9691 | Train Bad Recall: 0.9286 | Train Bad Precision: 0.9512 | Train F2: 0.9330
      Valid Loss: 0.0287 | Valid Acc: 0.9918 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9692 | Valid F2: 0.9937
      Valid Threshold: 0.15 | Selection: target_recall>=0.9
      TP=63 TN=178 FP=2 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [12/20]
      Train Loss: 0.0655 | Train Acc: 0.9722 | Train Bad Recall: 0.9405 | Train Bad Precision: 0.9518 | Train F2: 0.9427
      Valid Loss: 0.0500 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [13/20]
      Train Loss: 0.0545 | Train Acc: 0.9825 | Train Bad Recall: 0.9524 | Train Bad Precision: 0.9796 | Train F2: 0.9577
      Valid Loss: 0.0335 | Valid Acc: 0.9959 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9844 | Valid F2: 0.9968
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=179 FP=1 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [14/20]
      Train Loss: 0.0490 | Train Acc: 0.9866 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9878 | Train F2: 0.9657
      Valid Loss: 0.0551 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.85 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [15/20]
      Train Loss: 0.0619 | Train Acc: 0.9763 | Train Bad Recall: 0.9405 | Train Bad Precision: 0.9673 | Train F2: 0.9457
      Valid Loss: 0.0212 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.60 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [16/20]
      Train Loss: 0.0398 | Train Acc: 0.9866 | Train Bad Recall: 0.9683 | Train Bad Precision: 0.9799 | Train F2: 0.9706
      Valid Loss: 0.0224 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.55 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [17/20]
      Train Loss: 0.0338 | Train Acc: 0.9918 | Train Bad Recall: 0.9802 | Train Bad Precision: 0.9880 | Train F2: 0.9817
      Valid Loss: 0.0213 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [18/20]
      Train Loss: 0.0421 | Train Acc: 0.9856 | Train Bad Recall: 0.9603 | Train Bad Precision: 0.9837 | Train F2: 0.9649
      Valid Loss: 0.0331 | Valid Acc: 1.0000 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 1.0000 | Valid F2: 1.0000
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=180 FP=0 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [19/20]
      Train Loss: 0.0197 | Train Acc: 0.9959 | Train Bad Recall: 0.9921 | Train Bad Precision: 0.9921 | Train F2: 0.9921
      Valid Loss: 0.0203 | Valid Acc: 0.9918 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9692 | Valid F2: 0.9937
      Valid Threshold: 0.05 | Selection: target_recall>=0.9
      TP=63 TN=178 FP=2 FN=0
    ----------------------------------------------------------------------------------------------------
    Epoch [20/20]
      Train Loss: 0.0332 | Train Acc: 0.9907 | Train Bad Recall: 0.9881 | Train Bad Precision: 0.9765 | Train F2: 0.9857
      Valid Loss: 0.0175 | Valid Acc: 0.9877 | Valid Bad Recall: 1.0000 | Valid Bad Precision: 0.9545 | Valid F2: 0.9906
      Valid Threshold: 0.10 | Selection: target_recall>=0.9
      TP=63 TN=177 FP=3 FN=0
    ----------------------------------------------------------------------------------------------------
    [Fold 5] Best Epoch: 4
    [Fold 5] Best Threshold: 0.30
    [Fold 5] Acc=1.0000 Recall=1.0000 Precision=1.0000 F2=1.0000
    [Fold 5] saved to: /content/drive/MyDrive/defect_cv_with_final_test/cv/fold_5



```python
# =========================================================
# 11. CV集計 + OOF threshold + FINAL_EPOCHS決定
# =========================================================
fold_results = []
epoch_history_all_folds = []
oof_probs = np.zeros(len(trainval_paths), dtype=np.float32)
oof_labels = trainval_labels.copy()

for fold_number in range(1, NUM_FOLDS + 1):
    train_idx, valid_idx = fold_splits[fold_number - 1]
    fold_dir = cv_dir / f'fold_{fold_number}'

    summary_path = fold_dir / 'summary.json'
    history_path = fold_dir / 'history.csv'
    valid_pred_path = fold_dir / 'valid_predictions.csv'

    if not (summary_path.exists() and history_path.exists() and valid_pred_path.exists()):
        raise FileNotFoundError(f"Fold {fold_number} is not completed yet: {fold_dir}")

    summary = load_json(summary_path)
    history_df = pd.read_csv(history_path)
    valid_pred_df = pd.read_csv(valid_pred_path)

    fold_results.append(summary)
    epoch_history_all_folds.extend(history_df.to_dict('records'))

    loaded_probs = valid_pred_df['bad_prob'].values.astype(np.float32)
    if len(loaded_probs) != len(valid_idx):
        raise ValueError(f"Fold {fold_number}: valid_predictions length mismatch")

    oof_probs[valid_idx] = loaded_probs

cv_summary_df = pd.DataFrame(fold_results)
cv_summary_df.to_csv(RESULT_DIR / 'cv_summary.csv', index=False)

print("\n" + "=" * 100)
print("CV Summary")
print("=" * 100)
print(cv_summary_df)

for col in ['acc', 'bad_recall', 'bad_precision', 'bad_fbeta', 'best_threshold', 'best_epoch']:
    print(f"{col}: mean={cv_summary_df[col].mean():.4f}, std={cv_summary_df[col].std(ddof=0):.4f}")

epoch_history_df = pd.DataFrame(epoch_history_all_folds)
epoch_history_df.to_csv(RESULT_DIR / 'all_epoch_history.csv', index=False)

epoch_mean_df = epoch_history_df.groupby('epoch').agg({
    'train_loss': 'mean',
    'train_acc': 'mean',
    'train_bad_recall': 'mean',
    'train_bad_precision': 'mean',
    'train_bad_fbeta': 'mean',
    'valid_loss': 'mean',
    'valid_acc': 'mean',
    'valid_bad_recall': 'mean',
    'valid_bad_precision': 'mean',
    'valid_bad_fbeta': ['mean', 'std']
}).reset_index()

epoch_mean_df.columns = [
    'epoch',
    'train_loss_mean',
    'train_acc_mean',
    'train_bad_recall_mean',
    'train_bad_precision_mean',
    'train_bad_fbeta_mean',
    'valid_loss_mean',
    'valid_acc_mean',
    'valid_bad_recall_mean',
    'valid_bad_precision_mean',
    'valid_bad_fbeta_mean',
    'valid_bad_fbeta_std'
]

epoch_mean_df.to_csv(RESULT_DIR / 'epoch_mean_summary.csv', index=False)

best_epoch_row = epoch_mean_df.sort_values(
    by=['valid_bad_fbeta_mean', 'valid_bad_recall_mean', 'valid_bad_precision_mean'],
    ascending=[False, False, False]
).iloc[0]

FINAL_EPOCHS = int(best_epoch_row['epoch'])

print("\n" + "=" * 100)
print("Final epoch selected from 5-fold mean")
print("=" * 100)
print(best_epoch_row)
print("Selected FINAL_EPOCHS =", FINAL_EPOCHS)

global_best_thr_info, global_thr_table = find_best_threshold(
    labels=oof_labels,
    bad_probs=oof_probs,
    thresholds=THRESHOLDS,
    beta=BETA,
    target_recall=TARGET_RECALL
)

global_thr_df = pd.DataFrame(global_thr_table)
global_thr_df.to_csv(RESULT_DIR / 'global_threshold_search.csv', index=False)

FINAL_THRESHOLD = float(global_best_thr_info['threshold'])

print("\n" + "=" * 100)
print("Global OOF Threshold Search Result")
print("=" * 100)
print(f"Selected Threshold: {FINAL_THRESHOLD:.2f}")
print(f"Selection Mode    : {global_best_thr_info['selection_mode']}")
print(f"Acc               : {global_best_thr_info['acc']:.4f}")
print(f"Bad Recall        : {global_best_thr_info['bad_recall']:.4f}")
print(f"Bad Precision     : {global_best_thr_info['bad_precision']:.4f}")
print(f"Bad F{BETA:.0f}           : {global_best_thr_info['bad_fbeta']:.4f}")
print(f"TP={global_best_thr_info['tp']} TN={global_best_thr_info['tn']} FP={global_best_thr_info['fp']} FN={global_best_thr_info['fn']}")

aggregate_summary = {
    'final_epochs': int(FINAL_EPOCHS),
    'final_threshold': float(FINAL_THRESHOLD),
    'selection_mode': global_best_thr_info['selection_mode'],
    'oof_acc': float(global_best_thr_info['acc']),
    'oof_bad_recall': float(global_best_thr_info['bad_recall']),
    'oof_bad_precision': float(global_best_thr_info['bad_precision']),
    'oof_bad_fbeta': float(global_best_thr_info['bad_fbeta']),
    'beta': float(BETA),
    'target_recall': float(TARGET_RECALL)
}
save_json(RESULT_DIR / AGGREGATE_RESULT_NAME, aggregate_summary)

print("\nSaved aggregate summary to:", RESULT_DIR / AGGREGATE_RESULT_NAME)
```

    
    ====================================================================================================
    CV Summary
    ====================================================================================================
       fold  best_epoch  best_threshold       acc  bad_recall  bad_precision  \
    0     1           6            0.15  0.995885    1.000000       0.984375   
    1     2           9            0.25  0.995885    1.000000       0.984375   
    2     3           9            0.05  0.991770    0.968254       1.000000   
    3     4          11            0.15  0.995885    1.000000       0.984375   
    4     5           4            0.30  1.000000    1.000000       1.000000   
    
       bad_fbeta  tp   tn  fp  fn  \
    0   0.996835  63  179   1   0   
    1   0.996835  63  179   1   0   
    2   0.974441  61  180   0   2   
    3   0.996835  63  179   1   0   
    4   1.000000  63  180   0   0   
    
                                              model_path  
    0  /content/drive/MyDrive/defect_cv_with_final_te...  
    1  /content/drive/MyDrive/defect_cv_with_final_te...  
    2  /content/drive/MyDrive/defect_cv_with_final_te...  
    3  /content/drive/MyDrive/defect_cv_with_final_te...  
    4  /content/drive/MyDrive/defect_cv_with_final_te...  
    acc: mean=0.9959, std=0.0026
    bad_recall: mean=0.9937, std=0.0127
    bad_precision: mean=0.9906, std=0.0077
    bad_fbeta: mean=0.9930, std=0.0094
    best_threshold: mean=0.1800, std=0.0872
    best_epoch: mean=7.8000, std=2.4819
    
    ====================================================================================================
    Final epoch selected from 5-fold mean
    ====================================================================================================
    epoch                       10.000000
    train_loss_mean              0.068654
    train_acc_mean               0.977984
    train_bad_recall_mean        0.951587
    train_bad_precision_mean     0.963033
    train_bad_fbeta_mean         0.953833
    valid_loss_mean              0.050971
    valid_acc_mean               0.986831
    valid_bad_recall_mean        0.990476
    valid_bad_precision_mean     0.960926
    valid_bad_fbeta_mean         0.984260
    valid_bad_fbeta_std          0.015118
    Name: 9, dtype: float64
    Selected FINAL_EPOCHS = 10
    
    ====================================================================================================
    Global OOF Threshold Search Result
    ====================================================================================================
    Selected Threshold: 0.15
    Selection Mode    : target_recall>=0.9
    Acc               : 0.9885
    Bad Recall        : 0.9873
    Bad Precision     : 0.9688
    Bad F2           : 0.9836
    TP=311 TN=890 FP=10 FN=4
    
    Saved aggregate summary to: /content/drive/MyDrive/defect_cv_with_final_test/aggregate_summary.json



```python
# =========================================================
# 12. train_val 全体で最終学習
# =========================================================
aggregate_summary_path = RESULT_DIR / AGGREGATE_RESULT_NAME
if not aggregate_summary_path.exists():
    raise FileNotFoundError("Run セル16 first.")

aggregate_summary = load_json(aggregate_summary_path)
FINAL_EPOCHS = int(aggregate_summary['final_epochs'])
FINAL_THRESHOLD = float(aggregate_summary['final_threshold'])

print("FINAL_EPOCHS   =", FINAL_EPOCHS)
print("FINAL_THRESHOLD=", FINAL_THRESHOLD)

trainval_dataset = DefectDataset(trainval_paths, trainval_labels, transform=train_transform)
trainval_loader = DataLoader(
    trainval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

final_model = create_model(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

final_train_history = []

for epoch in range(FINAL_EPOCHS):
    train_metrics = train_one_epoch(final_model, trainval_loader, criterion, final_optimizer, device, beta=BETA)
    final_train_history.append({
        'epoch': epoch + 1,
        'train_loss': train_metrics['loss'],
        'train_acc': train_metrics['acc'],
        'train_bad_recall': train_metrics['bad_recall'],
        'train_bad_precision': train_metrics['bad_precision'],
        'train_bad_fbeta': train_metrics['bad_fbeta']
    })

    print(f"Final Epoch [{epoch+1}/{FINAL_EPOCHS}]")
    print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f} | Train Bad Recall: {train_metrics['bad_recall']:.4f} | Train Bad Precision: {train_metrics['bad_precision']:.4f} | Train F{BETA:.0f}: {train_metrics['bad_fbeta']:.4f}")
    print("-" * 100)

final_model_path = RESULT_DIR / FINAL_MODEL_NAME
torch.save(final_model.state_dict(), final_model_path)
pd.DataFrame(final_train_history).to_csv(RESULT_DIR / FINAL_TRAIN_HISTORY_NAME, index=False)

print("Saved final model to:", final_model_path)
```

    FINAL_EPOCHS   = 10
    FINAL_THRESHOLD= 0.15
    Final Epoch [1/10]
      Train Loss: 0.3099 | Train Acc: 0.8617 | Train Bad Recall: 0.6508 | Train Bad Precision: 0.7795 | Train F2: 0.6730
    ----------------------------------------------------------------------------------------------------
    Final Epoch [2/10]
      Train Loss: 0.1760 | Train Acc: 0.9399 | Train Bad Recall: 0.8444 | Train Bad Precision: 0.9172 | Train F2: 0.8581
    ----------------------------------------------------------------------------------------------------
    Final Epoch [3/10]
      Train Loss: 0.1293 | Train Acc: 0.9605 | Train Bad Recall: 0.9143 | Train Bad Precision: 0.9320 | Train F2: 0.9178
    ----------------------------------------------------------------------------------------------------
    Final Epoch [4/10]
      Train Loss: 0.1175 | Train Acc: 0.9572 | Train Bad Recall: 0.9016 | Train Bad Precision: 0.9311 | Train F2: 0.9073
    ----------------------------------------------------------------------------------------------------
    Final Epoch [5/10]
      Train Loss: 0.0817 | Train Acc: 0.9679 | Train Bad Recall: 0.9238 | Train Bad Precision: 0.9510 | Train F2: 0.9291
    ----------------------------------------------------------------------------------------------------
    Final Epoch [6/10]
      Train Loss: 0.0888 | Train Acc: 0.9621 | Train Bad Recall: 0.9111 | Train Bad Precision: 0.9410 | Train F2: 0.9169
    ----------------------------------------------------------------------------------------------------
    Final Epoch [7/10]
      Train Loss: 0.0826 | Train Acc: 0.9778 | Train Bad Recall: 0.9460 | Train Bad Precision: 0.9675 | Train F2: 0.9503
    ----------------------------------------------------------------------------------------------------
    Final Epoch [8/10]
      Train Loss: 0.0764 | Train Acc: 0.9794 | Train Bad Recall: 0.9460 | Train Bad Precision: 0.9739 | Train F2: 0.9515
    ----------------------------------------------------------------------------------------------------
    Final Epoch [9/10]
      Train Loss: 0.0504 | Train Acc: 0.9835 | Train Bad Recall: 0.9587 | Train Bad Precision: 0.9773 | Train F2: 0.9624
    ----------------------------------------------------------------------------------------------------
    Final Epoch [10/10]
      Train Loss: 0.0359 | Train Acc: 0.9860 | Train Bad Recall: 0.9619 | Train Bad Precision: 0.9838 | Train F2: 0.9662
    ----------------------------------------------------------------------------------------------------
    Saved final model to: /content/drive/MyDrive/defect_cv_with_final_test/final_resnet18_512_with_holdout_test.pth



```python
# =========================================================
# 13. hold-out final test で1回だけ評価
# =========================================================
aggregate_summary_path = RESULT_DIR / AGGREGATE_RESULT_NAME
final_model_path = RESULT_DIR / FINAL_MODEL_NAME

if not aggregate_summary_path.exists():
    raise FileNotFoundError("Run セル16 first.")
if not final_model_path.exists():
    raise FileNotFoundError("Run セル17 first.")

aggregate_summary = load_json(aggregate_summary_path)
FINAL_EPOCHS = int(aggregate_summary['final_epochs'])
FINAL_THRESHOLD = float(aggregate_summary['final_threshold'])

print("FINAL_EPOCHS   =", FINAL_EPOCHS)
print("FINAL_THRESHOLD=", FINAL_THRESHOLD)

test_dataset = DefectDataset(test_paths, test_labels, transform=valid_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

final_model = create_model(device)
final_model.load_state_dict(torch.load(final_model_path, map_location=device))
final_model.eval()

test_out = collect_probs(final_model, test_loader, criterion, device)
test_metrics = metrics_from_probs(
    labels=test_out['labels'],
    bad_probs=test_out['bad_probs'],
    threshold=FINAL_THRESHOLD,
    beta=BETA
)

print(f"Final Test Threshold: {FINAL_THRESHOLD:.2f}")
print(f"Final Test Acc      : {test_metrics['acc']:.4f}")
print(f"Final Test Recall   : {test_metrics['bad_recall']:.4f}")
print(f"Final Test Precision: {test_metrics['bad_precision']:.4f}")
print(f"Final Test F{BETA:.0f}       : {test_metrics['bad_fbeta']:.4f}")
print(f"TP={test_metrics['tp']} TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']}")

save_failure_images(
    images_cpu=test_out['images_cpu'],
    labels=test_out['labels'],
    bad_probs=test_out['bad_probs'],
    preds=test_metrics['preds'],
    paths=test_out['paths'],
    save_dir=RESULT_DIR / 'final_test_failure_images'
)

with open(RESULT_DIR / FINAL_TEST_RESULT_NAME, 'w', encoding='utf-8') as f:
    json.dump({
        'final_epochs': FINAL_EPOCHS,
        'final_threshold': FINAL_THRESHOLD,
        'acc': float(test_metrics['acc']),
        'bad_recall': float(test_metrics['bad_recall']),
        'bad_precision': float(test_metrics['bad_precision']),
        'bad_fbeta': float(test_metrics['bad_fbeta']),
        'tp': int(test_metrics['tp']),
        'tn': int(test_metrics['tn']),
        'fp': int(test_metrics['fp']),
        'fn': int(test_metrics['fn']),
        'beta': BETA,
        'target_recall': TARGET_RECALL,
        'model_path': str(final_model_path)
    }, f, ensure_ascii=False, indent=2)

print("\nSaved final test result to:", RESULT_DIR / FINAL_TEST_RESULT_NAME)
print("Saved results to         :", RESULT_DIR)
print("Done.")
```

    FINAL_EPOCHS   = 10
    FINAL_THRESHOLD= 0.15
    Final Test Threshold: 0.15
    Final Test Acc      : 0.9926
    Final Test Recall   : 0.9714
    Final Test Precision: 1.0000
    Final Test F2       : 0.9770
    TP=34 TN=100 FP=0 FN=1
    
    Saved final test result to: /content/drive/MyDrive/defect_cv_with_final_test/final_test_result.json
    Saved results to         : /content/drive/MyDrive/defect_cv_with_final_test
    Done.

