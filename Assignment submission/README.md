# Defect-Detection-Model

製品画像を **bad（不良品） / good（良品）** に分類する画像認識モデルの実装・評価リポジトリです。  
本プロジェクトでは、**不良品の見逃し抑制**を重視し、ResNet18 を用いた二値分類モデルを構築しました。  
評価は **5-fold Cross Validation + hold-out final test** で実施し、モデル選定の安定性と最終評価の公平性を両立しています。

---

## 1. Project Overview

### 目的
製品画像から **bad（不良品） / good（良品）** を自動判定し、検査の効率化と品質管理の安定化につなげることを目的としています。  
特に、実運用を意識して **bad クラスの再現率（Recall）** を重要指標として設定しました。

### KPI
- 主KPI: **不良品再現率（bad Recall）90%以上**
- 副KPI: **不良品適合率（bad Precision）80%以上**
- 補助KPI: **F2-score 最大化**

F2-score を採用した理由は、Precision よりも Recall を重く評価し、不良品の見逃し防止をより強く反映させるためです。

---

## 2. Dataset

データセットは以下の 2 クラスから構成されます。

- `bad`: 350枚
- `good`: 1000枚
- 合計: **1350枚**

評価の公平性を確保するため、最初に hold-out final test を分離しました。

### Hold-out split
- trainval: **1215枚**
  - bad: 315
  - good: 900
- final test: **135枚**
  - bad: 35
  - good: 100

---

## 3. Model

最終モデルには **ResNet18** を採用しました。  
PyTorch の事前学習済み重みを使用し、最終全結合層を 2 クラス分類用に置き換えています。

### 採用理由
初期ベースラインとして自作 SimpleCNN を試したものの、検証データでの bad Recall は最大でも **0.8286** にとどまり、主KPI の 90% に届きませんでした。  
そのため、より高い特徴抽出能力を持ち、少量データでも有効な特徴を利用しやすい ResNet18 へ変更しました。

### Baseline (SimpleCNN)
- 入力画像サイズ: 512×512
- 4層の畳み込みブロック + 全結合層
- bad クラス重み付き CrossEntropyLoss を使用
- best epoch は valid bad Recall 最優先で保存

---

## 4. Data Augmentation / Preprocessing

学習時には以下の前処理・拡張を適用しました。

### Train transform
- Resize: 512×512
- RandomHorizontalFlip(p=0.5)
- RandomRotation
- Normalize(ImageNet mean/std)

### Validation / Test transform
- Resize: 512×512
- Normalize(ImageNet mean/std)

---

## 5. Evaluation Strategy

本プロジェクトでは、単一 split に依存した楽観的な評価を避けるために、  
**5-fold CV** と **hold-out final test** を組み合わせた評価設計を採用しました。

### 5-fold CV の目的
- 分割による偶然の当たり外れを減らす
- fold ごとの性能ばらつきを確認する
- 最終設定（epoch / threshold）の安定した選定に使う

### hold-out final test の目的
- モデル選定に使っていない完全未使用データで最終評価する
- CV に対する過適合や評価の楽観化を防ぐ

### Final setting selection
- **Final epoch** は、各 fold の epoch 履歴を集約し、平均 validation 性能が最も良い epoch から決定
- **Final threshold** は、fold ごとの閾値の単純平均ではなく、OOF（out-of-fold）予測全体をまとめて再探索して決定

これにより、単一 fold に依存しない、再現性の高い最終設定を採用しています。

---

## 6. Results

### Final Test Result (hold-out)
最終テストでは以下の性能を達成しました。

- Final Epochs: **10**
- Final Threshold: **0.15**
- Accuracy: **0.9926**
- bad Recall: **0.9714**
- bad Precision: **1.0000**
- F2-score: **0.9770**
- Confusion Matrix:
  - TP = 34
  - TN = 100
  - FP = 0
  - FN = 1

### KPI達成状況
- 主KPI（bad Recall 90%以上）: **達成**
- 副KPI（bad Precision 80%以上）: **達成**
- 補助KPI（F2-score 最大化）: **高水準で達成**

### 解釈
- 不良品 35 枚のうち **34 枚を正しく検出**
- 見逃しは **1件**
- 良品 100 枚に対する **誤検知は 0件**

Recall を重視した設計でありながら、Precision を損なわずに高性能を実現できた点が今回の重要な結果です。

### 5-fold CV Summary
- Accuracy: **0.9959 ± 0.0026**
- bad Recall: **0.9937 ± 0.0127**
- bad Precision: **0.9906 ± 0.0077**
- bad F2-score: **0.9930 ± 0.0094**

分割を変えても高性能を維持しており、評価結果のばらつきは小さいことを確認できました。

---

## 7. Baseline Comparison

SimpleCNN をベースラインとして構築し、まず基準性能を確認しました。  
ただし、best epoch においても valid bad Recall は **0.8286** で、主KPI の 90% には到達しませんでした。  
この結果から、限られたデータ数に対しては自作CNNよりも転移学習モデルの方が有効と判断し、ResNet18 に移行しました。

---

## 8. Outputs

本プロジェクトでは、学習・評価の結果として以下のファイルを出力します。

### 評価結果
- `outputs/final_test_result.json`  
  hold-out final test の評価結果を保存
- `outputs/cv_summary.csv`  
  5-fold CV の要約結果を保存
- `outputs/epoch_mean_summary.csv`  
  epoch ごとの平均 validation 性能を保存
- `outputs/trainval_split.csv`  
  trainval データ一覧を保存
- `outputs/final_test_split.csv`  
  final test データ一覧を保存

### 可視化結果
- `results/confusion_matrix.png`  
  最終テストの混同行列
- `results/misclassified_examples.png`  
  誤分類画像の可視化
- `results/gradcam_examples.png`  
  Grad-CAM による注目領域可視化（作成した場合）

### 重みファイル
- `final_resnet18_512_with_holdout_test.pth`  
  最終提出用の学習済み重みファイル  
  ※ 重みファイルは Google Drive に別途アップロード

---

## 9. How to Run

### 実行環境
- Python
- PyTorch
- torchvision
- pandas
- numpy
- Pillow
- Google Colab + Google Drive を想定

### 実行手順
1. データセットを Google Drive に配置
2. Google Colab で Drive をマウント
3. `main.py` を実行
4. 5-fold CV を実施
5. CV 集計および OOF threshold を選定
6. trainval 全体で最終学習
7. hold-out final test で最終評価

---

## 10. Repository Structure

```text
Defect-Detection-Model/
├── README.md
├── requirements.txt
├── main.py
├── time.py
├── docs/
│   ├── main.md
│   └── time.md
├── outputs/
│   ├── final_test_result.json
│   ├── cv_summary.csv
│   ├── epoch_mean_summary.csv
│   ├── trainval_split.csv
│   └── final_test_split.csv
└── results/
    ├── confusion_matrix.png
    ├── misclassified_examples.png
    └── gradcam_examples.png
