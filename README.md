# CARLA LiDAR Point Cloud Processing with PointNet

このプロジェクトは、CARLAシミュレータを使用して自律運転車両から収集したLiDARポイントクラウドデータを、PointNetアーキテクチャを用いて処理する一連のパイプラインを実装しています。

## 主な機能

- CARLAシミュレータでのLiDARデータ収集
- 点群データの前処理とデータセット作成
- PointNetを使用した点群の分類モデルのトレーニング
- 学習したモデルによるリアルタイム推論

## プロジェクト構造

```
carla-lidar-pointnet/
├── scripts/                    # 主要スクリプト
│   ├── modified_automatic_control.py  # データ収集
│   ├── dataset_preprocessing.py       # データセット準備
│   ├── train_pointnet.py             # モデル訓練
│   └── predict_pointnet.py           # 推論
├── _out/                      # 出力ディレクトリ（大きなファイルはgitignoreされる）
│   ├── lidar_data/            # 生のLiDARデータ
│   ├── lidar_dataset/         # 前処理済みデータセット
│   ├── pointnet_model/        # 訓練済みモデルとグラフ
│   └── predictions/           # 予測結果と可視化
├── results/                   # 公開用の結果サンプル
│   ├── metrics/               # 評価指標サンプル
│   └── visualizations/        # 可視化サンプル
├── docs/                      # プロジェクトドキュメント
│   └── methodology.md         # 手法の詳細説明
└── data/                      # 追加データリソース
```

## 要件

- Python 3.7+
- CARLA 0.9.10+
- TensorFlow 2.4+
- その他の依存パッケージ（requirements.txtを参照）

## インストール方法

```bash
# CARLAのインストール（公式サイトからダウンロード）
# https://carla.org/download/

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 1. LiDARデータの収集

```bash
python scripts/modified_automatic_control.py --sync --weather ClearNoon --output-dir _out/lidar_data
```

### 2. データの前処理とデータセット作成

```bash
python scripts/dataset_preprocessing.py
```

### 3. PointNetモデルのトレーニング

```bash
python scripts/train_pointnet.py
```

### 4. モデルを使った推論

```bash
python scripts/predict_pointnet.py
```

## データセットの詳細

- 点群サイズ: 各サンプル2,000点
- 点の特徴量: [x, y, z, intensity]
- サンプリングレート: 5フレームごとに1フレーム使用
- データ分割: 訓練(60%)/検証(20%)/テスト(20%)

## モデルアーキテクチャ

PointNetアーキテクチャを使用し、以下の特徴があります：
- 入力: 可変サイズの点群（2,000点に正規化）
- 特徴抽出: 共有MLPによる点ごとの特徴抽出
- グローバル特徴: 最大プーリングによる順列不変性の確保
- 分類ヘッド: 512-256-128-2ユニットのMLP層

詳細な技術情報は[methodology.md](docs/methodology.md)を参照してください。

## 結果

トレーニング結果の概要:
- 訓練精度: XX%
- 検証精度: XX%
- テスト精度: XX%

詳細な評価結果とグラフは[results](results/)ディレクトリにあります。

## 参考文献

1. CARLA Simulator: https://carla.org/
2. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation: https://arxiv.org/abs/1612.00593

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。