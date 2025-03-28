# CARLA LiDAR Point Cloud Processing - 技術手法の詳細

## 1. データ収集プロセス

### LiDARセンサー構成
CARLAシミュレータ内で使用されるLiDARセンサーは以下の設定を使用しています：
- チャネル数: 32
- 回転周波数: 10 Hz
- 点生成頻度: 100,000 points/秒
- 範囲: 50m

### データ形式
各フレームで2つのファイルが生成されます：
- `lidar_points_*.npy`: 点群データ (N×4 配列、各点は [x, y, z, intensity])
- `transform_*.txt`: 車両の位置と姿勢情報

## 2. 点群前処理

### 正規化
- 各点群を中心化（平均位置を原点に移動）
- 固定サイズ（2,000点）に正規化：大きい点群はランダムサンプリング、小さい点群は0パディング

### データ分割
- 訓練セット (60%): モデルの学習に使用
- 検証セット (20%): ハイパーパラメータのチューニングと早期停止
- テストセット (20%): 最終評価

## 3. PointNetアーキテクチャ

### モデル構造
1. **入力レイヤー**: (None, 2000, 4) - バッチ、点の数、特徴量
2. **共有MLP**: 点ごとの特徴抽出
   - Conv1D(64) → BN → ReLU
   - Conv1D(128) → BN → ReLU
   - Conv1D(256) → BN → ReLU
3. **グローバル特徴抽出**: GlobalMaxPooling
4. **分類ヘッド**:
   - Dense(512) → BN → Dropout(0.3)
   - Dense(256) → BN → Dropout(0.3)
   - Dense(128) → BN
   - Dense(2) → Softmax

### 順列不変性
PointNetアーキテクチャの核心は、入力点の順序に影響されない（順列不変）特徴抽出です。これは共有MLPと最大プーリング操作によって実現されています。

## 4. トレーニング方法

### ハイパーパラメータ
- バッチサイズ: 32
- エポック数: 最大50（早期停止を適用）
- オプティマイザ: Adam (学習率 0.001)
- 損失関数: カテゴリカルクロスエントロピー

### 正則化テクニック
- バッチ正規化
- ドロップアウト (0.3)
- 早期停止 (検証損失が10エポック改善しない場合)
- 学習率スケジューリング (検証損失が5エポック改善しない場合、学習率を半分に)

### クラス不均衡への対処
- クラス重み付け: クラスの頻度の逆数に基づいて重みを割り当て

## 5. 評価指標

- 精度 (Accuracy)
- 損失 (Categorical Cross-Entropy)
- 混同行列 (Confusion Matrix)
- 分類レポート (Precision, Recall, F1-score)