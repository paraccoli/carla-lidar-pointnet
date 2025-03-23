import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# GPU設定部分の改善
import tensorflow as tf

# Check for GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

if physical_devices:
    try:
        # 全GPUのメモリ増加を許可
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Set memory growth for GPU: {gpu}")
        
        # GPUメモリ使用量制限（必要に応じて）
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB制限
        )
        
        print("GPU configuration successful")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU available, using CPU")

def create_pointnet_model(input_shape, num_classes=2):
    """
    PointNetモデルの構築
    
    Args:
        input_shape: 入力点群の形状 (例: (2000, 4))
        num_classes: 分類クラス数
        
    Returns:
        PointNetモデル
    """
    # 入力レイヤー
    inputs = layers.Input(shape=input_shape)
    
    # 座標とその他の特徴量を分離
    coords = inputs[:, :, :3]  # xyz座標
    features = inputs  # すべての特徴を含む
    
    # ポイントごとの特徴抽出 (Shared MLP)
    x = layers.Conv1D(64, 1, activation='relu')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # グローバル特徴
    x = layers.GlobalMaxPooling1D()(x)
    
    # 分類ブランチ
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 出力レイヤー
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # モデルの定義
    model = models.Model(inputs=inputs, outputs=outputs, name='pointnet')
    
    return model

def load_dataset(dataset_dir):
    """
    保存されたデータセットを読み込む
    
    Args:
        dataset_dir: データセットディレクトリパス
        
    Returns:
        各データセットとそのラベル
    """
    X_train = np.load(os.path.join(dataset_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(dataset_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(dataset_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(dataset_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # クラスの分布を確認
    print(f"Training labels distribution: {np.bincount(y_train.astype(np.int32))}")
    print(f"Validation labels distribution: {np.bincount(y_val.astype(np.int32))}")
    print(f"Test labels distribution: {np.bincount(y_test.astype(np.int32))}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_class_weights(y_train):
    """
    クラスの重みを計算してクラス不均衡に対処
    
    Args:
        y_train: トレーニングラベル
        
    Returns:
        クラスの重み
    """
    # クラスの分布を計算
    class_counts = np.bincount(y_train.astype(np.int32))
    total = len(y_train)
    
    # クラスの重みを計算（少数クラスの重みを増加）
    weights = total / (len(class_counts) * class_counts)
    
    # 重み辞書を作成
    class_weights = {i: weight for i, weight in enumerate(weights)}
    print(f"Class weights: {class_weights}")
    
    return class_weights

def train_model(dataset_dir='_out/lidar_dataset', output_dir='_out/pointnet_model', epochs=30, batch_size=32):
    """
    PointNetモデルの訓練
    
    Args:
        dataset_dir: データセットのディレクトリ
        output_dir: 出力モデルとグラフの保存先
        epochs: エポック数
        batch_size: バッチサイズ
    """
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # データセットの読み込み
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_dir)
    
    # クラス数
    num_classes = len(np.unique(y_train))
    
    # ラベルをone-hot形式に変換
    y_train_onehot = to_categorical(y_train, num_classes)
    y_val_onehot = to_categorical(y_val, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)
    
    # クラスの重みを計算
    class_weights = compute_class_weights(y_train)
    
    # モデルの構築
    model = create_pointnet_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    
    # モデルの要約の保存
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # モデルのコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # コールバックの定義
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            update_freq='epoch'
        )
    ]
    
    # モデルの訓練
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        class_weight=class_weights if num_classes > 1 else None,
        verbose=1
    )
    
    # 訓練履歴の保存
    np.save(os.path.join(output_dir, 'training_history.npy'), history.history)
    
    # 学習曲線の描画
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # テストデータでの評価
    print("\nEvaluating on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # 混同行列の作成
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 分類レポートの保存
    report = classification_report(y_test, y_pred_classes)
    print("\nClassification Report:")
    print(report)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # 結果の保存
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }
    
    # モデルの保存
    model.save(os.path.join(output_dir, 'pointnet_model.h5'))
    print(f"Model saved to {os.path.join(output_dir, 'pointnet_model.h5')}")
    
    return model, history, results

if __name__ == "__main__":
    # PointNetディレクトリから実行している場合は、上位ディレクトリを参照するパスに変更
    dataset_dir = '../_out/lidar_dataset'
    output_dir = '../_out/pointnet_model'
    
    train_model(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=50,
        batch_size=32
    )