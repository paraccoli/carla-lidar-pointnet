import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob

def load_model(model_path):
    """保存されたモデルを読み込む"""
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    return model

def preprocess_point_cloud(point_cloud_path, max_points=2000):
    """点群データを前処理する"""
    # 点群データの読み込み
    points = np.load(point_cloud_path)
    
    # 点群の中心化
    points_xyz = points[:, :3]
    center = np.mean(points_xyz, axis=0)
    points_centered = points.copy()
    points_centered[:, :3] = points_xyz - center
    
    # 点数の調整
    num_points = points_centered.shape[0]
    if num_points > max_points:
        # ランダムにサンプリング
        indices = np.random.choice(num_points, max_points, replace=False)
        points_centered = points_centered[indices]
    else:
        # 足りない部分を0埋め
        padded = np.zeros((max_points, points.shape[1]), dtype=np.float32)
        padded[:num_points] = points_centered
        points_centered = padded
    
    # バッチ次元の追加
    return np.expand_dims(points_centered, axis=0)

def visualize_point_cloud(points, predictions=None, title="Point Cloud Visualization"):
    """点群を可視化"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # パディングされた点（すべて0の点）をフィルタリング
    non_zero = ~np.all(points[:, :3] == 0, axis=1)
    filtered_points = points[non_zero]
    
    # 点のプロット
    ax.scatter(
        filtered_points[:, 0],
        filtered_points[:, 1],
        filtered_points[:, 2],
        c=filtered_points[:, 2],  # Z座標で色付け
        cmap='viridis',
        s=2
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if predictions is not None:
        pred_class = np.argmax(predictions)
        prob = predictions[pred_class]
        title = f"{title} - Predicted Class: {pred_class}, Probability: {prob:.4f}"
    
    ax.set_title(title)
    plt.tight_layout()
    return fig

def predict_on_new_data(model_path='_out/pointnet_model/pointnet_model.h5', data_dir='_out/lidar_data'):
    """新しいデータで予測を行う"""
    model = load_model(model_path)
    
    # 最新のLiDARデータディレクトリを特定
    run_dirs = glob(os.path.join(data_dir, 'run_*'))
    if not run_dirs:
        print("No LiDAR data directories found")
        return
    
    latest_run = max(run_dirs, key=os.path.getctime)
    print(f"Using data from: {latest_run}")
    
    # いくつかのサンプルを選択
    lidar_files = glob(os.path.join(latest_run, 'lidar_points_*.npy'))
    
    if not lidar_files:
        print("No LiDAR data files found")
        return
    
    # 出力ディレクトリ
    output_dir = '_out/predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ランダムに10個のサンプルを選択
    sample_files = np.random.choice(lidar_files, min(10, len(lidar_files)), replace=False)
    
    for i, file_path in enumerate(sample_files):
        print(f"Processing file: {os.path.basename(file_path)}")
        
        # 前処理
        processed_data = preprocess_point_cloud(file_path)
        
        # 予測
        predictions = model.predict(processed_data)[0]
        
        # 可視化
        fig = visualize_point_cloud(processed_data[0], predictions)
        plt.savefig(os.path.join(output_dir, f'prediction_{i}.png'))
        plt.close(fig)
        
        print(f"Prediction: Class {np.argmax(predictions)} with probability {np.max(predictions):.4f}")
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    predict_on_new_data()