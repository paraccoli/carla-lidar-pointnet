import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 日本語フォント設定
import matplotlib
# 日本語フォントの設定（適宜環境に合わせたフォントに変更可能）
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'IPAGothic', 'Arial']
# 負の値を表示する場合の設定
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_point_cloud(points, predictions=None, ground_truth=None, title="Point Cloud Visualization", 
                          save_path=None, show=True, figsize=(10, 8), point_size=2):
    """
    点群データを3D可視化する関数
    
    Args:
        points (numpy.ndarray): 点群データ (N, 3)または(N, 4)の形状
        predictions (numpy.ndarray, optional): 予測クラス (N,)の形状
        ground_truth (numpy.ndarray, optional): 正解クラス (N,)の形状
        title (str): 図のタイトル
        save_path (str, optional): 図を保存するパス
        show (bool): 図を表示するかどうか
        figsize (tuple): 図のサイズ
        point_size (float): 点のサイズ
        
    Returns:
        matplotlib.figure.Figure: 作成された図のオブジェクト
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # パディングされた点（すべて0の点）をフィルタリング
    if points.shape[1] >= 3:  # 少なくとも xyz 座標があることを確認
        non_zero = ~np.all(points[:, :3] == 0, axis=1)
        filtered_points = points[non_zero]
    else:
        filtered_points = points
    
    # 点の色付け設定
    if predictions is not None and ground_truth is not None:
        # 予測と正解ラベルの両方がある場合
        filtered_predictions = predictions[non_zero] if len(predictions) == len(points) else None
        filtered_ground_truth = ground_truth[non_zero] if len(ground_truth) == len(points) else None
        
        # 正解/不正解で色分け
        colors = np.zeros(len(filtered_points), dtype=np.int32)
        colors[filtered_predictions == filtered_ground_truth] = 1  # 正解は緑
        colors[filtered_predictions != filtered_ground_truth] = 2  # 不正解は赤
        
        color_map = {0: 'blue', 1: 'green', 2: 'red'}
        color_list = [color_map[c] for c in colors]
        scatter = ax.scatter(
            filtered_points[:, 0],
            filtered_points[:, 1],
            filtered_points[:, 2],
            c=color_list,
            s=point_size
        )
        
        # 凡例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='正解'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='不正解')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
    elif predictions is not None:
        # 予測のみがある場合
        filtered_predictions = predictions[non_zero] if len(predictions) == len(points) else None
        scatter = ax.scatter(
            filtered_points[:, 0],
            filtered_points[:, 1],
            filtered_points[:, 2],
            c=filtered_predictions,
            cmap='viridis',
            s=point_size
        )
        plt.colorbar(scatter, ax=ax, label='予測クラス')
        
    else:
        # 色付けなしの場合
        if filtered_points.shape[1] >= 4:  # インテンシティがある場合
            scatter = ax.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                filtered_points[:, 2],
                c=filtered_points[:, 3],  # インテンシティで色付け
                cmap='viridis',
                s=point_size
            )
            plt.colorbar(scatter, ax=ax, label='Intensity')
        else:
            # 高さ（Z座標）で色付け
            scatter = ax.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                filtered_points[:, 2],
                c=filtered_points[:, 2],
                cmap='viridis',
                s=point_size
            )
            plt.colorbar(scatter, ax=ax, label='高さ (Z)')
    
    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 軸の範囲を均等に（アスペクト比を保つ）
    x_range = filtered_points[:, 0].max() - filtered_points[:, 0].min()
    y_range = filtered_points[:, 1].max() - filtered_points[:, 1].min()
    z_range = filtered_points[:, 2].max() - filtered_points[:, 2].min()
    max_range = max(x_range, y_range, z_range) / 2
    
    mid_x = (filtered_points[:, 0].max() + filtered_points[:, 0].min()) / 2
    mid_y = (filtered_points[:, 1].max() + filtered_points[:, 1].min()) / 2
    mid_z = (filtered_points[:, 2].max() + filtered_points[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def visualize_training_history(history, save_path=None, show=True, figsize=(15, 6)):
    """
    モデルの学習履歴を可視化する関数
    
    Args:
        history (tf.keras.callbacks.History): 学習履歴オブジェクト
        save_path (str, optional): 図を保存するパス
        show (bool): 図を表示するかどうか
        figsize (tuple): 図のサイズ
        
    Returns:
        matplotlib.figure.Figure: 作成された図のオブジェクト
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 精度のプロット
    ax1.plot(history.history['accuracy'], label='訓練精度')
    ax1.plot(history.history['val_accuracy'], label='検証精度')
    ax1.set_title('モデル精度')
    ax1.set_xlabel('エポック')
    ax1.set_ylabel('精度')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # 損失のプロット
    ax2.plot(history.history['loss'], label='訓練損失')
    ax2.plot(history.history['val_loss'], label='検証損失')
    ax2.set_title('モデル損失')
    ax2.set_xlabel('エポック')
    ax2.set_ylabel('損失')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, 
                           title='Confusion Matrix', save_path=None, show=True, figsize=(8, 6)):
    """
    混同行列を可視化する関数
    
    Args:
        y_true (numpy.ndarray): 正解ラベル
        y_pred (numpy.ndarray): 予測ラベル
        class_names (list, optional): クラス名のリスト
        normalize (bool): 混同行列を正規化するかどうか
        title (str): 図のタイトル
        save_path (str, optional): 図を保存するパス
        show (bool): 図を表示するかどうか
        figsize (tuple): 図のサイズ
        
    Returns:
        matplotlib.figure.Figure: 作成された図のオブジェクト
    """
    # 混同行列を計算
    cm = confusion_matrix(y_true, y_pred)
    
    # 正規化オプション
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # クラス名が指定されていない場合はインデックスを使用
    if class_names is None:
        class_names = [f'クラス {i}' for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('予測ラベル')
    ax.set_ylabel('真のラベル')
    ax.set_title(title)
    
    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def visualize_voxel_grid(points, voxel_size=0.5, threshold=1, 
                        title="Voxel Grid Visualization", save_path=None, 
                        show=True, figsize=(10, 8)):
    """
    点群データをボクセルグリッドとして可視化する関数
    
    Args:
        points (numpy.ndarray): 点群データ (N, 3)または(N, 4)の形状
        voxel_size (float): ボクセルのサイズ
        threshold (int): ボクセルを描画するための最小点数
        title (str): 図のタイトル
        save_path (str, optional): 図を保存するパス
        show (bool): 図を表示するかどうか
        figsize (tuple): 図のサイズ
        
    Returns:
        matplotlib.figure.Figure: 作成された図のオブジェクト
    """
    # パディングされた点（すべて0の点）をフィルタリング
    if points.shape[1] >= 3:
        non_zero = ~np.all(points[:, :3] == 0, axis=1)
        filtered_points = points[non_zero, :3]  # xyz座標のみ使用
    else:
        filtered_points = points
    
    # ボクセルグリッドの作成
    voxels = {}
    
    for p in filtered_points:
        # 点の座標をボクセルインデックスに変換
        voxel_idx = tuple((p // voxel_size).astype(int))
        
        # 既存のボクセルカウントを更新または新規作成
        if voxel_idx in voxels:
            voxels[voxel_idx] += 1
        else:
            voxels[voxel_idx] = 1
    
    # 閾値以上の点を含むボクセルのみを保持
    voxels = {k: v for k, v in voxels.items() if v >= threshold}
    
    # 可視化
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 各ボクセルの中心座標とカウントを取得
    voxel_coords = np.array([k for k in voxels.keys()])
    voxel_counts = np.array([v for v in voxels.values()])
    
    # 中心座標に変換（ボクセルサイズの半分をオフセット）
    voxel_centers = (voxel_coords + 0.5) * voxel_size
    
    # 点数に基づいて色付け
    scatter = ax.scatter(
        voxel_centers[:, 0],
        voxel_centers[:, 1],
        voxel_centers[:, 2],
        c=voxel_counts,
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    
    plt.colorbar(scatter, ax=ax, label='点数')
    
    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\nボクセルサイズ: {voxel_size}, 閾値: {threshold}')
    
    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def visualize_multi_point_clouds(point_clouds, titles=None, max_points=2000, 
                                ncols=3, save_path=None, show=True, figsize=None):
    """
    複数の点群を並べて可視化する関数
    
    Args:
        point_clouds (list): 点群データのリスト [(N, 3), ...]
        titles (list, optional): 各点群のタイトルのリスト
        max_points (int): 表示する最大点数
        ncols (int): 列数
        save_path (str, optional): 図を保存するパス
        show (bool): 図を表示するかどうか
        figsize (tuple, optional): 図のサイズ
        
    Returns:
        matplotlib.figure.Figure: 作成された図のオブジェクト
    """
    n = len(point_clouds)
    nrows = int(np.ceil(n / ncols))
    
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig = plt.figure(figsize=figsize)
    
    for i, points in enumerate(point_clouds):
        # パディングされた点（すべて0の点）をフィルタリング
        if points.shape[1] >= 3:
            non_zero = ~np.all(points[:, :3] == 0, axis=1)
            filtered_points = points[non_zero]
        else:
            filtered_points = points
        
        # 点数を制限
        if len(filtered_points) > max_points:
            indices = np.random.choice(len(filtered_points), max_points, replace=False)
            filtered_points = filtered_points[indices]
        
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        
        if filtered_points.shape[1] >= 4:  # インテンシティがある場合
            scatter = ax.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                filtered_points[:, 2],
                c=filtered_points[:, 3],  # インテンシティで色付け
                cmap='viridis',
                s=2
            )
        else:
            # Z座標で色付け
            scatter = ax.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                filtered_points[:, 2],
                c=filtered_points[:, 2],
                cmap='viridis',
                s=2
            )
        
        # タイトル設定
        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f'Point Cloud {i+1}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 軸の範囲を均等に
        x_range = filtered_points[:, 0].max() - filtered_points[:, 0].min()
        y_range = filtered_points[:, 1].max() - filtered_points[:, 1].min()
        z_range = filtered_points[:, 2].max() - filtered_points[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2
        
        mid_x = (filtered_points[:, 0].max() + filtered_points[:, 0].min()) / 2
        mid_y = (filtered_points[:, 1].max() + filtered_points[:, 1].min()) / 2
        mid_z = (filtered_points[:, 2].max() + filtered_points[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

# 使用例
if __name__ == "__main__":
    # サンプルデータの生成
    np.random.seed(42)
    n_points = 1000
    
    # 球状の点群を生成
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0.8, 1.0, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # インテンシティを追加
    intensity = np.random.uniform(0, 1, n_points)
    
    # 点群データを作成
    points = np.column_stack((x, y, z, intensity))
    
    # クラスラベル（サンプル）
    labels = (z > 0).astype(int)
    predictions = (z > -0.2).astype(int)  # わざと誤分類を含める
    
    # 各可視化関数のテスト
    print("点群の可視化テスト...")
    visualize_point_cloud(points, title="Sample Point Cloud")
    
    print("点群の予測結果可視化テスト...")
    visualize_point_cloud(points, predictions=predictions, ground_truth=labels, 
                         title="Point Cloud with Predictions")
    
    print("ボクセルグリッド可視化テスト...")
    visualize_voxel_grid(points, voxel_size=0.2, threshold=5, 
                        title="Sample Voxel Grid")
    
    print("混同行列可視化テスト...")
    plot_confusion_matrix(labels, predictions, 
                         class_names=['下部', '上部'], 
                         title="Sample Confusion Matrix")
    
    print("複数点群の可視化テスト...")
    # 2つ目の点群（平面状）を作成
    n_points2 = 800
    x2 = np.random.uniform(-1, 1, n_points2)
    y2 = np.random.uniform(-1, 1, n_points2)
    z2 = 0.1 * np.random.randn(n_points2) + 0.5  # 少しノイズのある平面
    intensity2 = np.sqrt(x2**2 + y2**2)  # 中心からの距離をインテンシティに
    
    points2 = np.column_stack((x2, y2, z2, intensity2))
    
    visualize_multi_point_clouds([points, points2], 
                               titles=["球状点群", "平面状点群"],
                               save_path="_out/visualizations/multi_point_clouds.png")