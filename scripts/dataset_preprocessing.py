import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_lidar_data(data_dir, max_points=None):
    """指定ディレクトリからLiDARデータを読み込む"""
    print(f"Loading data from {data_dir}")
    
    # 全ての点群ファイルを取得
    lidar_files = sorted(glob.glob(os.path.join(data_dir, 'lidar_points_*.npy')))
    transform_files = sorted(glob.glob(os.path.join(data_dir, 'transform_*.txt')))
    
    print(f"Found {len(lidar_files)} point cloud files")
    print(f"Found {len(transform_files)} transform files")
    
    # マッチングするファイルペアのみを使用
    valid_pairs = []
    for lidar_file in lidar_files:
        # LiDARファイル名から対応する変換ファイル名を生成
        base_name = os.path.basename(lidar_file)
        frame_id = base_name.replace('lidar_points_', '').replace('.npy', '')
        transform_file = os.path.join(data_dir, f'transform_{frame_id}.txt')
        
        if os.path.exists(transform_file):
            valid_pairs.append((lidar_file, transform_file))
    
    print(f"Found {len(valid_pairs)} matching file pairs")
    
    # サンプリング（全データを使うとメモリ不足になる場合）
    sample_rate = 5  # 例: 5フレームごとに1フレーム使用
    valid_pairs = valid_pairs[::sample_rate]
    
    print(f"Using {len(valid_pairs)} file pairs after sampling")
    
    # まずは点群のサイズを調査
    if not valid_pairs:
        raise ValueError("No valid data pairs found.")
    
    # サンプルファイルを読み込んでサイズを確認
    sample_points = np.load(valid_pairs[0][0])
    point_dim = sample_points.shape[1]  # 点の次元数（通常は4: x, y, z, intensity）
    
    # 最大点数を決定（指定がない場合はサンプルの点数を使用）
    if max_points is None:
        # 最初の10ファイルの平均点数を使用
        point_counts = []
        for i in range(min(10, len(valid_pairs))):
            points = np.load(valid_pairs[i][0])
            point_counts.append(points.shape[0])
        max_points = int(np.mean(point_counts))
    
    print(f"Using maximum {max_points} points per cloud with {point_dim} dimensions")
    
    # 固定サイズの配列を初期化
    point_clouds = np.zeros((len(valid_pairs), max_points, point_dim), dtype=np.float32)
    transforms = np.zeros((len(valid_pairs), 6), dtype=np.float32)  # x, y, z, pitch, yaw, roll
    
    # 有効なペアのみ処理
    for i, (lidar_file, transform_file) in enumerate(valid_pairs):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(valid_pairs)}")
        
        try:
            # 点群データ読み込み
            points = np.load(lidar_file)
            
            # データの前処理（例：点群を中心化）
            points_xyz = points[:, :3]
            center = np.mean(points_xyz, axis=0)
            points_centered = points.copy()
            points_centered[:, :3] = points_xyz - center
            
            # 点数の調整（多すぎる場合はランダムサンプリング、少なすぎる場合はパディング）
            num_points = points_centered.shape[0]
            
            if num_points > max_points:
                # ランダムに点を選択
                indices = np.random.choice(num_points, max_points, replace=False)
                points_centered = points_centered[indices]
                point_clouds[i, :, :] = points_centered
            else:
                # 足りない部分は0埋め
                point_clouds[i, :num_points, :] = points_centered
            
            # 変換情報の読み込み（テキストファイルから）
            with open(transform_file, 'r') as f:
                lines = f.readlines()
                # 位置情報を抽出
                loc_line = lines[1]
                x = float(loc_line.split('X=')[1].split(',')[0])
                y = float(loc_line.split('Y=')[1].split(',')[0])
                z = float(loc_line.split('Z=')[1].split(',')[0])
                
                # 回転情報を抽出
                rot_line = lines[2]
                pitch = float(rot_line.split('Pitch=')[1].split(',')[0])
                yaw = float(rot_line.split('Yaw=')[1].split(',')[0])
                roll = float(rot_line.split('Roll=')[1].split(',')[0])
                
                transforms[i] = np.array([x, y, z, pitch, yaw, roll])
            
        except Exception as e:
            print(f"Error processing files {lidar_file} and {transform_file}: {e}")
            # エラーが発生した場合は0で埋める（または他の対処方法）
            point_clouds[i] = np.zeros((max_points, point_dim), dtype=np.float32)
            transforms[i] = np.zeros(6, dtype=np.float32)
    
    return point_clouds, transforms

def create_voxel_grid(points, grid_size=32, voxel_size=1.0):
    """点群からボクセルグリッドを作成"""
    # 点群座標を均一なグリッドにマッピング
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # 各点をグリッドにマッピング（点が0でない場合のみ）
    for point in points:
        if np.all(point[:3] == 0):  # パディングされた0の点はスキップ
            continue
            
        x, y, z = point[:3]  # インテンシティを含む場合は最初の3次元のみ使用
        
        # スケーリングと中心化
        x_scaled = int((x + grid_size * voxel_size / 2) / voxel_size)
        y_scaled = int((y + grid_size * voxel_size / 2) / voxel_size)
        z_scaled = int((z + grid_size * voxel_size / 2) / voxel_size)
        
        # グリッド内に収まる点のみ処理
        if (0 <= x_scaled < grid_size and 
            0 <= y_scaled < grid_size and 
            0 <= z_scaled < grid_size):
            # 占有度または密度を表現
            voxel_grid[x_scaled, y_scaled, z_scaled] += 1
    
    # 正規化
    if np.max(voxel_grid) > 0:
        voxel_grid = voxel_grid / np.max(voxel_grid)
    
    return voxel_grid

def prepare_dataset(data_dir, output_dir, task='classification'):
    """データセットを準備し保存"""
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # データの読み込み（固定サイズの配列を使用）
        point_clouds, transforms = load_lidar_data(data_dir, max_points=2000)
        
        print(f"Loaded {len(point_clouds)} data samples with shape {point_clouds.shape}")
        
        if len(point_clouds) == 0:
            print("No data was loaded. Check your data directory.")
            return
        
        # データセットの目的に基づいてラベルを作成
        if task == 'classification':
            # 例：位置に基づいて分類（道路上、建物の近く、など）
            labels = np.zeros(len(transforms), dtype=np.int32)
            for i, transform in enumerate(transforms):
                x, y = transform[0], transform[1]
                # 例：単純な2クラス分類（X座標が正か負か）
                labels[i] = 1 if x > 0 else 0
            
            print(f"Created classification labels with distribution: {np.bincount(labels)}")
        
        elif task == 'regression':
            # 例：速度の推定
            labels = np.sqrt(transforms[:, 0]**2 + transforms[:, 1]**2)  # 簡易的な速度計算
            print(f"Created regression labels with range: [{labels.min():.2f}, {labels.max():.2f}]")
        
        elif task == 'segmentation':
            # 点群セグメンテーションはより複雑（ここでは詳細を省略）
            labels = np.zeros((len(point_clouds), 32, 32, 32), dtype=np.float32)
            print(f"Created segmentation labels with shape: {labels.shape}")
        
        # データセットの分割（訓練・検証・テスト）
        X_train, X_test, y_train, y_test = train_test_split(
            point_clouds, labels, test_size=0.2, random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # サンプルのボクセルグリッド変換を表示
        if len(X_train) > 0:
            # 最初のサンプルのボクセルグリッド変換を表示
            try:
                sample_voxel = create_voxel_grid(X_train[0])
                plt.figure(figsize=(8, 8))
                plt.imshow(np.max(sample_voxel, axis=0))
                plt.title('Voxel Grid Projection (Top View)')
                plt.colorbar()
                plt.savefig(os.path.join(output_dir, 'voxel_sample.png'))
                plt.close()
                
                # オリジナル点群の可視化も追加
                plt.figure(figsize=(10, 10))
                non_zero_points = X_train[0][~np.all(X_train[0][:, :3] == 0, axis=1)]
                plt.scatter(non_zero_points[:, 0], non_zero_points[:, 1], s=0.5, c=non_zero_points[:, 2], cmap='viridis')
                plt.axis('equal')
                plt.title('Sample Point Cloud (Top View)')
                plt.colorbar(label='Height (Z)')
                plt.savefig(os.path.join(output_dir, 'point_cloud_sample.png'))
                plt.close()
            except Exception as e:
                print(f"Error creating visualization: {e}")
        
        # データセットの保存
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # データセット情報の保存
        with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
            f.write(f"Source directory: {data_dir}\n")
            f.write(f"Task: {task}\n")
            f.write(f"Total samples: {len(point_clouds)}\n")
            f.write(f"Point cloud shape: {point_clouds.shape}\n")
            f.write(f"Train samples: {len(X_train)}\n")
            f.write(f"Validation samples: {len(X_val)}\n")
            f.write(f"Test samples: {len(X_test)}\n")
            
            if task == 'classification':
                f.write(f"Label distribution: {np.bincount(labels)}\n")
            elif task == 'regression':
                f.write(f"Label range: [{labels.min():.2f}, {labels.max():.2f}]\n")
        
        print(f"Dataset prepared and saved to {output_dir}")
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 最新のデータディレクトリを自動検出
    base_dir = "_out/lidar_data"
    dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if dirs:
        data_dir = max(dirs, key=os.path.getctime)  # 最新のディレクトリ
        print(f"Using most recent data directory: {data_dir}")
        prepare_dataset(data_dir, '_out/lidar_dataset', task='classification')
    else:
        print(f"No data directories found in {base_dir}")