#!/usr/bin/env python3
"""
Script to analyze correlation between parquet data and video files in LeRobot datasets.
This demonstrates how the data is synchronized and correlated.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_data_video_correlation(dataset_path):
    """Analyze how parquet data correlates with video files."""
    
    print(f"🔍 Analyzing correlation between data and videos in: {dataset_path}")
    
    # Load metadata
    meta_path = Path(dataset_path) / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"\n📋 Dataset Metadata:")
        print(f"  Robot type: {meta.get('robot_type', 'Unknown')}")
        print(f"  Total episodes: {meta.get('total_episodes', 'Unknown')}")
        print(f"  Total frames: {meta.get('total_frames', 'Unknown')}")
        print(f"  FPS: {meta.get('fps', 'Unknown')}")
    
    # Load parquet data
    parquet_file = Path(dataset_path) / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_file.exists():
        print(f"❌ Parquet file not found: {parquet_file}")
        return
    
    df = pd.read_parquet(parquet_file)
    print(f"\n📊 Parquet Data Overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Find key columns
    timestamp_col = None
    frame_col = None
    episode_col = None
    
    for col in df.columns:
        if 'timestamp' in col.lower():
            timestamp_col = col
        elif 'frame' in col.lower():
            frame_col = col
        elif 'episode' in col.lower():
            episode_col = col
    
    print(f"\n🔗 Key Columns Found:")
    print(f"  Timestamp: {timestamp_col}")
    print(f"  Frame: {frame_col}")
    print(f"  Episode: {episode_col}")
    
    # Analyze timing
    if timestamp_col:
        print(f"\n⏱️ Timing Analysis:")
        print(f"  Start time: {df[timestamp_col].min():.3f}s")
        print(f"  End time: {df[timestamp_col].max():.3f}s")
        print(f"  Duration: {df[timestamp_col].max() - df[timestamp_col].min():.3f}s")
        print(f"  Total frames: {len(df)}")
        print(f"  Calculated FPS: {len(df) / (df[timestamp_col].max() - df[timestamp_col].min()):.1f}")
    
    # Check video files
    video_dir = Path(dataset_path) / "videos"
    if video_dir.exists():
        print(f"\n🎥 Video Files:")
        for video_subdir in video_dir.iterdir():
            if video_subdir.is_dir():
                print(f"  Camera: {video_subdir.name}")
                for video_file in video_subdir.rglob("*.mp4"):
                    print(f"    File: {video_file.name}")
                    print(f"    Size: {video_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Demonstrate correlation
    print(f"\n🔗 Data-Video Correlation:")
    print(f"  Each row in the parquet file corresponds to:")
    print(f"    - One frame in the video files")
    print(f"    - One timestamp in the robot's timeline")
    print(f"    - One set of robot state/action data")
    
    if timestamp_col and frame_col:
        print(f"\n📈 Frame-Timestamp Relationship:")
        print(f"  Frame 0: {df[timestamp_col].iloc[0]:.3f}s")
        print(f"  Frame 100: {df[timestamp_col].iloc[100]:.3f}s")
        print(f"  Frame 500: {df[timestamp_col].iloc[500]:.3f}s")
        
        # Calculate frame intervals
        time_diffs = df[timestamp_col].diff().dropna()
        print(f"  Average frame interval: {time_diffs.mean():.3f}s")
        print(f"  Expected interval (30 FPS): {1/30:.3f}s")
    
    # Show sample data
    print(f"\n📋 Sample Data (first 3 rows):")
    if timestamp_col and frame_col:
        sample_cols = [timestamp_col, frame_col]
        if episode_col:
            sample_cols.append(episode_col)
        
        # Add robot state/action columns
        for col in df.columns:
            if 'state' in col or 'action' in col:
                sample_cols.append(col)
                if len(sample_cols) >= 8:  # Limit columns for readability
                    break
        
        print(df[sample_cols].head(3).to_string())

def demonstrate_synchronization(dataset_path):
    """Demonstrate how to access synchronized data and video."""
    
    print(f"\n🎯 Synchronization Example:")
    print(f"To access frame 100 of the dataset:")
    print(f"  1. Load parquet data: df.iloc[100]")
    print(f"  2. Get timestamp: df.iloc[100]['timestamp']")
    print(f"  3. Extract video frame at that timestamp")
    print(f"  4. Both represent the same moment in time")
    
    # Load data
    parquet_file = Path(dataset_path) / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(parquet_file)
    
    # Show example
    if len(df) > 100:
        frame_100 = df.iloc[100]
        print(f"\n📊 Frame 100 Data:")
        for col in frame_100.index:
            if 'timestamp' in col.lower() or 'frame' in col.lower() or 'state' in col or 'action' in col:
                print(f"  {col}: {frame_100[col]}")

def main():
    """Main function to analyze correlation."""
    
    dataset_path = "svla_so101_pickplace"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    analyze_data_video_correlation(dataset_path)
    demonstrate_synchronization(dataset_path)
    
    print(f"\n✅ Correlation analysis complete!")
    print(f"\n💡 Key Points:")
    print(f"  • Parquet files contain robot state/action data")
    print(f"  • Video files contain visual observations")
    print(f"  • Both are synchronized by timestamps")
    print(f"  • Each row in parquet = one frame in video")
    print(f"  • Frame rate is typically 30 FPS")

if __name__ == "__main__":
    main() 