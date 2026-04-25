import time
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def main():
    
    if len(sys.argv) < 2:
        print("Usage: python kmeans_base.py <percentage>")
        sys.exit(1)

    percentage = float(sys.argv[1])

    # Load the dataset (header row consumed by read_csv; drop the first 2 ID columns)
    dataset = pd.read_csv("/afs/ece.cmu.edu/usr/zhuoqili/Private/OptimaKmeans/dataset/data/f1_data/processed/final_processed.csv")
    dataset = dataset.iloc[:, 2:]

    # Split based on percentage
    split_len = int(len(dataset) * percentage)
    dataset = dataset[:split_len]

    print(f"Dataset: {dataset.shape[0]} points, {dataset.shape[1]} dimensions")

    init_centroids = dataset[:5].copy()

    kmeans = KMeans(n_clusters=5, max_iter=500, n_init=1, tol=0, init=init_centroids, algorithm='lloyd')

    start = time.time()
    kmeans.fit(dataset)
    end = time.time()
    duration_ms = (end - start) * 1000
    duration_ms_per_iter = duration_ms / kmeans.n_iter_
    print(f"total time: {duration_ms:.2f} ms")
    print(f"iterations: {kmeans.n_iter_}")
    print(f"time per iteration: {duration_ms_per_iter:.2f} ms")
    print(f"inertia: {kmeans.inertia_}")

if __name__ == "__main__":
    main()
