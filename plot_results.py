import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_CSV = "/afs/ece.cmu.edu/usr/zhuoqili/Private/project/dataset/data/f1_data/processed/final_processed_adjusted.csv"
CENTROIDS_CSV = "/afs/ece.cmu.edu/usr/zhuoqili/Private/project/centroids.csv"
CLUSTERS_CSV = "/afs/ece.cmu.edu/usr/zhuoqili/Private/project/clusters.csv"
OUT_PNG = "/afs/ece.cmu.edu/usr/zhuoqili/Private/project/kmeans_styles_c.png"


def style_name(c, speed_idx, throttle_idx, brake_idx):
    s, t, b = c[speed_idx], c[throttle_idx], c[brake_idx]
    if b > 0.3:
        return "Heavy braking"
    if t > 0.3 and s > 0.3:
        return "Full throttle / high speed"
    if s < -0.3 and t < 0:
        return "Slow corner"
    if t > 0.0 and abs(s) < 0.5:
        return "Mid-speed cruise"
    return "Transition"


def main():
    percentage = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

    dataset = pd.read_csv(DATA_CSV)
    split_len = int(len(dataset) * percentage)
    dataset = dataset[:split_len]
    feature_names = list(dataset.columns)

    centers = np.loadtxt(CENTROIDS_CSV, delimiter=",")
    labels = np.loadtxt(CLUSTERS_CSV, dtype=int)
    if len(labels) != len(dataset):
        print(f"WARN: labels ({len(labels)}) != dataset rows ({len(dataset)}); "
              f"check that <percentage> matches the C run.")
        n = min(len(labels), len(dataset))
        labels = labels[:n]
        dataset = dataset.iloc[:n]

    k = centers.shape[0]
    speed_idx = feature_names.index("Speed")
    throttle_idx = feature_names.index("Throttle")
    brake_idx = feature_names.index("Brake")
    x_idx = feature_names.index("X")
    y_idx = feature_names.index("Y")

    counts = np.bincount(labels, minlength=k)
    pct = counts / counts.sum() * 100

    print("\n--- Driving style distribution (C k-means) ---")
    for i in range(k):
        name = style_name(centers[i], speed_idx, throttle_idx, brake_idx)
        print(f"Cluster {i} [{name}]: {pct[i]:.2f}%  "
              f"(Speed={centers[i, speed_idx]:.2f}, "
              f"Throttle={centers[i, throttle_idx]:.2f}, "
              f"Brake={centers[i, brake_idx]:.2f})")

    xs = dataset.iloc[:, x_idx].values
    ys = dataset.iloc[:, y_idx].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter = ax1.scatter(xs, ys, c=labels, cmap="tab10", s=2)
    ax1.set_title("Track map colored by driving-style cluster (C)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(*scatter.legend_elements(), title="Cluster", loc="best")

    bar_labels = [f"C{i}\n{style_name(centers[i], speed_idx, throttle_idx, brake_idx)}"
                  for i in range(k)]
    ax2.bar(bar_labels, pct, color=[plt.cm.tab10(i) for i in range(k)])
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Driving-style distribution (C)")
    for i, v in enumerate(pct):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved figure to {OUT_PNG}")


if __name__ == "__main__":
    main()
