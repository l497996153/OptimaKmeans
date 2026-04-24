import matplotlib.pyplot as plt
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

TOTAL_POINTS = 3800046
METRIC = "time per iterations"

df_base    = pd.read_csv(os.path.join(current_dir, "results_gpu_base.csv"))
df_warp    = pd.read_csv(os.path.join(current_dir, "results_gpu_warp.csv"))
df_warpmem = pd.read_csv(os.path.join(current_dir, "results_gpu_warpmem.csv"))
df_warpmem_found = pd.read_csv(os.path.join(current_dir, "results_gpu_warpmemfound.csv"))
df_py      = pd.read_csv("/afs/ece.cmu.edu/usr/zhuoqili/Private/project/external_baseline/results_py.csv")

PY_METRIC = "time_per_iter"

plt.figure(figsize=(8, 5))
plt.plot(df_base["percentage"] * TOTAL_POINTS, df_base[METRIC],
         marker="o", color="tab:blue",   label="base GPU")
plt.plot(df_warp["percentage"] * TOTAL_POINTS, df_warp[METRIC],
         marker="o", color="tab:orange", label="warp-level reduction")
plt.plot(df_warpmem["percentage"] * TOTAL_POINTS, df_warpmem[METRIC],
         marker="o", color="tab:green",  label="warp + shared memory")
plt.plot(df_warpmem_found["percentage"] * TOTAL_POINTS, df_warpmem_found[METRIC],
         marker="o", color="tab:pink",  label="warp + shared memory + found")
plt.plot(df_py["percentage"] * TOTAL_POINTS, df_py[PY_METRIC],
         marker="o", color="tab:red",    label="sklearn (Python)", linestyle="--")

plt.xlabel("Number of Data Points")
plt.ylabel("Time per Iteration (ms)")
plt.title("GPU KMeans Strategies: Time per Iteration vs Data Size")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "time_plot_gpu.png"), dpi=150)
print(f"Saved: {os.path.join(current_dir, 'time_plot_gpu.png')}")
