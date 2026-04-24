import matplotlib.pyplot as plt
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

TOTAL_POINTS = 3800046
METRIC = "time per iterations"  # matches header in results_v*.csv

df_v1 = pd.read_csv(os.path.join(current_dir, "results_v1.csv"))
df_v2 = pd.read_csv(os.path.join(current_dir, "results_v2.csv"))
df_v3 = pd.read_csv(os.path.join(current_dir, "results_v3.csv"))
df_py = pd.read_csv(os.path.join(current_dir, "external_baseline/results_py.csv"))

# Python CSV uses a different header name for the time-per-iteration column
PY_METRIC = "time_per_iter"

plt.figure(figsize=(8, 5))
plt.plot(df_v1["percentage"] * TOTAL_POINTS, df_v1[METRIC],
         marker="o", color="tab:blue",   label="local buffers")
plt.plot(df_v2["percentage"] * TOTAL_POINTS, df_v2[METRIC],
         marker="o", color="tab:orange", label="parallel with atomic")
plt.plot(df_v3["percentage"] * TOTAL_POINTS, df_v3[METRIC],
         marker="o", color="tab:green",  label="parallel with reduction")
plt.plot(df_py["percentage"] * TOTAL_POINTS, df_py[PY_METRIC],
         marker="o", color="tab:red",    label="sklearn (Python)", linestyle="--")

plt.xlabel("Number of Data Points")
plt.ylabel("Time per Iteration (ms)")
plt.title("KMeans Update-Step Strategies: Time per Iteration vs Data Size")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "time_plot_3V.png"), dpi=150)
print(f"Saved: {os.path.join(current_dir, 'time_plot_3V.png')}")
