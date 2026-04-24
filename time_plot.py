import matplotlib.pyplot as plt
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "results.csv"))
df_cpp = pd.read_csv(os.path.join(current_dir, "external_baseline/results_cpp.csv"))
df_py = pd.read_csv(os.path.join(current_dir, "external_baseline/results_py.csv"))

plt.plot(df["percentage"] * 1174691, df["time"], marker="o", color="blue", label="OptimaKmeans")
plt.plot(df_cpp["percentage"] * 1174691, df_cpp["time"], marker="o", color="red", label="mlpack (C++)")
plt.plot(df_py["percentage"] * 1174691, df_py["time"], marker="o", color="green", label="sklearn (Python)")

plt.xlabel("Number of Data Points")
plt.ylabel("Time (ms)")
plt.title("KMeans Execution Time vs Data Size")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(current_dir, "time_plot.png"))
