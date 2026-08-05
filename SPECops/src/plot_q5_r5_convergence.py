import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#simple L2-vs-epoch line plot of the full q5_r5 Adam convergence trace (epochs 100-1000), same single-line style as frequency_unit_conversion.py's plotScaling
def run(csvPath=None, plotPath=None):
    csvPath = csvPath or os.path.join(RESULTS_DIR, "q5_r5_adam_convergence.csv")
    plotPath = plotPath or os.path.join(RESULTS_DIR, "q5_r5_adam_convergence.png")

    epochs, l2Errors = [], []
    with open(csvPath, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            l2Errors.append(float(row["l2_error"]))

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, l2Errors, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("L2 error")
    plt.title("q5_r5 Adam convergence: L2 error vs epoch")
    plt.tight_layout()
    plt.savefig(plotPath)
    plt.close()
    print(f"saved {plotPath}", flush=True)
    return plotPath

if __name__ == "__main__":
    run()
