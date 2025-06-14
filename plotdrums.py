import sys
import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_json_files(basename):
    pattern = f"{basename}*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Keine Dateien gefunden für Muster: {pattern}")
        sys.exit(1)
    dfs = {}
    for f in files:
        dfs[f] = pd.read_json(f, lines=True)
    return files, dfs


def align_data_on_peak(dfs, feature='peak'):
    # Bestimme für jede Datei den Index des maximalen feature-Werts
    peak_idxs = {f: df[feature].idxmax() for f, df in dfs.items()}
    # Verkehrsgrößen für Padding
    max_pre = max(peak_idxs.values())
    max_post = max((len(df) - idx) for df, idx in zip(dfs.values(), peak_idxs.values()))
    total_len = max_pre + max_post

    aligned = {}
    for f, df in dfs.items():
        idx = peak_idxs[f]
        pad_pre = max_pre - idx
        pad_post = total_len - pad_pre - len(df)
        data = {}
        for col in df.columns:
            arr = df[col].to_numpy()
            data[col] = np.pad(arr, (pad_pre, pad_post), 'constant', constant_values=(0, 0))
        aligned[f] = pd.DataFrame(data)
    return aligned


def compute_summary_stats(series_list):
    max_vals = [np.max(s) for s in series_list]
    return {
        'mean': float(np.mean(max_vals)),
        'median': float(np.median(max_vals)),
        'min': float(np.min(max_vals)),
        'max': float(np.max(max_vals)),
        'std': float(np.std(max_vals))
    }


def plot_features(aligned, basename, out_pdf=None):
    features = list(next(iter(aligned.values())).columns)
    if out_pdf is None:
        out_pdf = f"{basename}_analysis_aligned.pdf"

    with PdfPages(out_pdf) as pdf:
        for feat in features:
            plt.figure(figsize=(8, 4))
            for f, df in aligned.items():
                plt.plot(df[feat], label=os.path.basename(f))
            plt.title(f"{feat} (aligned)")
            plt.xlabel("Frames (0 = detected peak)")
            plt.ylabel(feat)
            plt.legend(loc='upper right', fontsize='small')

            # Zusammenfassung der Maximalwerte
            series_list = [df[feat].to_numpy() for df in aligned.values()]
            stats = compute_summary_stats(series_list)
            text = (
                f"Anzahl Samples: {len(series_list)}\n"
                f"Max-Werte: mean={stats['mean']:.4f}, med={stats['median']:.4f},\n"
                f"          min={stats['min']:.4f}, max={stats['max']:.4f}, std={stats['std']:.4f}"
            )
            plt.gcf().text(0.02, 0.95, text, va='top', ha='left', fontsize='8', bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.5"))

            pdf.savefig()
            plt.close()

    print(f"Analyse-PDF gespeichert als: {out_pdf}")


def main():
    if len(sys.argv) < 2:
        print("Aufruf: python plotdrums.py <basename>")
        print("Beispiel: python plotdrums.py kick")
        sys.exit(1)
    basename = sys.argv[1]
    files, dfs = load_json_files(basename)
    print(f"Analysiere Dateien: {', '.join(os.path.basename(f) for f in files)}")
    aligned = align_data_on_peak(dfs)
    plot_features(aligned, basename)


if __name__ == '__main__':
    main()
