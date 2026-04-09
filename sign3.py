import pandas as pd

df = pd.read_csv("result_500/summary_0.5/results_3D/patient_metrics_3d_mean_std_by_seed_rounded.csv")

def format_sig(x):
    try:
        return float(f"{float(x):.3g}")
    except (ValueError, TypeError):
        return x

df = df.applymap(format_sig)

df.to_csv("result_500/summary_0.5/results_3D/patient_metrics_3d_mean_std_by_seed_rounded.csv", index=False)