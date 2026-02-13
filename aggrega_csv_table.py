import pandas as pd
import glob
import os

# Cartella dove stanno i CSV
folder_path = input("Inserisci il percorso della cartella con i CSV: ")

files = glob.glob(os.path.join(folder_path, "*.csv"))

dfs = []

for i, f in enumerate(files):
    df = pd.read_csv(f)

    # ID del run
    df["run_id"] = i + 1

    dfs.append(df)

# Unisce tutti i csv
data = pd.concat(dfs, ignore_index=True)

# Colonne parametriche (NO thresholds, NO Rank)
group_cols = [
    "end_inertia",
    "global_weight",
    "inertia",
    "local_weight",
    "population",
    "topology",
    # "velocity_clamp"
]

# ---------- RANK ----------
rank_wide = (
    data
    .set_index(group_cols + ["run_id"])["Rank"]
    .unstack("run_id")          # non perde NaN
)

rank_wide.columns = [f"Rank_{i}" for i in rank_wide.columns]

# ---------- THRESHOLDS ----------
thr_wide = (
    data
    .set_index(group_cols + ["run_id"])["thresholds"]
    .unstack("run_id")
)

thr_wide.columns = [f"Threshold_{i}" for i in thr_wide.columns]

# ---------- MERGE ----------
final_df = pd.concat([rank_wide, thr_wide], axis=1)

# ---------- MEAN ----------
final_df["Rank_mean"] = rank_wide.mean(axis=1).round(2)

# Reset index
final_df = final_df.reset_index()

# Ordinamento
final_df = final_df.sort_values("Rank_mean")
rank_cols = [c for c in final_df if c.startswith("Rank_")]
thr_cols  = [c for c in final_df if c.startswith("Threshold_")]

ordered = []
for r, t in zip(rank_cols, thr_cols):
    ordered += [r, t]

final_df = final_df[group_cols + ordered + ["Rank_mean"]]
final_df = final_df.reset_index(drop=True)
final_df = final_df.drop(columns=["Rank_mean"])  # se non serve in tabella

# Salva
final_df.to_csv("aggregated_results.csv", index=False)

print("CSV finale creato: aggregated_results.csv")

latex_table = final_df.to_latex(
    index=False,
    float_format="%.2f",   # arrotonda float
    na_rep="None",          # come mostrare i NaN
    caption="Risultati aggregati",
    label="tab:results",
    longtable=False,        # True se tabella lunga
)
print('\nLaTeX Table:\n')
print(latex_table)