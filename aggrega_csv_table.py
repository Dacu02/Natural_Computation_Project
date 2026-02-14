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

# ---------- CONTEGGIO APPARIZIONI ----------
final_df["Appearances"] = rank_wide.count(axis=1)

# ---------- MEAN (solo sui presenti) ----------
final_df["Rank_mean"] = rank_wide.mean(axis=1, skipna=True).round(2)

# Reset index
final_df = final_df.reset_index()

# ---------- ORDINAMENTO ----------
final_df = final_df.sort_values(
    by=["Appearances", "Rank_mean"],
    ascending=[False, True]   # più apparizioni prima, poi rank migliore
)

# ---------- RIORDINO COLONNE ----------
rank_cols = [c for c in final_df if c.startswith("Rank_")]
thr_cols  = [c for c in final_df if c.startswith("Threshold_")]

ordered = []
for r, t in zip(rank_cols, thr_cols):
    ordered += [r, t]

final_df = final_df[
    group_cols +
    # ["Appearances"] +
    ordered 
    # ["Rank_mean"]+
]

final_df = final_df.reset_index(drop=True)

# ---------- SALVA CSV ----------
final_df.to_csv("aggregated_results.csv", index=False)

print("CSV finale creato: aggregated_results.csv")

# ---------- LATEX ----------
latex_table = final_df.to_latex(
    index=False,
    float_format="%.2f",
    na_rep="---",
    caption="Risultati aggregati",
    label="tab:results",
    longtable=False,
)

print('\nLaTeX Table:\n')
print(latex_table)
