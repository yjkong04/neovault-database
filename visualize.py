import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

SQL_FILE = "neovault-db.sql"
RESULTS_DIR = "results"

# ── Parse INSERT statements ──────────────────────────────────────────────────

def parse_insert(sql_text, table):
    pattern = rf"INSERT INTO `{table}` VALUES (.*?);"
    match = re.search(pattern, sql_text, re.DOTALL)
    if not match:
        return []
    rows_str = match.group(1)
    rows = re.findall(r"\(([^)]+)\)", rows_str)
    result = []
    for row in rows:
        vals = []
        for v in row.split(","):
            v = v.strip().strip("'")
            try:
                vals.append(float(v) if "." in v else int(v))
            except ValueError:
                vals.append(v)
        result.append(vals)
    return result

with open(SQL_FILE, "r") as f:
    sql = f.read()

# ── Build DataFrames ─────────────────────────────────────────────────────────

neonate_rows = parse_insert(sql, "neonate")
df_n = pd.DataFrame(neonate_rows, columns=[
    "id", "age", "size", "headsize", "weight",
    "apgar1", "apgar5", "crib", "neurologicalrisk", "gender",
    "registrationdate", "latestupdate"
])
df_n.replace(-1, float("nan"), inplace=True)
df_n["id"] = df_n["id"].astype(int)

physio_rows = parse_insert(sql, "physiologicalinformation")
df_p = pd.DataFrame(physio_rows, columns=[
    "id", "idneonate", "timestamp", "heartrate", "saturation"
])
df_p.replace(-1, float("nan"), inplace=True)
df_p["idneonate"] = df_p["idneonate"].astype(int)
df_p["time"] = pd.to_datetime(df_p["timestamp"], unit="s")

pose_rows = parse_insert(sql, "pose")
df_pose = pd.DataFrame(pose_rows, columns=[
    "id", "idneonate", "starttime", "endtime", "path_file", "path_file_gif"
])
df_pose["idneonate"] = df_pose["idneonate"].astype(int)
df_pose["duration_min"] = (df_pose["endtime"] - df_pose["starttime"]) / 60

COLORS = plt.cm.tab10.colors

# ── Figure 1: Neonate Overview ───────────────────────────────────────────────

fig1, axes = plt.subplots(2, 3, figsize=(16, 9))
fig1.suptitle("Neonate Overview", fontsize=16, fontweight="bold")

ids = df_n["id"].astype(str)
bar_kw = dict(color=COLORS[:len(df_n)], edgecolor="white", linewidth=0.5)

# Weight
axes[0, 0].bar(ids, df_n["weight"], **bar_kw)
axes[0, 0].set_title("Birth Weight (g)")
axes[0, 0].set_xlabel("Neonate ID")
axes[0, 0].axhline(df_n["weight"].mean(), color="red", linestyle="--", linewidth=1, label=f"Mean {df_n['weight'].mean():.0f}g")
axes[0, 0].legend(fontsize=8)

# Size
axes[0, 1].bar(ids, df_n["size"], **bar_kw)
axes[0, 1].set_title("Body Size (cm)")
axes[0, 1].set_xlabel("Neonate ID")

# Head size
df_head = df_n[df_n["headsize"].notna()].reset_index(drop=True)
head_ids = df_head["id"].astype(str)
axes[0, 2].bar(head_ids, df_head["headsize"],
               color=[COLORS[i % 10] for i in range(len(df_head))], edgecolor="white")
axes[0, 2].set_title("Head Circumference (cm)")
axes[0, 2].set_xlabel("Neonate ID")

# Apgar scores
x = range(len(df_n))
axes[1, 0].bar([i - 0.2 for i in x], df_n["apgar1"], width=0.4, label="1 min", color="#4C72B0")
axes[1, 0].bar([i + 0.2 for i in x], df_n["apgar5"], width=0.4, label="5 min", color="#DD8452")
axes[1, 0].set_title("Apgar Scores")
axes[1, 0].set_xticks(list(x))
axes[1, 0].set_xticklabels(ids)
axes[1, 0].set_xlabel("Neonate ID")
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 11)

# Gestational age vs weight scatter
sc = axes[1, 1].scatter(df_n["age"], df_n["weight"], c=df_n["id"], cmap="tab10", s=100, zorder=3)
for _, row in df_n.iterrows():
    axes[1, 1].annotate(f" {int(row['id'])}", (row["age"], row["weight"]), fontsize=8)
axes[1, 1].set_title("Gestational Age vs Birth Weight")
axes[1, 1].set_xlabel("Gestational Age (weeks)")
axes[1, 1].set_ylabel("Weight (g)")
axes[1, 1].grid(True, alpha=0.3)

# Neurological risk
risk_counts = df_n["neurologicalrisk"].value_counts().sort_index()
labels = {0: "No risk", 1: "At risk"}
axes[1, 2].pie(
    risk_counts.values,
    labels=[labels.get(int(k), str(k)) for k in risk_counts.index],
    autopct="%1.0f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90
)
axes[1, 2].set_title("Neurological Risk")

plt.tight_layout()
fig1.savefig(f"{RESULTS_DIR}/neonate_overview.png", dpi=150, bbox_inches="tight")
print("Saved neonate_overview.png")

# ── Figure 2: Vitals Over Time ───────────────────────────────────────────────

neonates = sorted(df_p["idneonate"].unique())
n = len(neonates)
cols = 3
rows = (n + cols - 1) // cols

fig2, axes2 = plt.subplots(rows * 2, cols, figsize=(18, rows * 5))
fig2.suptitle("Vitals Over Time per Neonate", fontsize=16, fontweight="bold")

for i, nid in enumerate(neonates):
    row_hr = (i // cols) * 2
    row_sp = row_hr + 1
    col = i % cols

    df_sub = df_p[df_p["idneonate"] == nid].sort_values("time")
    t = df_sub["time"]

    ax_hr = axes2[row_hr, col]
    ax_hr.plot(t, df_sub["heartrate"], color="#E53935", linewidth=0.8)
    ax_hr.set_title(f"Neonate {nid} — Heart Rate (bpm)", fontsize=9)
    ax_hr.tick_params(axis="x", labelsize=6)
    ax_hr.grid(True, alpha=0.3)

    ax_sp = axes2[row_sp, col]
    ax_sp.plot(t, df_sub["saturation"], color="#1E88E5", linewidth=0.8)
    ax_sp.set_title(f"Neonate {nid} — SpO₂ (%)", fontsize=9)
    ax_sp.tick_params(axis="x", labelsize=6)
    ax_sp.grid(True, alpha=0.3)
    ax_sp.set_ylim(80, 102)

# Hide unused subplots
for j in range(i + 1, rows * cols):
    axes2[(j // cols) * 2, j % cols].set_visible(False)
    axes2[(j // cols) * 2 + 1, j % cols].set_visible(False)

plt.tight_layout()
fig2.savefig(f"{RESULTS_DIR}/vitals_over_time.png", dpi=150, bbox_inches="tight")
print("Saved vitals_over_time.png")

# ── Figure 3: Pose Durations ─────────────────────────────────────────────────

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Pose Durations per Neonate", fontsize=16, fontweight="bold")

# Total pose time per neonate
total = df_pose.groupby("idneonate")["duration_min"].sum().sort_index()
ax1.bar(total.index.astype(str), total.values,
        color=[COLORS[i % 10] for i in range(len(total))], edgecolor="white")
ax1.set_title("Total Recorded Pose Time (min)")
ax1.set_xlabel("Neonate ID")
ax1.set_ylabel("Minutes")
ax1.grid(True, axis="y", alpha=0.3)

# Distribution of individual pose segment durations
for nid in sorted(df_pose["idneonate"].unique()):
    durations = df_pose[df_pose["idneonate"] == nid]["duration_min"]
    ax2.plot(sorted(durations.values), label=f"Neonate {nid}", linewidth=1.2)
ax2.set_title("Pose Segment Duration Distribution (sorted)")
ax2.set_xlabel("Segment rank")
ax2.set_ylabel("Duration (min)")
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(f"{RESULTS_DIR}/pose_durations.png", dpi=150, bbox_inches="tight")
print("Saved pose_durations.png")

plt.show()
print("\nDone. Three PNG files saved in the project directory.")
