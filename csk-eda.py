# csk_eda.py
# End-to-end EDA for Chennai Super Kings (deliveries-level data)
# ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   # safe to import even if you prefer pure-matplotlib
from pathlib import Path

# ----------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------
DATA_PATH = "D:/Csk data analysis/extracted data/csk_deliveries.csv"   # adjust if needed
df = pd.read_csv(DATA_PATH)

# Ensure numeric where required
numeric_cols = [
    "runs_off_bat", "extras", "wides", "noballs",
    "byes", "legbyes", "penalty"
]
df[numeric_cols] = df[numeric_cols].fillna(0).astype(int)

# ----------------------------------------------------------------
# 2. Helper columns
# ----------------------------------------------------------------
df["total_runs"] = df["runs_off_bat"] + df["extras"]
df["dismissal"]   = df["wicket_type"].notna()

# ----------------------------------------------------------------
# 3. Batting & bowling vs opponent teams
# ----------------------------------------------------------------
batting_vs_opp  = (df[df["batting_team"] == "Chennai Super Kings"]
                   .groupby("bowling_team")["total_runs"].sum()
                   .sort_values(ascending=False))

bowling_vs_opp  = (df[df["bowling_team"] == "Chennai Super Kings"]
                   .groupby("batting_team")["total_runs"].sum()
                   .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(batting_vs_opp.index, batting_vs_opp.values, label="Runs scored")
ax.bar(bowling_vs_opp.index, bowling_vs_opp.values, bottom=batting_vs_opp.reindex(bowling_vs_opp.index, fill_value=0),
       label="Runs conceded", alpha=0.6)
ax.set_title("CSK batting & bowling vs opponent teams")
ax.set_ylabel("Runs")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("CSK_batting_and_bowling_vs_opponent_teams.png")
plt.close()

# ----------------------------------------------------------------
# 4. Total runs by season
# ----------------------------------------------------------------
runs_by_season = (df[df["batting_team"] == "Chennai Super Kings"]
                  .groupby("season")["total_runs"].sum())
runs_by_season.plot(kind="bar", figsize=(10,5), title="CSK total runs by season")
plt.ylabel("Runs")
plt.tight_layout()
plt.savefig("CSK_total_runs_by_season.png")
plt.close()

# ----------------------------------------------------------------
# 5. Year-wise total wickets
# ----------------------------------------------------------------
wickets_by_season = (df[(df["bowling_team"] == "Chennai Super Kings") & df["dismissal"]]
                     .groupby("season")["dismissal"].count())
wickets_by_season.plot(kind="bar", figsize=(10,5), color="grey",
                       title="CSK wickets taken by season")
plt.ylabel("Wickets")
plt.tight_layout()
plt.savefig("CSK_year_wise_total_wickets.png")
plt.close()

# ----------------------------------------------------------------
# 6. Top-10 batting partnerships
# ----------------------------------------------------------------
# Sort the two names alphabetically so (Dhoni,Raina) == (Raina,Dhoni)
pair = df["striker"].where(df["striker"] < df["non_striker"],
                           df["non_striker"] + " & " + df["striker"])
pair = pair.where(df["striker"] < df["non_striker"],
                  df["striker"] + " & " + df["non_striker"])
partnerships = (df[df["batting_team"] == "Chennai Super Kings"]
                .assign(pair=pair)
                .groupby("pair")["runs_off_bat"].sum()
                .nlargest(10))
partnerships.plot(kind="barh", figsize=(8,6), title="Top 10 CSK Batting Partnerships")
plt.xlabel("Runs")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("Top_10_CSK_Batting_Partnerships.png")
plt.close()

# ----------------------------------------------------------------
# 7. Top-10 batsmen (4s & 6s)
# ----------------------------------------------------------------
boundaries = df[(df["batting_team"] == "Chennai Super Kings")
                & df["runs_off_bat"].isin([4,6])]
batsmen_46  = (boundaries.groupby("striker")["runs_off_bat"].count()
               .nlargest(10))
batsmen_46.plot(kind="barh", figsize=(8,6), color="orange",
                title="Top 10 CSK batsmen – 4s & 6s (counts)")
plt.xlabel("Boundaries")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("Top_10_CSK_batsmen_4s_and_6s.png")
plt.close()

# ----------------------------------------------------------------
# 8. Top-10 batsmen (total runs)
# ----------------------------------------------------------------
top_batsmen = (df[df["batting_team"] == "Chennai Super Kings"]
               .groupby("striker")["runs_off_bat"].sum()
               .nlargest(10))
top_batsmen.plot(kind="barh", figsize=(8,6), color="green",
                 title="Top 10 CSK batsmen – total runs")
plt.xlabel("Runs")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("Top_10_CSK_batsmen.png")
plt.close()

# ----------------------------------------------------------------
# 9. Most economical bowlers (min 30 overs)
# ----------------------------------------------------------------
bowling = df[df["bowling_team"] == "Chennai Super Kings"]
balls_bowled = bowling.groupby("bowler")["ball"].count()
runs_conceded = bowling.groupby("bowler")["total_runs"].sum()
overs = balls_bowled / 6
economy = (runs_conceded / overs).replace([np.inf, -np.inf], np.nan).dropna()
eco_filtered = economy[overs >= 30].nsmallest(10)
eco_filtered.plot(kind="barh", figsize=(8,6), color="purple",
                  title="Top 10 CSK most economical bowlers (≥30 overs)")
plt.xlabel("Economy rate")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("Top_10_CSK_most_economical_bowlers.png")
plt.close()

# ----------------------------------------------------------------
# 10. Wickets vs opponent list
# ----------------------------------------------------------------
wickets_vs_opp = (df[(df["bowling_team"] == "Chennai Super Kings") & df["dismissal"]]
                  .groupby("batting_team")["dismissal"].count()
                  .sort_values(ascending=False))
wickets_vs_opp.plot(kind="bar", figsize=(10,5),
                    title="Total wickets taken by CSK bowlers vs opponents")
plt.ylabel("Wickets")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("Total_wickets_by_CSK_bowlers_vs_opponents.png")
plt.close()

# ----------------------------------------------------------------
# 11. Top-10 wicket-taking bowlers
# ----------------------------------------------------------------
top_bowlers = (df[(df["bowling_team"] == "Chennai Super Kings") & df["dismissal"]]
               .groupby("bowler")["dismissal"].count()
               .nlargest(10))
top_bowlers.plot(kind="barh", figsize=(8,6), color="red",
                 title="Top 10 CSK bowlers – wickets")
plt.xlabel("Wickets")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("Top_10_CSK_bowlers_wickets.png")
plt.close()

print("✅  All analyses complete. PNG files saved alongside the script.")