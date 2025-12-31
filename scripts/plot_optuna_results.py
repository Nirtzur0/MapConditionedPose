import optuna
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np

# Connect to DB to get study names
try:
    conn = sqlite3.connect("optuna_study.db")
    cursor = conn.cursor()
    cursor.execute("SELECT study_name FROM studies")
    study_names = [row[0] for row in cursor.fetchall()]
    conn.close()
except Exception as e:
    print(f"Error reading DB: {e}")
    exit()

print(f"Found studies: {study_names}")

if not study_names:
    print("No studies found.")
    exit()

# Load the latest study
study_name = sorted(study_names)[-1]
print(f"Loading study: {study_name}")

study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna_study.db")

# Extract dataframe
df = study.trials_dataframe()

# Filter for complete trials
df_complete = df[df.state == "COMPLETE"].copy()

if df_complete.empty:
    print("No complete trials found yet.")
    exit()

print(f"Number of complete trials: {len(df_complete)}")

# --- Plot 1: Optimization History ---
plt.figure(figsize=(10, 6))
# Plot all trials
plt.plot(df_complete["number"], df_complete["value"], 'b-', alpha=0.3, label='_nolegend_')
plt.scatter(df_complete["number"], df_complete["value"], s=100, c="blue", alpha=0.6, label="Trial")

# Highlight best
best_trial = study.best_trial
plt.scatter(best_trial.number, best_trial.value, color="red", s=150, zorder=5, label="Best Trial")

plt.title(f"Optimization History: {study_name}")
plt.xlabel("Trial Number")
plt.ylabel("Validation Median Error (m)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("docs/paper/figures/optuna_history.png", dpi=300, bbox_inches="tight")
print("Saved docs/paper/figures/optuna_history.png")

# --- Plot 2: Parameter Importance ---
try:
    if len(df_complete) > 3:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        scores = list(importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(scores)
        params = [params[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(params, scores, color='teal')
        plt.title("Hyperparameter Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("docs/paper/figures/param_importance.png", dpi=300, bbox_inches="tight")
        print("Saved docs/paper/figures/param_importance.png")
except Exception as e:
    print(f"Could not plot importance: {e}")
