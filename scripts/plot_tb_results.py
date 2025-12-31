from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import glob
import os

# Find the latest tfevent file
log_dir = "lightning_logs"
files = glob.glob(f"{log_dir}/**/events.out.tfevents*", recursive=True)
if not files:
    print("No tfevents found.")
    exit()

# Sort by modification time
latest_file = sorted(files, key=os.path.getmtime)[-1]
print(f"Reading {latest_file}")

ea = EventAccumulator(latest_file)
ea.Reload()

print("Available tags:", ea.Tags()['scalars'])

# Extract scalars
metrics = {}
for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    metrics[tag] = [(e.step, e.value) for e in events]

if not metrics:
    print("No scalars found.")
    exit()

# Plot
plt.figure(figsize=(12, 5))

# Plot 1: Loss
plt.subplot(1, 2, 1)
if 'train_loss' in metrics:
    steps, values = zip(*metrics['train_loss'])
    plt.plot(steps, values, label='Train Loss')
if 'val_loss' in metrics:
    steps, values = zip(*metrics['val_loss'])
    plt.plot(steps, values, label='Val Loss')
plt.title("Loss Curves")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Median Error
plt.subplot(1, 2, 2)
if 'val_median_error' in metrics:
    steps, values = zip(*metrics['val_median_error'])
    plt.plot(steps, values, 'r-', label='Median Error (m)')
    plt.title("Localization Error")
    plt.xlabel("Step")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("docs/paper/figures/training_dynamics.png", dpi=300)
print("Saved docs/paper/figures/training_dynamics.png")
