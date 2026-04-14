import csv
import os
import math

results_dir = r"C:\Users\USER\Desktop\Projects\PGM State Observer\results_multistep"

files = [
    "form_submission_baseline_seed0.csv",
    "form_submission_baseline_seed1.csv",
    "form_submission_baseline_seed2.csv",
    "form_submission_observer_seed0.csv",
    "form_submission_observer_seed1.csv",
    "form_submission_observer_seed2.csv",
    "search_paginated_baseline_seed0.csv",
    "search_paginated_baseline_seed1.csv",
    "search_paginated_baseline_seed2.csv",
    "search_paginated_observer_seed0.csv",
    "search_paginated_observer_seed1.csv",
    "search_paginated_observer_seed2.csv",
    "checkout_baseline_seed0.csv",
    "checkout_baseline_seed1.csv",
    "checkout_baseline_seed2.csv",
    "checkout_observer_seed0.csv",
    "checkout_observer_seed1.csv",
    "checkout_observer_seed2.csv",
]

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def analyze_file(filepath, last_n=1000):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_episodes = len(rows)
    subset = rows[-last_n:] if len(rows) >= last_n else rows

    episode_returns = [float(r['episode_return']) for r in subset]
    total_steps = [float(r['total_steps']) for r in subset]
    completion_rates = [float(r['completion_rate']) for r in subset]
    premature_execs = [float(r['premature_executions']) for r in subset]
    correct_execs = [float(r['correct_executions']) for r in subset]
    nodes_completed = [float(r['nodes_completed']) for r in subset]
    nodes_skipped = [float(r['nodes_skipped']) for r in subset]

    return {
        'total_episodes': total_episodes,
        'n_analyzed': len(subset),
        'episode_return_mean': mean(episode_returns),
        'episode_return_std': std(episode_returns),
        'total_steps_mean': mean(total_steps),
        'total_steps_std': std(total_steps),
        'completion_rate_mean': mean(completion_rates),
        'premature_executions_mean': mean(premature_execs),
        'correct_executions_mean': mean(correct_execs),
        'nodes_completed_mean': mean(nodes_completed),
        'nodes_skipped_mean': mean(nodes_skipped),
    }

def early_training(filepath, first_n=50):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= first_n:
                break
            rows.append(row)

    episode_returns = [float(r['episode_return']) for r in rows]
    total_steps = [float(r['total_steps']) for r in rows]
    completion_rates = [float(r['completion_rate']) for r in rows]
    premature_execs = [float(r['premature_executions']) for r in rows]
    correct_execs = [float(r['correct_executions']) for r in rows]
    nodes_completed = [float(r['nodes_completed']) for r in rows]
    nodes_skipped = [float(r['nodes_skipped']) for r in rows]

    return {
        'n': len(rows),
        'episode_return_mean': mean(episode_returns),
        'episode_return_std': std(episode_returns),
        'total_steps_mean': mean(total_steps),
        'total_steps_std': std(total_steps),
        'completion_rate_mean': mean(completion_rates),
        'premature_executions_mean': mean(premature_execs),
        'correct_executions_mean': mean(correct_execs),
        'nodes_completed_mean': mean(nodes_completed),
        'nodes_skipped_mean': mean(nodes_skipped),
    }

# Per-file results
print("=" * 120)
print("PER-FILE RESULTS (last 1000 episodes)")
print("=" * 120)

all_results = {}
for fname in files:
    fpath = os.path.join(results_dir, fname)
    r = analyze_file(fpath, last_n=1000)
    all_results[fname] = r
    print(f"\n--- {fname} ---")
    print(f"  Total episodes: {r['total_episodes']}")
    print(f"  Analyzed (last 1000): {r['n_analyzed']}")
    print(f"  episode_return:  mean={r['episode_return_mean']:.4f}  std={r['episode_return_std']:.4f}")
    print(f"  total_steps:     mean={r['total_steps_mean']:.4f}  std={r['total_steps_std']:.4f}")
    print(f"  completion_rate: mean={r['completion_rate_mean']:.6f}")
    print(f"  premature_exec:  mean={r['premature_executions_mean']:.4f}")
    print(f"  correct_exec:    mean={r['correct_executions_mean']:.4f}")
    print(f"  nodes_completed: mean={r['nodes_completed_mean']:.4f}")
    print(f"  nodes_skipped:   mean={r['nodes_skipped_mean']:.4f}")

# Grouped results
print("\n" + "=" * 120)
print("GROUPED RESULTS (by workflow and mode, aggregated across 3 seeds)")
print("=" * 120)

workflows = ['form_submission', 'search_paginated', 'checkout']
modes = ['baseline', 'observer']
metrics = ['episode_return_mean', 'episode_return_std', 'total_steps_mean', 'total_steps_std',
           'completion_rate_mean', 'premature_executions_mean', 'correct_executions_mean',
           'nodes_completed_mean', 'nodes_skipped_mean', 'total_episodes']

for wf in workflows:
    for mode in modes:
        seed_files = [f"{wf}_{mode}_seed{s}.csv" for s in range(3)]
        seed_results = [all_results[f] for f in seed_files]

        print(f"\n--- {wf} / {mode} ---")

        # For each metric, compute mean and std across the 3 seeds
        for metric in metrics:
            vals = [sr[metric] for sr in seed_results]
            m = mean(vals)
            s = std(vals)
            print(f"  {metric:30s}: {m:.4f} +/- {s:.4f}   (seeds: {', '.join(f'{v:.4f}' for v in vals)})")

# Early training
print("\n" + "=" * 120)
print("EARLY TRAINING (first 50 episodes)")
print("=" * 120)

for fname in ["form_submission_baseline_seed0.csv", "form_submission_observer_seed0.csv"]:
    fpath = os.path.join(results_dir, fname)
    r = early_training(fpath, first_n=50)
    print(f"\n--- {fname} (first {r['n']} episodes) ---")
    print(f"  episode_return:  mean={r['episode_return_mean']:.4f}  std={r['episode_return_std']:.4f}")
    print(f"  total_steps:     mean={r['total_steps_mean']:.4f}  std={r['total_steps_std']:.4f}")
    print(f"  completion_rate: mean={r['completion_rate_mean']:.6f}")
    print(f"  premature_exec:  mean={r['premature_executions_mean']:.4f}")
    print(f"  correct_exec:    mean={r['correct_executions_mean']:.4f}")
    print(f"  nodes_completed: mean={r['nodes_completed_mean']:.4f}")
    print(f"  nodes_skipped:   mean={r['nodes_skipped_mean']:.4f}")
