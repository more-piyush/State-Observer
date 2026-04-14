import csv, os, statistics, json

RESULTS_DIR = r"C:\Users\USER\Desktop\Projects\PGM State Observer\results"

def read_csv(filepath):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def compute_metrics(rows):
    ep_returns = [float(r['episode_return']) for r in rows]
    ep_lengths = [float(r['episode_length']) for r in rows]
    action_correct = [float(r['action_correct']) for r in rows]
    premature = [float(r['premature_action']) for r in rows]
    return {
        'ep_return_mean': round(statistics.mean(ep_returns), 4),
        'ep_return_std': round(statistics.stdev(ep_returns), 4) if len(ep_returns) > 1 else 0,
        'ep_length_mean': round(statistics.mean(ep_lengths), 4),
        'ep_length_std': round(statistics.stdev(ep_lengths), 4) if len(ep_lengths) > 1 else 0,
        'accuracy': round(statistics.mean(action_correct), 4),
        'premature_rate': round(statistics.mean(premature), 4),
        'n_episodes': len(rows),
    }

results = {}
for group in ['baseline', 'observer']:
    results[group] = {'seeds': {}}
    for seed in range(5):
        fname = f"{group}_seed{seed}.csv"
        fpath = os.path.join(RESULTS_DIR, fname)
        all_rows = read_csv(fpath)
        total = len(all_rows)
        last_1000 = all_rows[-1000:] if total >= 1000 else all_rows
        m = compute_metrics(last_1000)
        m['total_episodes'] = total
        results[group]['seeds'][seed] = m

    # Aggregate
    seed_metrics = results[group]['seeds']
    agg = {}
    for key in ['ep_return_mean','ep_return_std','ep_length_mean','ep_length_std','accuracy','premature_rate']:
        vals = [seed_metrics[s][key] for s in range(5)]
        agg[key+'_mean'] = round(statistics.mean(vals), 4)
        agg[key+'_std'] = round(statistics.stdev(vals), 4)
    total_eps = [seed_metrics[s]['total_episodes'] for s in range(5)]
    agg['total_episodes_mean'] = round(statistics.mean(total_eps), 1)
    agg['total_episodes_std'] = round(statistics.stdev(total_eps), 1)
    results[group]['aggregate'] = agg

# Early training
b0 = read_csv(os.path.join(RESULTS_DIR, 'baseline_seed0.csv'))[:100]
o0 = read_csv(os.path.join(RESULTS_DIR, 'observer_seed0.csv'))[:100]
results['early_training'] = {
    'baseline_seed0_first100': compute_metrics(b0),
    'observer_seed0_first100': compute_metrics(o0),
}

# Print results
print("=" * 80)
print("PER-SEED RESULTS (Last 1000 episodes)")
print("=" * 80)
for group in ['baseline', 'observer']:
    print(f"\n--- {group.upper()} ---")
    for seed in range(5):
        m = results[group]['seeds'][seed]
        print(f"  Seed {seed} (total: {m['total_episodes']}, analyzed: {m['n_episodes']}):")
        print(f"    ep_return:    {m['ep_return_mean']:.4f} +/- {m['ep_return_std']:.4f}")
        print(f"    ep_length:    {m['ep_length_mean']:.4f} +/- {m['ep_length_std']:.4f}")
        print(f"    accuracy:     {m['accuracy']:.4f}")
        print(f"    premature:    {m['premature_rate']:.4f}")

print(f"\n{'='*80}")
print("AGGREGATED ACROSS SEEDS (Last 1000 episodes)")
print("=" * 80)
for group in ['baseline', 'observer']:
    a = results[group]['aggregate']
    print(f"\n--- {group.upper()} (n=5) ---")
    print(f"  ep_return:    {a['ep_return_mean_mean']:.4f} +/- {a['ep_return_mean_std']:.4f}  (within-seed std: {a['ep_return_std_mean']:.4f} +/- {a['ep_return_std_std']:.4f})")
    print(f"  ep_length:    {a['ep_length_mean_mean']:.4f} +/- {a['ep_length_mean_std']:.4f}  (within-seed std: {a['ep_length_std_mean']:.4f} +/- {a['ep_length_std_std']:.4f})")
    print(f"  accuracy:     {a['accuracy_mean']:.4f} +/- {a['accuracy_std']:.4f}")
    print(f"  premature:    {a['premature_rate_mean']:.4f} +/- {a['premature_rate_std']:.4f}")
    print(f"  total_eps:    {a['total_episodes_mean']:.0f} +/- {a['total_episodes_std']:.1f}")

print(f"\n{'='*80}")
print("EARLY TRAINING (First 100 episodes, seed 0)")
print("=" * 80)
for label, key in [("BASELINE", 'baseline_seed0_first100'), ("OBSERVER", 'observer_seed0_first100')]:
    m = results['early_training'][key]
    print(f"\n--- {label} seed0 ---")
    print(f"  ep_return:    {m['ep_return_mean']:.4f} +/- {m['ep_return_std']:.4f}")
    print(f"  ep_length:    {m['ep_length_mean']:.4f} +/- {m['ep_length_std']:.4f}")
    print(f"  accuracy:     {m['accuracy']:.4f}")
    print(f"  premature:    {m['premature_rate']:.4f}")
