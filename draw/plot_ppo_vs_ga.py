import matplotlib.pyplot as plt
import numpy as np


def plot_cuasmrl_ppo_ga_comparison(save_path=None):
    """
    Plot grouped boxplot comparison between CuAsmRL PPO and GA performance
    using the four kernels available in GA data
    """

    # PPO performance data - each run's results (from the reference code)
    triton_baseline_ppo = {
        'mm_leakyrelu': [11.72, 14.61, 12.57, 10.15, 19.97],
        'rmsnorm': [74.75, 72.90, 73.27, 72.88, 73.51],
        'batch_matmul': [19.89, 29.36, 30.54, 27.13, 32.15],
        'fused_softmax': [483.13, 551.52, 378.93, 434.56, 249.53],
        # todo
        # 'flash_attention':
    }

    cuasmrl_ppo_results = {
        'mm_leakyrelu': [15.94, 13.75, 18.80, 13.55, 19.88],
        'rmsnorm': [91.43, 81.80, 92.30, 92.34, 92.53],
        'batch_matmul': [27.91, 28.91, 29.01, 23.82, 20.82],
        'fused_softmax': [350.73, 316.86, 351.42, 303.53, 377.11],
    }

    # GA performance data - from ga_inference2.log (matching kernels only)
    triton_baseline_ga = {
        'mm_leakyrelu': [14.871922, 16.018939, 12.301493, 15.161268, 11.877433],
        'rmsnorm': [72.797557, 83.912931, 72.656319, 73.326993, 73.285994],
        'batch_matmul': [26.945389, 19.703738, 21.280227, 21.021203, 27.583889],
        'fused_softmax': [596.505593, 267.673783, 232.077535, 370.884496, 352.181127],
    }

    ga_results = {
        'mm_leakyrelu': [13.606639, 16.266535, 12.919128, 12.085252, 11.677855],
        'rmsnorm': [75.317912, 91.72288, 87.41047, 77.283019, 91.291664],
        'batch_matmul': [25.643434, 19.614755, 30.791107, 23.593118, 22.975794],
        'fused_softmax': [318.535666, 491.891278, 347.291918, 275.421353, 322.144053],
    }

    # Calculate improvement ratios correctly - pairwise comparison
    common_kernels = ['mm_leakyrelu', 'rmsnorm', 'batch_matmul', 'fused_softmax']

    ppo_ratios = {}
    ga_ratios = {}

    for kernel in common_kernels:
        # PPO ratios - each CuAsmRL result divided by corresponding Triton result
        triton_perfs = triton_baseline_ppo[kernel]
        cuasmrl_perfs = cuasmrl_ppo_results[kernel]
        ppo_ratios[kernel] = [cuasmrl_perfs[i] / triton_perfs[i] for i in range(len(cuasmrl_perfs))]

        # GA ratios - each GA result divided by corresponding Triton result
        triton_perfs = triton_baseline_ga[kernel]
        ga_perfs = ga_results[kernel]
        ga_ratios[kernel] = [ga_perfs[i] / triton_perfs[i] for i in range(len(ga_perfs))]

    # Create single figure with grouped boxplots
    plt.figure(figsize=(15, 9))
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'

    # Prepare data for grouped boxplot
    num_kernels = len(common_kernels)
    positions_ppo = np.arange(1, num_kernels * 3, 3)  # 1, 4, 7, 10
    positions_ga = np.arange(2, num_kernels * 3 + 1, 3)  # 2, 5, 8, 11

    # Create PPO boxplots
    data_to_plot_ppo = [ppo_ratios[kernel] for kernel in common_kernels]
    box_plot_ppo = plt.boxplot(data_to_plot_ppo, positions=positions_ppo,
                               patch_artist=True, widths=0.6,
                               boxprops=dict(facecolor='#3498db', alpha=0.7),
                               medianprops=dict(color='orange', linewidth=2))

    # Create GA boxplots
    data_to_plot_ga = [ga_ratios[kernel] for kernel in common_kernels]
    box_plot_ga = plt.boxplot(data_to_plot_ga, positions=positions_ga,
                              patch_artist=True, widths=0.6,
                              boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                              medianprops=dict(color='orange', linewidth=2))

    # Add baseline
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                alpha=0.7, label='Triton Baseline (1.0x)')

    # Set labels and title
    # plt.xlabel('GPU Kernel Operations', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Throughput', fontsize=14, fontweight='bold')
    plt.title('Performance Comparison: PPO vs GA',
              fontsize=16, fontweight='bold', pad=20)

    # Set x-axis ticks and labels
    tick_positions = (positions_ppo + positions_ga) / 2
    plt.xticks(tick_positions, common_kernels, rotation=45, ha='right')

    # Set y-axis range to show all data points clearly
    all_values = []
    for ratios in ppo_ratios.values():
        all_values.extend(ratios)
    for ratios in ga_ratios.values():
        all_values.extend(ratios)

    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.1
    plt.ylim(y_min, y_max)

    # Set y-axis ticks
    y_tick_spacing = 0.1 if (y_max - y_min) < 1.0 else 0.2
    y_ticks = np.arange(np.floor(y_min * 10) / 10, y_max + 0.01, y_tick_spacing)
    plt.yticks(y_ticks, [f'{tick:.1f}' for tick in y_ticks])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, label='PPO'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='GA'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Triton Baseline')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Add grid
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"CuAsmRL PPO vs GA boxplot saved to: {save_path}")

    # plt.show()

    # Print comparative statistics
    print("\n=== CuAsmRL PPO vs GA Performance Analysis (Pairwise Comparison) ===")
    print(f"{'Kernel':<20} {'PPO Median':<12} {'GA Median':<12} {'PPO Range':<20} {'GA Range':<20} {'Winner'}")
    print("-" * 100)

    ppo_better = 0
    ga_better = 0

    for kernel in common_kernels:
        ppo_median = np.median(ppo_ratios[kernel])
        ga_median = np.median(ga_ratios[kernel])
        ppo_range = f"{min(ppo_ratios[kernel]):.3f}-{max(ppo_ratios[kernel]):.3f}"
        ga_range = f"{min(ga_ratios[kernel]):.3f}-{max(ga_ratios[kernel]):.3f}"

        if ppo_median > ga_median:
            winner = "PPO"
            ppo_better += 1
        else:
            winner = "GA"
            ga_better += 1

        print(f"{kernel:<20} {ppo_median:<12.3f} {ga_median:<12.3f} {ppo_range:<20} {ga_range:<20} {winner}")

    print(
        f"\nSummary: PPO wins {ppo_better}/{len(common_kernels)} kernels, GA wins {ga_better}/{len(common_kernels)} kernels")

    # Overall averages
    ppo_overall_mean = np.mean([np.mean(ratios) for ratios in ppo_ratios.values()])
    ga_overall_mean = np.mean([np.mean(ratios) for ratios in ga_ratios.values()])

    print(f"\nOverall Performance (Pairwise Comparison):")
    print(f"PPO average performance ratio: {ppo_overall_mean:.3f} ({(ppo_overall_mean - 1) * 100:.1f}% improvement)")
    print(f"GA average performance ratio: {ga_overall_mean:.3f} ({(ga_overall_mean - 1) * 100:.1f}% improvement)")

    # Print individual run details for verification
    print(f"\n=== Detailed Pairwise Ratios ===")
    for kernel in common_kernels:
        ppo_improvement = [(r - 1) * 100 for r in ppo_ratios[kernel]]
        ga_improvement = [(r - 1) * 100 for r in ga_ratios[kernel]]
        print(f"\n{kernel}:")
        print(f"  PPO ratios: {[f'{r:.3f}' for r in ppo_ratios[kernel]]}")
        print(f"  GA ratios: {[f'{r:.3f}' for r in ga_ratios[kernel]]}")
        print(f"  PPO improvement %: {[f'{p:.1f}%' for p in ppo_improvement]}")
        print(f"  GA improvement %: {[f'{p:.1f}%' for p in ga_improvement]}")
        print(f"  PPO mean improvement: {(np.mean(ppo_ratios[kernel]) - 1) * 100:.1f}%")
        print(f"  GA mean improvement: {(np.mean(ga_ratios[kernel]) - 1) * 100:.1f}%")

    return {
        'ppo_ratios': ppo_ratios,
        'ga_ratios': ga_ratios,
        'common_kernels': common_kernels,
        'ppo_overall_mean': ppo_overall_mean,
        'ga_overall_mean': ga_overall_mean
    }


# Usage
if __name__ == "__main__":
    results = plot_cuasmrl_ppo_ga_comparison(save_path='../results/cuasmrl_ppo_ga_boxplot(inference).pdf')