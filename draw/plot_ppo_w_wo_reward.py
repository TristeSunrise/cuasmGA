import matplotlib.pyplot as plt
import numpy as np


def plot_corrected_grouped_boxplot_comparison(save_path=None):
    """
    Plot grouped boxplot comparison with correct pairwise relative performance calculation
    """

    # PPO performance data - each run's results
    triton_baseline_ppo = {
        'mm_leakyrelu': [11.72, 14.61, 12.57, 10.15, 19.97],
        'rmsnorm': [74.75, 72.90, 73.27, 72.88, 73.51],
        'fused_feedforward': [3.57, 5.21, 4.01, 4.92, 5.84],
        'batch_matmul': [19.89, 29.36, 30.54, 27.13, 32.15],
        'fused_softmax': [483.13, 551.52, 378.93, 434.56, 249.53],
    }

    cuasmrl_ppo_results = {
        'mm_leakyrelu': [15.94, 13.75, 18.80, 13.55, 19.88],
        'rmsnorm': [91.43, 81.80, 92.30, 92.34, 92.53],
        'fused_feedforward': [4.81, 4.93, 4.32, 3.98, 5.30],
        'batch_matmul': [27.91, 28.91, 29.01, 23.82, 20.82],
        'fused_softmax': [350.73, 316.86, 351.42, 303.53, 377.11],
    }

    # Uniform Distribution performance data - each run's results
    triton_baseline_uniform = {
        'mm_leakyrelu': [38.995, 37.874, 21.923, 39.921, 17.982],
        'rmsnorm': [69.543, 72.656, 73.327, 73.173, 72.980],
        'fused_feedforward': [6.849, 8.747, 5.387, 8.318, 4.595],
        'batch_matmul': [54.303, 21.949, 43.592, 55.019, 47.706],
        'fused_softmax': [654.595, 414.127, 678.492, 531.237, 880.525],
    }

    cuasmrl_uniform_results = {
        'mm_leakyrelu': [35.704, 35.384, 24.247, 36.295, 17.517],
        'rmsnorm': [72.919, 72.156, 92.272, 76.160, 69.849],
        'fused_feedforward': [7.820, 9.637, 7.069, 9.203, 6.869],
        'batch_matmul': [33.423, 37.173, 53.679, 44.472, 50.291],
        'fused_softmax': [560.176, 363.674, 977.358, 379.003, 913.511],
    }

    # Calculate improvement ratios correctly - pairwise comparison
    common_kernels = ['mm_leakyrelu', 'rmsnorm', 'batch_matmul', 'fused_softmax', 'fused_feedforward']

    ppo_ratios = {}
    uniform_ratios = {}

    for kernel in common_kernels:
        # PPO ratios - each CuAsmRL result divided by corresponding Triton result
        triton_perfs = triton_baseline_ppo[kernel]
        cuasmrl_perfs = cuasmrl_ppo_results[kernel]
        ppo_ratios[kernel] = [cuasmrl_perfs[i] / triton_perfs[i] for i in range(len(cuasmrl_perfs))]

        # Uniform ratios - each CuAsmRL result divided by corresponding Triton result
        triton_perfs = triton_baseline_uniform[kernel]
        cuasmrl_perfs = cuasmrl_uniform_results[kernel]
        uniform_ratios[kernel] = [cuasmrl_perfs[i] / triton_perfs[i] for i in range(len(cuasmrl_perfs))]

    # Create single figure with grouped boxplots
    plt.figure(figsize=(15, 9))
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'

    # Prepare data for grouped boxplot
    num_kernels = len(common_kernels)
    positions_ppo = np.arange(1, num_kernels * 3, 3)  # 1, 4, 7, 10, 13
    positions_uniform = np.arange(2, num_kernels * 3 + 1, 3)  # 2, 5, 8, 11, 14

    # Create PPO boxplots
    data_to_plot_ppo = [ppo_ratios[kernel] for kernel in common_kernels]
    box_plot_ppo = plt.boxplot(data_to_plot_ppo, positions=positions_ppo,
                               patch_artist=True, widths=0.6,
                               boxprops=dict(facecolor='#3498db', alpha=0.7),
                               medianprops=dict(color='orange', linewidth=2))

    # Create Uniform boxplots
    data_to_plot_uniform = [uniform_ratios[kernel] for kernel in common_kernels]
    box_plot_uniform = plt.boxplot(data_to_plot_uniform, positions=positions_uniform,
                                   patch_artist=True, widths=0.6,
                                   boxprops=dict(facecolor='#e67e22', alpha=0.7),
                                   medianprops=dict(color='orange', linewidth=2))

    # Add baseline
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                alpha=0.7, label='Triton Baseline (1.0x)')

    # Set labels and title
    # plt.xlabel('GPU Kernel Operations', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Throughput', fontsize=14, fontweight='bold')
    plt.title('PPO Performance Comparison w/wo the reward phase',
              fontsize=16, fontweight='bold', pad=20)

    # Set x-axis ticks and labels
    tick_positions = (positions_ppo + positions_uniform) / 2
    plt.xticks(tick_positions, common_kernels, rotation=45, ha='right')

    # Set y-axis range to show all data points clearly
    all_values = []
    for ratios in ppo_ratios.values():
        all_values.extend(ratios)
    for ratios in uniform_ratios.values():
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
        Patch(facecolor='#3498db', alpha=0.7, label='Learned Policy'),
        Patch(facecolor='#e67e22', alpha=0.7, label='Uniform Distribution'),
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
        import os
        save_dir = '../results'
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(save_path)
        if not filename.endswith('.pdf'):
            filename = filename.rsplit('.', 1)[0] + '.pdf'
        full_save_path = os.path.join(save_dir, filename)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', format='pdf')
        print(f"Corrected grouped boxplot saved to: {full_save_path}")

    # plt.show()

    # Print comparative statistics
    print("\n=== PPO vs Uniform Distribution Performance Analysis (Pairwise Comparison) ===")
    print(
        f"{'Kernel':<20} {'PPO Median':<12} {'Uniform Median':<15} {'PPO Range':<20} {'Uniform Range':<20} {'Winner'}")
    print("-" * 110)

    ppo_better = 0
    uniform_better = 0

    for kernel in common_kernels:
        ppo_median = np.median(ppo_ratios[kernel])
        uniform_median = np.median(uniform_ratios[kernel])
        ppo_range = f"{min(ppo_ratios[kernel]):.3f}-{max(ppo_ratios[kernel]):.3f}"
        uniform_range = f"{min(uniform_ratios[kernel]):.3f}-{max(uniform_ratios[kernel]):.3f}"

        if ppo_median > uniform_median:
            winner = "PPO"
            ppo_better += 1
        else:
            winner = "Uniform"
            uniform_better += 1

        print(f"{kernel:<20} {ppo_median:<12.3f} {uniform_median:<15.3f} {ppo_range:<20} {uniform_range:<20} {winner}")

    print(
        f"\nSummary: PPO wins {ppo_better}/{len(common_kernels)} kernels, Uniform wins {uniform_better}/{len(common_kernels)} kernels")

    # Overall averages
    ppo_overall_mean = np.mean([np.mean(ratios) for ratios in ppo_ratios.values()])
    uniform_overall_mean = np.mean([np.mean(ratios) for ratios in uniform_ratios.values()])

    print(f"\nOverall Performance (Pairwise Comparison):")
    print(f"PPO average performance ratio: {ppo_overall_mean:.3f} ({(ppo_overall_mean - 1) * 100:.1f}% improvement)")
    print(
        f"Uniform average performance ratio: {uniform_overall_mean:.3f} ({(uniform_overall_mean - 1) * 100:.1f}% improvement)")

    # Print individual run details for verification
    print(f"\n=== Detailed Pairwise Ratios ===")
    for kernel in common_kernels:
        print(f"\n{kernel}:")
        print(f"  PPO ratios: {[f'{r:.3f}' for r in ppo_ratios[kernel]]}")
        print(f"  Uniform ratios: {[f'{r:.3f}' for r in uniform_ratios[kernel]]}")

    return {
        'ppo_ratios': ppo_ratios,
        'uniform_ratios': uniform_ratios,
        'common_kernels': common_kernels,
        'ppo_overall_mean': ppo_overall_mean,
        'uniform_overall_mean': uniform_overall_mean
    }


# Usage
if __name__ == "__main__":
    results = plot_corrected_grouped_boxplot_comparison(save_path='corrected_ppo_uniform_boxplot(inference).pdf')