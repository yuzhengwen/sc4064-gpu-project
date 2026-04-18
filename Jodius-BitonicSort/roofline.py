import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. HARDWARE SPECIFICATIONS (UPDATED)
# =====================================================================
PEAK_BANDWIDTH_GBPS = 1555.0    
PEAK_COMPUTE_GOPS = 19500.0    

# =====================================================================
# 2. YOUR ALGORITHM DATA (N = 2^20)
# =====================================================================
stages = {
    "Stage 1: CPU Baseline":    {"ai": 0.125, "perf": 0.00016, "color": "cyan", "marker": "o", "size": 120},
    "Stage 2: Naive GPU":       {"ai": 0.125, "perf": 139.54, "color": "navy", "marker": "o", "size": 120},
    "Stage 3: Shared Mem Tile": {"ai": 0.500, "perf": 167.96, "color": "purple", "marker": "^", "size": 120},
    "Stage 4: Warp Shuffle":    {"ai": 1.000, "perf": 157.06, "color": "green", "marker": "D", "size": 120},
    "Stage 5: Thrust Library":  {"ai": 0.125, "perf": 49.11, "color": "red", "marker": "*", "size": 250} 
}

# =====================================================================
# 3. PLOT GENERATION
# =====================================================================
ai_values = np.logspace(-2, 2, 500)

y_memory_bound = ai_values * PEAK_BANDWIDTH_GBPS
y_compute_bound = np.full_like(ai_values, PEAK_COMPUTE_GOPS)
y_roof = np.minimum(y_memory_bound, y_compute_bound)

plt.figure(figsize=(10, 7), dpi=300)
plt.style.use('default')

# Plot the Updated GPU Roofline
plt.plot(ai_values, y_roof, color='#ff9999', linestyle='--', linewidth=1.5, label='A100 Roofline (1555 GB/s)')

# Plot your data points
for name, data in stages.items():
    plt.scatter(data["ai"], data["perf"], color=data["color"], marker=data["marker"], 
                s=data["size"], zorder=5, label=name)

# Formatting the chart
plt.xscale('log', base=10)
plt.yscale('log', base=10)

plt.title('Roofline Model: Bitonic Sort Optimizations ($N=2^{20}$)', fontsize=12)
plt.xlabel('Arithmetic Intensity (Comparisons / Byte)', fontsize=10)
plt.ylabel('Performance (Giga-Comparisons / Second)', fontsize=10)

# Y-limits adjusted
plt.xlim(10**-2, 10**1) 
plt.ylim(10**-4, 10**4)   

# Faint grids matching the visual style
plt.grid(True, which="major", ls="-", color='lightgrey', alpha=0.6, linewidth=0.5)
plt.grid(True, which="minor", ls="-", color='lightgrey', alpha=0.3, linewidth=0.5)

plt.legend(loc='lower right', fontsize=9, framealpha=1.0, edgecolor='lightgrey')

plt.tight_layout()
plt.savefig('Bitonic_Roofline_1555GBps.png')
print("Saved successfully as 'Bitonic_Roofline_1555GBps.png'")