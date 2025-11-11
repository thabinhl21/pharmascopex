import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def plot_pareto_front_analysis(IC50, AUC, 
                               minimize_IC50=True,
                               maximize_AUC=True,
                               xlabel='IC50',
                               ylabel='AUC',
                               title='Pareto Front Analysis: Minimize IC50 & Maximize AUC',
                               figsize=(10, 6),
                               show_plot=True):
    
    # Convert inputs to numpy arrays
    IC50 = np.asarray(IC50)
    AUC = np.asarray(AUC)
    
    # Validate inputs
    if len(IC50) != len(AUC):
        raise ValueError(f"IC50 and AUC must have the same length. Got {len(IC50)} and {len(AUC)}")
    
    if len(IC50) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Combine the objectives into a single array
    points = np.column_stack([IC50, AUC])
    
    # Minimize IC50 (lower is better) and maximize AUC (higher is better)
    
    # Sort by IC50 first for better performance
    sorted_order = np.argsort(IC50)
    sorted_IC50 = IC50[sorted_order]
    sorted_AUC = AUC[sorted_order]
    
    # Find Pareto front: iterate through sorted points, keep track of best AUC seen
    pareto_mask = []
    max_auc_so_far = -np.inf
    
    for i in range(len(sorted_IC50)):
        # Since sorted by IC50 (ascending), current point has IC50 >= all previous
        # It's on Pareto front if its AUC is better than all points with lower IC50
        if sorted_AUC[i] > max_auc_so_far:
            pareto_mask.append(True)
            max_auc_so_far = sorted_AUC[i]
        else:
            pareto_mask.append(False)
    
    pareto_mask = np.array(pareto_mask)
    
    # Extract Pareto points (already sorted by IC50)
    pareto_IC50 = sorted_IC50[pareto_mask]
    pareto_AUC = sorted_AUC[pareto_mask]
    pareto_indices = sorted_order[pareto_mask]
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    
    # Get Dominated point. These are points with HIGHER IC50 (worse) and LOWER AUC (worse) than at least one Pareto point
    dominated_mask = np.zeros(len(IC50), dtype=bool)
    
    for i in range(len(IC50)):
        if i in pareto_indices:
            continue  # Skip Pareto points
        
        # Check if this point is dominated by ANY Pareto point
        for p_ic50, p_auc in zip(pareto_IC50, pareto_AUC):
            # Point is dominated if Pareto point has:
            # - LOWER or equal IC50 (better or equal potency) AND
            # - HIGHER or equal AUC (better or equal response) AND
            # - At least one is strictly better
            if (p_ic50 <= IC50[i] and p_auc >= AUC[i] and 
                (p_ic50 < IC50[i] or p_auc > AUC[i])):
                dominated_mask[i] = True
                break
    
    # Separate points into categories
    IC50_dominated = IC50[dominated_mask]
    AUC_dominated = AUC[dominated_mask]
    IC50_other = IC50[~dominated_mask & ~np.isin(np.arange(len(IC50)), pareto_indices)]
    AUC_other = AUC[~dominated_mask & ~np.isin(np.arange(len(IC50)), pareto_indices)]
    
    print(f"\nNumber of clearly dominated points (yellow): {len(IC50_dominated)}")
    print(f"Number of other non-Pareto points (blue): {len(IC50_other)}")
    print(f"Number of Pareto points (red): {len(pareto_IC50)}")
    
    # Plot other non-dominated, non-Pareto points 
    if len(IC50_other) > 0:
        plt.scatter(IC50_other, AUC_other, alpha=0.6, s=50, c='lightblue', 
                    edgecolors='navy', linewidth=0.5, label=f'Other Solutions ({len(IC50_other)})', 
                    zorder=1)
    
    # Plot dominated points 
    if len(IC50_dominated) > 0:
        plt.scatter(IC50_dominated, AUC_dominated, alpha=0.7, s=80, c='gold', 
                    edgecolors='darkorange', linewidth=1.2, 
                    label=f'Dominated Solutions ({len(IC50_dominated)})', 
                    zorder=2)
    
    # Connect Pareto front points with line
    plt.plot(pareto_IC50, pareto_AUC, color='red', linewidth=2.5, 
             linestyle='-', alpha=0.85, label='Pareto Front Line', zorder=4)
    
    # Plot Pareto front points transparently
    plt.scatter(pareto_IC50, pareto_AUC, alpha=0.4, s=120, c='orangered', 
                edgecolors='darkred', linewidth=2, label='Pareto Front Points', 
                zorder=5, marker='o')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
       
    if show_plot:
        plt.show()
    
    return fig, pareto_IC50, pareto_AUC, pareto_indices


#========================================================================
# Example with random data
#===========================================================================
np.random.seed(99)
n_points = 100

experimental_IC50 = np.random.exponential(scale=2.0, size=n_points)  # Right-skewed distribution
experimental_AUC = np.random.beta(a=2, b=5, size=n_points)  # Left-skewed distribution

# Add noise
experimental_AUC = experimental_AUC + 0.1 * np.random.randn(n_points)
experimental_AUC = np.clip(experimental_AUC, 0.01, 0.99)

print(f"Generated {len(experimental_IC50)} experimental measurements")
print(f"IC50 range: {experimental_IC50.min():.2f} - {experimental_IC50.max():.2f} μM")
print(f"AUC range: {experimental_AUC.min():.2f} - {experimental_AUC.max():.2f}\n")

# Call the function with the experimental data
fig, pareto_IC50_exp, pareto_AUC_exp, pareto_idx_exp = plot_pareto_front_analysis(
    experimental_IC50,
    experimental_AUC,
    xlabel='IC50 (μM)',
    ylabel='AUC',
    title='Experimental Drug Screening: Pareto Front Analysis'
)
