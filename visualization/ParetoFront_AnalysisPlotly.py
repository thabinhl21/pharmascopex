import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
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
    
    ### CORRECTION: minimize IC50 and maximize AUC
    pareto_mask = np.zeros(len(IC50), dtype=bool)

    for i in range(len(IC50)):
        dominant = False
        for j in range(len(IC50)):
            if j == i:
                continue

            # j dominates i if
            ### ic50_j < ic50_i (better potency)
            ### AUC_j > AUC_i (better efficiacy)
            if (IC50[j] < IC50[i]) and (AUC[j] > AUC[i]):
                dominant = True
                break

        if not dominant:
            pareto_mask[i] = True

    # extract Pareto points (unsorted)
    pareto_indices = np.where(pareto_mask)[0]
    pareto_IC50 = IC50[pareto_indices]
    pareto_AUC = AUC[pareto_indices]

    # sort Pareto front by IC50 for plotting
    sort_order = np.argsort(pareto_IC50)
    pareto_IC50 = pareto_IC50[sort_order]
    pareto_AUC = pareto_AUC[sort_order]
    pareto_indices   = pareto_indices[sort_order]
    ### CORRECTION END
    
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
    
    # Create the plot using Plotly
    fig = go.Figure()
    
    # Plot other non-dominated, non-Pareto points 
    if len(IC50_other) > 0:
        fig.add_trace(go.Scatter(
            x=IC50_other, 
            y=AUC_other,
            mode='markers',
            marker=dict(
                size=8,
                color='lightblue',
                opacity=0.6,
                line=dict(color='navy', width=0.5)
            ),
            name=f'Other Solutions ({len(IC50_other)})'
        ))
    
    # Plot dominated points 
    if len(IC50_dominated) > 0:
        fig.add_trace(go.Scatter(
            x=IC50_dominated, 
            y=AUC_dominated,
            mode='markers',
            marker=dict(
                size=10,
                color='gold',
                opacity=0.7,
                line=dict(color='darkorange', width=1.2)
            ),
            name=f'Dominated Solutions ({len(IC50_dominated)})'
        ))
    
    # Connect Pareto front points with line
    fig.add_trace(go.Scatter(
        x=pareto_IC50, 
        y=pareto_AUC,
        mode='lines',
        line=dict(color='red', width=2.5),
        opacity=0.85,
        name='Pareto Front Line'
    ))
    
    # Plot Pareto front points
    fig.add_trace(go.Scatter(
        x=pareto_IC50, 
        y=pareto_AUC,
        mode='markers',
        marker=dict(
            size=12,
            color='orangered',
            opacity=0.4,
            line=dict(color='darkred', width=2)
        ),
        name='Pareto Front Points'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=dict(text=xlabel, font=dict(size=12)),
        yaxis_title=dict(text=ylabel, font=dict(size=12)),
        legend=dict(font=dict(size=10)),
        width=figsize[0]*100,
        height=figsize[1]*100,
        showlegend=True,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # if show_plot:
    #     # Save to HTML file and open in browser
    #     import os
    #     html_file = 'pareto_front_plot.html'
    #     fig.write_html(html_file, auto_open=True)
    #     print(f"\nPlot saved to {os.path.abspath(html_file)}")
    #     print("Opening plot in your default browser...")
    
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
    title='Pareto Front Analysis'
)
