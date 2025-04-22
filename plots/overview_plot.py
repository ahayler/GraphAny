import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

# --------------------------------------------------
# Example Data (approximate, based on visual inspection)
# --------------------------------------------------

# "Trained on 1 Graph" data (blue squares)
# Points are roughly at x=1k, 10k
# 'k' in the plot's x-label likely means "thousands", 
# but we can just use numerical values and label them "k".
x_1 = np.array([1.2 * 1e-1, 9.1 * 1e1, 2*1e2])     # in 'thousands'
y_1 = np.array([67.24, 67.45, 67.48])
labels_1 = ["GraphAny\n(Wisconsin)", "GraphAny\n(Arxiv)", "GraphAny\n(Products)"]

# "Trained on 31 Graphs" data (red hexagons)
# Points are roughly at x=100k
x_2 = np.array([5.1 * 1e2, 5.1 * 1e2])  # all at 100k
y_2 = np.array([67.03, 66.55])
labels_2 = ["GAT", "GCN"]

# --------------------------------------------------
# Plotting
# --------------------------------------------------
plt.figure(figsize=(7, 7))

# Set light blue background
plt.gca().set_facecolor('#e6f3ff')  # Light blue background
plt.gcf().set_facecolor('white')    # White figure background

# Set grid to appear below data points
plt.gca().set_axisbelow(True)

# Add grid before plotting data points
plt.grid(alpha=0.5, which='major', linestyle='-', color='white', linewidth=1.5)

# Plot the "1 Graph" data (blue squares)
plt.scatter(x_1, y_1, marker='s', color='tab:blue', 
            edgecolor='black', s=100, label='Trained on 1 Graph')

text_x_factor = 1.2 # We need to multiply by a factor as we have a log scale
font_size = 13

# Label each point
for i, label in enumerate(labels_1):
    if i != 1:
        plt.text(x_1[i]*text_x_factor, y_1[i], label, 
                 ha='left', va='center', fontsize=font_size)
    else:
        plt.text(x_1[i]*0.2, y_1[i], label, 
                 ha='left', va='center', fontsize=font_size,)

# Plot the "31 Graphs" data (red hexagons)
plt.scatter(x_2, y_2, marker='h', color='tab:red', 
            edgecolor='black', s=100, label='Trained on 31 Graphs')

# Label each point
for i, label in enumerate(labels_2):
    # Shift text a bit to the right
    plt.text(x_2[i]*text_x_factor, y_2[i], label, 
             ha='left', va='center', fontsize=font_size)

# Add horizontal line for GraphNP (Arxiv)
plt.axhline(y=68.74, color='green', linestyle='--', linewidth=2, label='Non-parametric')
plt.text(0.9, 68.74, 'GraphNP', ha='left', va='bottom', fontsize=font_size)

# --------------------------------------------------
# Axes, Scales, and Limits
# --------------------------------------------------
plt.xscale('log')  # log-scale on the x-axis
plt.xlabel('Total training labeled nodes (k)', fontsize=14)
plt.ylabel('Avg test acc. on 31 graphs', fontsize=14)
plt.ylim([66.25, 69])          # match the vertical range from figure
plt.xlim([0.9, 1200])       # just past 1 and up to about 1000 in log scale

# Set x-ticks at 0, 0.1, 1, 10, 100, 1000 (all multiplied by 1000)
plt.xticks([0.1, 1, 10, 100, 1000],
           ['0.1', '1', '10', '100', '1000'],
           fontsize=13)
# Remove minor ticks from x-axis
plt.gca().xaxis.set_minor_locator(NullLocator())
plt.yticks(fontsize=13)

# --------------------------------------------------
# Legend and Grid
# --------------------------------------------------
plt.legend(loc='lower left', fontsize=13, frameon=True)

# --------------------------------------------------
# Show or Save
# --------------------------------------------------
plt.tight_layout()
# plt.show()  # Comment out the show command
plt.savefig('plots/overview_plot.pdf', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
