import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrow

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.5)
ax.axis('off')

layers_info = [
    ("Input\n8x8x1", "#CCCCCC"),
    ("Conv2D\n16 filters, 3x3\n+ BatchNorm", "#3D5A80"),
    ("Conv2D\n32 filters, 3x3\n+ BatchNorm", "#3D5A80"),
    ("MaxPool2D\n2x2 -> 4x4x32", "#98C1D9"),
    ("Dropout\n0.3", "#E0E0E0"),
    ("Flatten\n512", "#B8B8B8"),
    ("Dense\n64, ReLU", "#E07A5F"),
    ("Dropout\n0.4", "#E0E0E0"),
    ("Dense\n10, Softmax", "#4C956C"),
]

n = len(layers_info)
box_w, box_h = 1.15, 1.6
gap = 0.35
total_w = n*box_w + (n-1)*gap
start_x = (13 - total_w) / 2
y_center = 2.6

for i, (label, color) in enumerate(layers_info):
    x = start_x + i*(box_w+gap)
    box = FancyBboxPatch((x, y_center-box_h/2), box_w, box_h,
                          boxstyle="round,pad=0.03,rounding_size=0.08",
                          facecolor=color, edgecolor='black', linewidth=1.1, alpha=0.9)
    ax.add_patch(box)
    ax.text(x+box_w/2, y_center, label, ha='center', va='center', fontsize=8.3,
             fontweight='bold', color='white' if color not in ["#CCCCCC","#E0E0E0","#B8B8B8"] else 'black')
    if i < n-1:
        arrow_x = x + box_w + 0.03
        ax.annotate('', xy=(arrow_x+gap-0.06, y_center), xytext=(arrow_x, y_center),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.4))

ax.text(6.5, 0.6, "Final CNN Architecture — 38,474 total parameters", ha='center', fontsize=11, fontweight='bold')
ax.text(6.5, 4.1, "Input (8x8 grayscale) -> Convolutional feature extraction -> Classification head (10 digit classes)",
        ha='center', fontsize=9, style='italic', color='#444444')

plt.tight_layout()
plt.savefig('fig2_architecture_diagram.png', dpi=130)
plt.close()
print("Saved architecture diagram.")
