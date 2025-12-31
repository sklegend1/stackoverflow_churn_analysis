"""
Neural Network Architecture Visualization - LSTM Temporal Model
Creates a professional deep learning diagram showing neurons, layers, and connections

Output: High-resolution PNG image that you can edit in PowerPoint, Inkscape, etc.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

print("=" * 80)
print("CREATING NEURAL NETWORK DIAGRAM FOR LSTM TEMPORAL MODEL")
print("=" * 80)

# =============================================================================
# Configuration
# =============================================================================

COLORS = {
    'input': '#E3F2FD',
    'embed': '#C8E6C9',
    'lstm': '#FFF9C4',
    'concat': '#FFE0B2',
    'fc': '#FFCDD2',
    'head': '#E1BEE7',
    'output': '#CE93D8',
    'loss': '#CFD8DC',
    'neuron': '#FFFFFF',
    'connection': '#666666',
}

# =============================================================================
# Create Figure
# =============================================================================

fig = plt.figure(figsize=(16, 20), facecolor='white', dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 24)
ax.axis('off')

# =============================================================================
# Helper Functions
# =============================================================================

def draw_layer_box(x, y, width, height, label, sublabel, color, neuron_count=0):
    """Draw a layer as a rounded box with optional neurons"""
    # Main box
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.15",
        edgecolor='black',
        facecolor=color,
        linewidth=2.5,
        alpha=0.85,
        zorder=5
    )
    ax.add_patch(box)

    # Main label
    ax.text(x, y + height/2 - 0.25, label,
            fontsize=12, fontweight='bold', ha='center', va='top',
            zorder=10)

    # Sublabel
    if sublabel:
        ax.text(x, y, sublabel,
                fontsize=9, ha='center', va='center',
                style='italic', zorder=10, color='#333333')

    # Draw sample neurons
    if neuron_count > 0:
        max_neurons = min(neuron_count, 6)
        neuron_y = y - height/2 + 0.35
        neuron_spacing = (width - 0.5) / max_neurons

        for i in range(max_neurons):
            neuron_x = x - width/2 + 0.25 + neuron_spacing * (i + 0.5)
            circle = Circle(
                (neuron_x, neuron_y),
                0.12,
                color=COLORS['neuron'],
                ec='black',
                linewidth=1.5,
                zorder=15
            )
            ax.add_patch(circle)

def draw_neuron_layer(x, y, rows, cols, spacing, color):
    """Draw a grid of neurons (for showing neural network structure)"""
    neurons = []
    for row in range(rows):
        for col in range(cols):
            neuron_x = x + (col - cols/2) * spacing
            neuron_y = y + (row - rows/2) * spacing
            circle = Circle(
                (neuron_x, neuron_y),
                0.08,
                color=color,
                ec='black',
                linewidth=1.2,
                zorder=10,
                alpha=0.9
            )
            ax.add_patch(circle)
            neurons.append((neuron_x, neuron_y))
    return neurons

def draw_arrow(x1, y1, x2, y2, label='', width=2, color=COLORS['connection']):
    """Draw a fancy arrow between layers"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.6',
        color=color,
        linewidth=width,
        mutation_scale=20,
        zorder=1,
        alpha=0.7
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, label,
                fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                         edgecolor='gray', alpha=0.9),
                zorder=20)

def draw_multi_connection(x1_start, x1_end, y1, x2_start, x2_end, y2):
    """Draw multiple connection lines between two layers (fan-out/fan-in)"""
    # Draw a few sample connections
    for i in range(3):
        offset1 = (x1_end - x1_start) * (i / 2.5)
        offset2 = (x2_end - x2_start) * ((2 - i) / 2.5)
        draw_arrow(x1_start + offset1, y1, x2_start + offset2, y2, width=1, color='#CCCCCC')

# =============================================================================
# Title
# =============================================================================

ax.text(8, 23, 'LSTM Temporal Model for Ap/Tp Prediction',
        fontsize=18, fontweight='bold', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='black', linewidth=2))

ax.text(8, 22.2, 'Multi-Task Deep Learning Architecture',
        fontsize=11, ha='center', va='top', style='italic', color='#555555')

# =============================================================================
# INPUT LAYER
# =============================================================================

y_current = 20

# Header
ax.text(8, y_current + 0.5, '━━━━━ INPUT LAYER ━━━━━',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        color='#1976D2')

# Three input branches
draw_layer_box(3, y_current, 3, 1.4,
               'Event Sequence',
               '[batch, 50, 2]\nGain + Time Δt',
               COLORS['input'], neuron_count=2)

draw_layer_box(8, y_current, 3, 1.4,
               'Global Features',
               '[batch, 4]\nRecency, Era, Month, Year',
               COLORS['input'], neuron_count=4)

draw_layer_box(13, y_current, 3, 1.4,
               'Persona Features',
               '[batch, 10]\nStats from NMF clusters',
               COLORS['input'], neuron_count=5)

# =============================================================================
# PERSONA EMBEDDING
# =============================================================================

y_current -= 2.5

draw_layer_box(13, y_current, 3, 1.2,
               'Persona Embedding',
               '[batch, 32]\nLinear(10→32) + ReLU',
               COLORS['embed'], neuron_count=5)

draw_arrow(13, 20 - 0.7, 13, y_current + 0.6, '10D', width=2.5, color='#4CAF50')

# =============================================================================
# LSTM LAYER
# =============================================================================

y_current -= 2.5

# Draw LSTM as a special cell structure
lstm_x = 3
lstm_y = y_current

# LSTM main box
draw_layer_box(lstm_x, lstm_y, 3.5, 2,
               'LSTM',
               '[batch, 50, 128]\n━━━━━━━━━━\nLayers: 2\nHidden: 128\nDropout: 0.2',
               COLORS['lstm'], neuron_count=0)

# Draw LSTM internal structure (simplified)
lstm_neurons_50 = draw_neuron_layer(lstm_x - 1, lstm_y + 0.5, 3, 5, 0.25, '#FFE082')
lstm_neurons_128 = draw_neuron_layer(lstm_x + 1, lstm_y + 0.5, 3, 5, 0.25, '#FFF59D')

# Connection from sequence input to LSTM
draw_arrow(3, 20 - 0.7, 3, lstm_y + 1.0, '50 steps\n2D each', width=2.5, color='#FF9800')

# =============================================================================
# LSTM HIDDEN STATE EXTRACTION
# =============================================================================

y_current -= 2.8

draw_layer_box(3, y_current, 3, 1.2,
               'LSTM Hidden State',
               '[batch, 128]\nh_n[-1] (last layer output)',
               COLORS['lstm'], neuron_count=8)

draw_arrow(3, lstm_y - 1.0, 3, y_current + 0.6, 'Extract h_n[-1]', width=2.5, color='#FF9800')

# =============================================================================
# CONCATENATION
# =============================================================================

y_current -= 2.5

draw_layer_box(8, y_current, 4, 1.5,
               'Feature Fusion (Concatenate)',
               '[batch, 164]\n━━━━━━━━━━\nLSTM (128) + Global (4) + Persona (32)',
               COLORS['concat'], neuron_count=8)

# Arrows to concat
draw_arrow(3, y_current + 2.8, 6.5, y_current + 0.75, '128D', width=2, color='#FF5722')
draw_arrow(8, 20 - 0.7, 8, y_current + 0.75, '4D', width=2, color='#FF5722')
draw_arrow(13, y_current + 2.9, 9.5, y_current + 0.75, '32D', width=2, color='#FF5722')

# =============================================================================
# SHARED FC LAYERS
# =============================================================================

y_current -= 2.5

# Header
ax.text(8, y_current + 1.8, '━━━━━ SHARED PROCESSING ━━━━━',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        color='#C62828')

# FC1
fc1_neurons = draw_neuron_layer(8, y_current + 0.3, 6, 8, 0.35, '#FFCDD2')
draw_layer_box(8, y_current, 4.5, 1.8,
               'Fully Connected 1',
               '[batch, 256]\nLinear(164→256) + ReLU + Dropout(0.3)',
               COLORS['fc'], neuron_count=0)

draw_arrow(8, y_current + 2.75, 8, y_current + 0.9, '164D → 256D', width=2.5, color='#E91E63')

y_current -= 2.5

# FC2
fc2_neurons = draw_neuron_layer(8, y_current + 0.3, 5, 7, 0.35, '#FFCDD2')
draw_layer_box(8, y_current, 4.5, 1.8,
               'Fully Connected 2',
               '[batch, 128]\nLinear(256→128) + ReLU + Dropout(0.2)',
               COLORS['fc'], neuron_count=0)

draw_arrow(8, y_current + 2.8, 8, y_current + 0.9, '256D → 128D', width=2.5, color='#E91E63')

# =============================================================================
# MULTI-TASK HEADS
# =============================================================================

y_current -= 3

# Header
ax.text(8, y_current + 1.2, '━━━━━ MULTI-TASK HEADS ━━━━━',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        color='#6A1B9A')

# === CHURN HEAD ===
churn_x = 3

# FC Layer
draw_layer_box(churn_x, y_current, 2.5, 1.2,
               'Churn FC',
               '[batch, 64]\nLinear + ReLU',
               COLORS['head'], neuron_count=6)
draw_arrow(7, y_current + 2.1, churn_x + 0.8, y_current + 0.6, '128→64', width=2, color='#9C27B0')

# Output
y_current_churn = y_current - 1.8
draw_layer_box(churn_x, y_current_churn, 2.5, 1.0,
               'Churn Output',
               '[batch, 1]\nSigmoid',
               COLORS['output'], neuron_count=1)
draw_arrow(churn_x, y_current - 0.6, churn_x, y_current_churn + 0.5, '64→1', width=2, color='#9C27B0')

# Final output
y_final_churn = y_current_churn - 1.3
ax.text(churn_x, y_final_churn, '🎯 Churn Probability',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#BA68C8',
                 edgecolor='black', linewidth=2, alpha=0.9))
ax.text(churn_x, y_final_churn - 0.4, '[0, 1]',
        fontsize=9, ha='center', va='top', style='italic')
draw_arrow(churn_x, y_current_churn - 0.5, churn_x, y_final_churn + 0.3, width=2, color='#9C27B0')

# === AP HEAD ===
ap_x = 8

# FC Layer
draw_layer_box(ap_x, y_current, 2.5, 1.2,
               'Ap FC',
               '[batch, 64]\nLinear + ReLU',
               COLORS['head'], neuron_count=6)
draw_arrow(8, y_current + 2.1, ap_x, y_current + 0.6, '128→64', width=2, color='#9C27B0')

# Output
y_current_ap = y_current - 1.8
draw_layer_box(ap_x, y_current_ap, 2.5, 1.0,
               'Ap Output',
               '[batch, 1]\nSigmoid',
               COLORS['output'], neuron_count=1)
draw_arrow(ap_x, y_current - 0.6, ap_x, y_current_ap + 0.5, '64→1', width=2, color='#9C27B0')

# Final output
y_final_ap = y_current_ap - 1.3
ax.text(ap_x, y_final_ap, '🎯 Ap Threshold',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#BA68C8',
                 edgecolor='black', linewidth=2, alpha=0.9))
ax.text(ap_x, y_final_ap - 0.4, '[0, 1]',
        fontsize=9, ha='center', va='top', style='italic')
draw_arrow(ap_x, y_current_ap - 0.5, ap_x, y_final_ap + 0.3, width=2, color='#9C27B0')

# === TP HEAD ===
tp_x = 13

# FC Layer
draw_layer_box(tp_x, y_current, 2.5, 1.2,
               'Tp FC',
               '[batch, 64]\nLinear + ReLU',
               COLORS['head'], neuron_count=6)
draw_arrow(9, y_current + 2.1, tp_x - 0.8, y_current + 0.6, '128→64', width=2, color='#9C27B0')

# Output
y_current_tp = y_current - 1.8
draw_layer_box(tp_x, y_current_tp, 2.5, 1.0,
               'Tp Output',
               '[batch, 1]\nReLU',
               COLORS['output'], neuron_count=1)
draw_arrow(tp_x, y_current - 0.6, tp_x, y_current_tp + 0.5, '64→1', width=2, color='#9C27B0')

# Final output
y_final_tp = y_current_tp - 1.3
ax.text(tp_x, y_final_tp, '🎯 Tp Window',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#BA68C8',
                 edgecolor='black', linewidth=2, alpha=0.9))
ax.text(tp_x, y_final_tp - 0.4, '[0, ∞) days',
        fontsize=9, ha='center', va='top', style='italic')
draw_arrow(tp_x, y_current_tp - 0.5, tp_x, y_final_tp + 0.3, width=2, color='#9C27B0')

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

y_loss = y_final_churn - 1.5

# Header
ax.text(8, y_loss + 0.8, '━━━━━ LOSS FUNCTIONS ━━━━━',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        color='#37474F')

# Individual losses
draw_layer_box(churn_x, y_loss, 2.2, 0.7,
               'BCE Loss',
               'Weight: 2.0',
               COLORS['loss'], neuron_count=0)
draw_arrow(churn_x, y_final_churn - 0.5, churn_x, y_loss + 0.35, width=1.5, color='red')

draw_layer_box(ap_x, y_loss, 2.2, 0.7,
               'MSE Loss',
               'Weight: 1.0',
               COLORS['loss'], neuron_count=0)
draw_arrow(ap_x, y_final_ap - 0.5, ap_x, y_loss + 0.35, width=1.5, color='red')

draw_layer_box(tp_x, y_loss, 2.2, 0.7,
               'MSE Loss',
               'Weight: 0.1',
               COLORS['loss'], neuron_count=0)
draw_arrow(tp_x, y_final_tp - 0.5, tp_x, y_loss + 0.35, width=1.5, color='red')

# Combined loss
y_loss_final = y_loss - 1.2
draw_layer_box(8, y_loss_final, 5, 0.9,
               'Combined Loss',
               'L = 2.0×BCE(Churn) + 1.0×MSE(Ap) + 0.1×MSE(Tp)',
               '#B0BEC5', neuron_count=0)

draw_arrow(churn_x, y_loss - 0.35, 6.5, y_loss_final + 0.45, width=1.5, color='red')
draw_arrow(ap_x, y_loss - 0.35, 8, y_loss_final + 0.45, width=1.5, color='red')
draw_arrow(tp_x, y_loss - 0.35, 9.5, y_loss_final + 0.45, width=1.5, color='red')

# =============================================================================
# Footer Information
# =============================================================================

footer_y = y_loss_final - 1.2

ax.text(8, footer_y, '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
        fontsize=10, ha='center', va='top', color='#888888')

ax.text(8, footer_y - 0.3,
        'Total Parameters: ~1.5M  |  Framework: PyTorch  |  Optimizer: Adam (LR=0.001)',
        fontsize=9, ha='center', va='top', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EEEEEE', alpha=0.8))

# =============================================================================
# Legend
# =============================================================================

legend_elements = [
    mpatches.Patch(facecolor=COLORS['input'], edgecolor='black', label='Input Layer'),
    mpatches.Patch(facecolor=COLORS['embed'], edgecolor='black', label='Embedding'),
    mpatches.Patch(facecolor=COLORS['lstm'], edgecolor='black', label='LSTM'),
    mpatches.Patch(facecolor=COLORS['concat'], edgecolor='black', label='Concatenation'),
    mpatches.Patch(facecolor=COLORS['fc'], edgecolor='black', label='Fully Connected'),
    mpatches.Patch(facecolor=COLORS['head'], edgecolor='black', label='Task Heads'),
    mpatches.Patch(facecolor=COLORS['loss'], edgecolor='black', label='Loss Functions'),
]

ax.legend(handles=legend_elements, loc='lower center', ncol=7, frameon=True,
          fontsize=8, bbox_to_anchor=(0.5, -0.02))

# =============================================================================
# Save
# =============================================================================

plt.tight_layout()
plt.savefig('lstm_neural_network_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("✓ Saved: lstm_neural_network_diagram.png")
print("\n" + "=" * 80)
print("DIAGRAM CREATED SUCCESSFULLY!")
print("=" * 80)
print("\n📊 Visualization includes:")
print("   ✓ All layers with dimensions")
print("   ✓ Neuron representations (circles)")
print("   ✓ Connections between layers (arrows)")
print("   ✓ Activation functions (ReLU, Sigmoid)")
print("   ✓ Dropout rates")
print("   ✓ Multi-task learning structure")
print("   ✓ Loss function weights")
print("   ✓ Color-coded layer types")
print("\n📁 Output:")
print("   • lstm_neural_network_diagram.png (300 DPI, publication quality)")
print("\n🎨 How to customize:")
print("   1. Open in PowerPoint: Insert > Picture")
print("   2. Edit in Photoshop/GIMP: High-resolution PNG")
print("   3. Modify colors: Edit COLORS dict in this script")
print("   4. Change layout: Adjust x, y coordinates")
print("\n" + "=" * 80)
