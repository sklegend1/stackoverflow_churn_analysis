import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Professional Neural Network Architecture Visualization
for Temporal LSTM Ap/Tp Model

Creates a visual diagram similar to typical deep learning architecture diagrams
showing neurons, layers, and connections.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.axis('off')

# Colors
COLORS = {
    'input': '#E3F2FD',       # Light blue
    'input_border': '#1565C0',
    'lstm': '#FFF9C4',         # Light yellow
    'lstm_border': '#F57F17',
    'embed': '#C8E6C9',        # Light green
    'embed_border': '#2E7D32',
    'fc': '#E1BEE7',           # Light purple
    'fc_border': '#6A1B9A',
    'output_churn': '#FFCDD2', # Light red
    'output_ap': '#B2DFDB',    # Light teal
    'output_tp': '#FFE0B2',    # Light orange
    'neuron': '#90CAF9',       # Blue neurons
    'connection': '#BDBDBD',   # Gray connections
}

def draw_3d_block(ax, x, y, width, height, depth, color, border_color, label=None, sublabel=None):
    """Draw a 3D block representing feature maps or sequence data"""
    # Front face
    rect = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(rect)
    
    # Draw stacked layers for 3D effect
    offset = 0.08
    for i in range(min(int(depth), 8)):
        layer = FancyBboxPatch((x + offset*(i+1), y + offset*(i+1)), width, height,
                               boxstyle="round,pad=0.02,rounding_size=0.05",
                               facecolor=color, edgecolor=border_color, linewidth=1, alpha=0.7-i*0.08)
        ax.add_patch(layer)
    
    # Add label
    if label:
        ax.text(x + width/2, y - 0.4, label, ha='center', va='top', fontsize=10, fontweight='bold')
    if sublabel:
        ax.text(x + width/2, y - 0.7, sublabel, ha='center', va='top', fontsize=8, color='gray')

def draw_neuron_layer(ax, x, y, num_neurons, max_display=8, radius=0.15, color='#90CAF9', 
                      border_color='#1976D2', label=None, sublabel=None, vertical_spacing=0.5):
    """Draw a layer of neurons (circles)"""
    neurons_to_draw = min(num_neurons, max_display)
    total_height = (neurons_to_draw - 1) * vertical_spacing
    start_y = y - total_height / 2
    
    positions = []
    for i in range(neurons_to_draw):
        ny = start_y + i * vertical_spacing
        circle = Circle((x, ny), radius, facecolor=color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(circle)
        positions.append((x, ny))
        
        # Add dots if we're showing fewer neurons than exist
        if i == neurons_to_draw // 2 and num_neurons > max_display:
            ax.text(x, ny, '⋮', ha='center', va='center', fontsize=14, color='gray')
    
    if label:
        ax.text(x, y + total_height/2 + 0.6, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    if sublabel:
        ax.text(x, y - total_height/2 - 0.4, sublabel, ha='center', va='top', fontsize=8, color='gray')
    
    return positions

def draw_lstm_cell(ax, x, y, width, height, color, border_color, num_cells=2):
    """Draw LSTM cells with internal gates visualization"""
    for i in range(num_cells):
        cell_x = x + i * (width + 0.1)
        
        # Main cell body
        rect = FancyBboxPatch((cell_x, y), width, height,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)
        
        # Internal gates (simplified)
        gate_h = height / 4
        gate_w = width * 0.8
        gate_x = cell_x + (width - gate_w) / 2
        
        # Forget gate
        fg = Rectangle((gate_x, y + height*0.1), gate_w, gate_h, 
                       facecolor='#FFAB91', edgecolor='#E64A19', linewidth=1)
        ax.add_patch(fg)
        ax.text(gate_x + gate_w/2, y + height*0.1 + gate_h/2, 'f', ha='center', va='center', fontsize=7)
        
        # Input gate
        ig = Rectangle((gate_x, y + height*0.35), gate_w, gate_h,
                       facecolor='#A5D6A7', edgecolor='#388E3C', linewidth=1)
        ax.add_patch(ig)
        ax.text(gate_x + gate_w/2, y + height*0.35 + gate_h/2, 'i', ha='center', va='center', fontsize=7)
        
        # Output gate
        og = Rectangle((gate_x, y + height*0.6), gate_w, gate_h,
                       facecolor='#90CAF9', edgecolor='#1976D2', linewidth=1)
        ax.add_patch(og)
        ax.text(gate_x + gate_w/2, y + height*0.6 + gate_h/2, 'o', ha='center', va='center', fontsize=7)
        
        # Cell state
        cs = Rectangle((gate_x, y + height*0.8), gate_w, gate_h * 0.8,
                       facecolor='#CE93D8', edgecolor='#7B1FA2', linewidth=1)
        ax.add_patch(cs)
        ax.text(gate_x + gate_w/2, y + height*0.8 + gate_h*0.4, 'c', ha='center', va='center', fontsize=7)
        
        # Recurrent arrow
        if i < num_cells - 1:
            ax.annotate('', xy=(cell_x + width + 0.1, y + height/2),
                       xytext=(cell_x + width, y + height/2),
                       arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

def draw_arrow(ax, start, end, color='#757575', style='->', lw=1.5, connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                              connectionstyle=connectionstyle))

def draw_bracket(ax, x, y1, y2, direction='left', text=''):
    """Draw a bracket grouping elements"""
    bracket_x = x - 0.3 if direction == 'left' else x + 0.3
    ax.plot([bracket_x, bracket_x-0.1, bracket_x-0.1, bracket_x], 
            [y1, y1, y2, y2], color='gray', lw=1.5)
    ax.text(bracket_x - 0.3, (y1+y2)/2, text, ha='right', va='center', fontsize=9, rotation=90)

# =============================================================================
# Draw the Architecture
# =============================================================================

# Title
ax.text(10, 11.5, 'Temporal LSTM for Dynamic Ap/Tp Prediction', 
        ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(10, 11, 'TemporalChurnPredictor Architecture', 
        ha='center', va='center', fontsize=12, color='gray')

# -------------------------------------------------------------------------
# INPUT LAYER
# -------------------------------------------------------------------------

# Event Sequence Input (50 x 2)
draw_3d_block(ax, 0.5, 5, 1.2, 2.5, 50, COLORS['input'], COLORS['input_border'],
              'Event Sequence', '(50 × 2)')
ax.text(1.1, 7.8, 'gain', ha='center', fontsize=8, style='italic')
ax.text(1.1, 7.5, 'time_delta', ha='center', fontsize=8, style='italic')

# Global Features Input (4D)
global_pos = draw_neuron_layer(ax, 1.1, 2.5, 4, max_display=4, radius=0.18, 
                               color=COLORS['input'], border_color=COLORS['input_border'],
                               label='Global Features', sublabel='(4D)', vertical_spacing=0.5)

# Persona Features Input (10D)
persona_pos = draw_neuron_layer(ax, 1.1, 0.5, 10, max_display=6, radius=0.15,
                                color=COLORS['embed'], border_color=COLORS['embed_border'],
                                label='Persona Features', sublabel='(10D)', vertical_spacing=0.4)

# Labels for inputs
ax.text(1.1, 9.2, 'INPUT LAYER', ha='center', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# -------------------------------------------------------------------------
# LSTM LAYERS
# -------------------------------------------------------------------------

# Draw LSTM cells
lstm_x = 3.5
lstm_y = 4.5

ax.text(4.8, 9.2, 'LSTM LAYERS', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# LSTM Layer 1
draw_lstm_cell(ax, lstm_x, lstm_y + 1.5, 0.8, 1.8, COLORS['lstm'], COLORS['lstm_border'], num_cells=2)
ax.text(lstm_x + 0.9, lstm_y + 3.6, 'LSTM Layer 1', ha='center', fontsize=9, fontweight='bold')
ax.text(lstm_x + 0.9, lstm_y + 3.3, 'hidden=128', ha='center', fontsize=7, color='gray')

# LSTM Layer 2
draw_lstm_cell(ax, lstm_x + 2.2, lstm_y + 1.5, 0.8, 1.8, COLORS['lstm'], COLORS['lstm_border'], num_cells=2)
ax.text(lstm_x + 3.1, lstm_y + 3.6, 'LSTM Layer 2', ha='center', fontsize=9, fontweight='bold')
ax.text(lstm_x + 3.1, lstm_y + 3.3, 'hidden=128', ha='center', fontsize=7, color='gray')

# Dropout indication
ax.text(lstm_x + 2, lstm_y + 0.8, 'dropout=0.2', ha='center', fontsize=7, 
        style='italic', color='#E65100')

# Hidden state output
hidden_pos = draw_neuron_layer(ax, 8, 6.25, 128, max_display=6, radius=0.12,
                               color=COLORS['lstm'], border_color=COLORS['lstm_border'],
                               sublabel='h_n (128D)', vertical_spacing=0.35)

# -------------------------------------------------------------------------
# PERSONA EMBEDDING
# -------------------------------------------------------------------------

ax.text(4.5, 1.5, 'EMBEDDING', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# Embedding layer
embed_box = FancyBboxPatch((3.5, 0), 2, 1.2,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLORS['embed'], edgecolor=COLORS['embed_border'], linewidth=2)
ax.add_patch(embed_box)
ax.text(4.5, 0.7, 'Linear(10→32)', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(4.5, 0.4, '+ ReLU', ha='center', va='center', fontsize=8, color='gray')

# Embedded persona
embed_pos = draw_neuron_layer(ax, 8, 2.5, 32, max_display=5, radius=0.12,
                              color=COLORS['embed'], border_color=COLORS['embed_border'],
                              sublabel='Persona Embed (32D)', vertical_spacing=0.35)

# -------------------------------------------------------------------------
# FEATURE CONCATENATION
# -------------------------------------------------------------------------

concat_x = 9.5
ax.text(concat_x, 9.2, 'CONCATENATE', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# Concatenation symbol
concat_box = FancyBboxPatch((concat_x - 0.4, 3.5), 0.8, 3.5,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='#ECEFF1', edgecolor='#607D8B', linewidth=2)
ax.add_patch(concat_box)
ax.text(concat_x, 5.25, '⊕', ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(concat_x, 3.2, '164D', ha='center', fontsize=9, fontweight='bold', color='#455A64')
ax.text(concat_x, 2.8, '(128+4+32)', ha='center', fontsize=7, color='gray')

# -------------------------------------------------------------------------
# SHARED FC LAYERS
# -------------------------------------------------------------------------

fc_x = 11.5
ax.text(fc_x + 1, 9.2, 'SHARED LAYERS', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# FC1: 164 -> 256
fc1_pos = draw_neuron_layer(ax, fc_x, 5.25, 256, max_display=8, radius=0.13,
                            color=COLORS['fc'], border_color=COLORS['fc_border'],
                            label='FC1', sublabel='256 + ReLU', vertical_spacing=0.4)
ax.text(fc_x, 7.8, 'dropout=0.3', ha='center', fontsize=7, style='italic', color='#E65100')

# FC2: 256 -> 128
fc2_pos = draw_neuron_layer(ax, fc_x + 2, 5.25, 128, max_display=7, radius=0.13,
                            color=COLORS['fc'], border_color=COLORS['fc_border'],
                            label='FC2', sublabel='128 + ReLU', vertical_spacing=0.4)
ax.text(fc_x + 2, 7.5, 'dropout=0.2', ha='center', fontsize=7, style='italic', color='#E65100')

# -------------------------------------------------------------------------
# MULTI-TASK HEADS
# -------------------------------------------------------------------------

head_x = 15.5
ax.text(head_x + 1.5, 9.2, 'MULTI-TASK HEADS', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# Churn Head
churn_hidden = draw_neuron_layer(ax, head_x, 7.5, 64, max_display=5, radius=0.12,
                                 color=COLORS['output_churn'], border_color='#C62828',
                                 label='Churn Head', vertical_spacing=0.35)
churn_out = draw_neuron_layer(ax, head_x + 1.5, 7.5, 1, max_display=1, radius=0.25,
                              color='#EF5350', border_color='#B71C1C',
                              sublabel='P(churn)', vertical_spacing=0)
ax.text(head_x + 1.5, 8.3, 'Sigmoid', ha='center', fontsize=8, color='#B71C1C')

# Ap Head
ap_hidden = draw_neuron_layer(ax, head_x, 5.25, 64, max_display=5, radius=0.12,
                              color=COLORS['output_ap'], border_color='#00695C',
                              label='Ap Head', vertical_spacing=0.35)
ap_out = draw_neuron_layer(ax, head_x + 1.5, 5.25, 1, max_display=1, radius=0.25,
                           color='#26A69A', border_color='#004D40',
                           sublabel='Ap [0,1]', vertical_spacing=0)
ax.text(head_x + 1.5, 6.0, 'Sigmoid', ha='center', fontsize=8, color='#004D40')

# Tp Head
tp_hidden = draw_neuron_layer(ax, head_x, 3, 64, max_display=5, radius=0.12,
                              color=COLORS['output_tp'], border_color='#E65100',
                              label='Tp Head', vertical_spacing=0.35)
tp_out = draw_neuron_layer(ax, head_x + 1.5, 3, 1, max_display=1, radius=0.25,
                           color='#FFA726', border_color='#E65100',
                           sublabel='Tp (days)', vertical_spacing=0)
ax.text(head_x + 1.5, 3.75, 'ReLU', ha='center', fontsize=8, color='#E65100')

# -------------------------------------------------------------------------
# OUTPUT LABELS
# -------------------------------------------------------------------------

out_x = 18.5
ax.text(out_x, 9.2, 'OUTPUTS', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# Output boxes
out_box1 = FancyBboxPatch((out_x - 0.6, 7), 1.5, 1,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#EF5350', edgecolor='#B71C1C', linewidth=2)
ax.add_patch(out_box1)
ax.text(out_x + 0.15, 7.5, 'Churn\nProb', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

out_box2 = FancyBboxPatch((out_x - 0.6, 4.75), 1.5, 1,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#26A69A', edgecolor='#004D40', linewidth=2)
ax.add_patch(out_box2)
ax.text(out_x + 0.15, 5.25, 'Ap\nThreshold', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

out_box3 = FancyBboxPatch((out_x - 0.6, 2.5), 1.5, 1,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#FFA726', edgecolor='#E65100', linewidth=2)
ax.add_patch(out_box3)
ax.text(out_x + 0.15, 3, 'Tp\nWindow', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# -------------------------------------------------------------------------
# DRAW CONNECTIONS (simplified main flow arrows)
# -------------------------------------------------------------------------

# Sequence to LSTM
draw_arrow(ax, (2.2, 6.25), (3.4, 6.25), color='#1565C0', lw=2)

# LSTM internal flow
draw_arrow(ax, (5.3, 6.25), (5.7, 6.25), color=COLORS['lstm_border'], lw=2)

# LSTM to hidden state
draw_arrow(ax, (7.3, 6.25), (7.7, 6.25), color=COLORS['lstm_border'], lw=2)

# Global features to concat
draw_arrow(ax, (1.5, 2.5), (9, 4), color='#1565C0', lw=1.5, connectionstyle='arc3,rad=-0.2')

# Persona to embedding
draw_arrow(ax, (1.5, 0.5), (3.4, 0.6), color=COLORS['embed_border'], lw=1.5)

# Embedding to embedded
draw_arrow(ax, (5.6, 0.6), (7.7, 2.5), color=COLORS['embed_border'], lw=1.5)

# Hidden to concat
draw_arrow(ax, (8.3, 6.25), (9, 6), color=COLORS['lstm_border'], lw=1.5)

# Embedded to concat
draw_arrow(ax, (8.3, 2.5), (9, 4.5), color=COLORS['embed_border'], lw=1.5)

# Concat to FC1
draw_arrow(ax, (10, 5.25), (10.7, 5.25), color='#607D8B', lw=2)

# FC1 to FC2
draw_arrow(ax, (11.8, 5.25), (12.7, 5.25), color=COLORS['fc_border'], lw=2)

# FC2 to heads
draw_arrow(ax, (13.8, 6), (14.7, 7.5), color=COLORS['fc_border'], lw=1.5)
draw_arrow(ax, (13.8, 5.25), (14.7, 5.25), color=COLORS['fc_border'], lw=1.5)
draw_arrow(ax, (13.8, 4.5), (14.7, 3), color=COLORS['fc_border'], lw=1.5)

# Head internals to outputs
draw_arrow(ax, (15.3, 7.5), (15.7, 7.5), color='#C62828', lw=1.5)
draw_arrow(ax, (15.3, 5.25), (15.7, 5.25), color='#00695C', lw=1.5)
draw_arrow(ax, (15.3, 3), (15.7, 3), color='#E65100', lw=1.5)

# Outputs to final boxes
draw_arrow(ax, (16, 7.5), (17.8, 7.5), color='#B71C1C', lw=2)
draw_arrow(ax, (16, 5.25), (17.8, 5.25), color='#004D40', lw=2)
draw_arrow(ax, (16, 3), (17.8, 3), color='#E65100', lw=2)

# -------------------------------------------------------------------------
# LEGEND
# -------------------------------------------------------------------------

legend_y = 0.8
legend_items = [
    (0.5, 'Input', COLORS['input'], COLORS['input_border']),
    (2.5, 'LSTM', COLORS['lstm'], COLORS['lstm_border']),
    (4.5, 'Embedding', COLORS['embed'], COLORS['embed_border']),
    (6.5, 'FC Layer', COLORS['fc'], COLORS['fc_border']),
    (8.5, 'Churn Output', COLORS['output_churn'], '#C62828'),
    (10.5, 'Ap Output', COLORS['output_ap'], '#00695C'),
    (12.5, 'Tp Output', COLORS['output_tp'], '#E65100'),
]

for x, label, color, border in legend_items:
    rect = Rectangle((x, legend_y - 0.15), 0.3, 0.3, facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.45, legend_y, label, va='center', fontsize=8)

# Model info
info_text = "Total Parameters: ~165K | Sequence Length: 50 | LSTM Hidden: 128 | Device: CUDA/CPU"
ax.text(10, 0.2, info_text, ha='center', fontsize=9, color='gray', style='italic')

# Save
plt.tight_layout()
plt.savefig('lstm_architecture_visual.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('lstm_architecture_visual.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('lstm_architecture_visual.svg', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("=" * 60)
print("LSTM Architecture Diagram Generated!")
print("=" * 60)
print("Saved files:")
print("  - lstm_architecture_visual.png (for papers/presentations)")
print("  - lstm_architecture_visual.pdf (vector format)")
print("  - lstm_architecture_visual.svg (editable in Inkscape/Illustrator)")
print("\nYou can customize by editing the COLORS dict and draw functions.")
print("=" * 60)

plt.show()
