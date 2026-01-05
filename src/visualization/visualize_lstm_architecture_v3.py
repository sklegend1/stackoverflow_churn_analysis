import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Professional Neural Network Architecture Visualization
for Temporal LSTM Ap/Tp Model - Enhanced Version

Creates a detailed visual diagram showing neurons, layers, connections,
and architectural details for the TemporalChurnPredictor model.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# Set up the figure - wider for better spacing
fig, ax = plt.subplots(1, 1, figsize=(28, 16))
ax.set_xlim(0, 28)
ax.set_ylim(0, 16)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors - refined palette
COLORS = {
    'input_seq': '#BBDEFB',       # Light blue
    'input_seq_border': '#1565C0',
    'input_global': '#B3E5FC',    # Lighter blue
    'input_global_border': '#0277BD',
    'input_persona': '#C8E6C9',   # Light green
    'input_persona_border': '#2E7D32',
    'lstm': '#FFF9C4',            # Light yellow
    'lstm_border': '#F57F17',
    'lstm_gate': '#FFE082',
    'embed': '#DCEDC8',           # Light lime
    'embed_border': '#558B2F',
    'concat': '#E0E0E0',          # Gray
    'concat_border': '#616161',
    'fc': '#E1BEE7',              # Light purple
    'fc_border': '#6A1B9A',
    'head_churn': '#FFCDD2',      # Light red
    'head_churn_border': '#C62828',
    'head_ap': '#B2DFDB',         # Light teal
    'head_ap_border': '#00695C',
    'head_tp': '#FFE0B2',         # Light orange
    'head_tp_border': '#E65100',
    'output_churn': '#EF5350',
    'output_ap': '#26A69A',
    'output_tp': '#FFA726',
    'dropout': '#FF8A65',
    'activation': '#81C784',
}

def draw_neuron_layer(ax, x, y, num_neurons, max_display=8, radius=0.15, color='#90CAF9', 
                      border_color='#1976D2', vertical_spacing=0.45, show_dots=True):
    """Draw a layer of neurons (circles) and return positions"""
    neurons_to_draw = min(num_neurons, max_display)
    total_height = (neurons_to_draw - 1) * vertical_spacing
    start_y = y - total_height / 2
    
    positions = []
    dot_shown = False
    for i in range(neurons_to_draw):
        ny = start_y + i * vertical_spacing
        
        # Show dots in the middle if truncated
        if show_dots and num_neurons > max_display and i == neurons_to_draw // 2 and not dot_shown:
            ax.text(x, ny, '⋮', ha='center', va='center', fontsize=12, color='#666')
            dot_shown = True
        else:
            circle = Circle((x, ny), radius, facecolor=color, edgecolor=border_color, 
                           linewidth=1.5, zorder=3)
            ax.add_patch(circle)
        positions.append((x, ny))
    
    return positions, total_height

def draw_labeled_layer(ax, x, y, num_neurons, max_display, radius, color, border_color,
                       label, sublabel=None, dim_label=None, spacing=0.45):
    """Draw a neuron layer with labels"""
    positions, height = draw_neuron_layer(ax, x, y, num_neurons, max_display, radius, 
                                          color, border_color, spacing)
    
    # Top label
    ax.text(x, y + height/2 + 0.5, label, ha='center', va='bottom', 
            fontsize=10, fontweight='bold')
    
    # Dimension label
    if dim_label:
        ax.text(x, y - height/2 - 0.3, dim_label, ha='center', va='top', 
                fontsize=9, color='#333', fontweight='bold')
    
    # Sub label
    if sublabel:
        ax.text(x, y - height/2 - 0.6, sublabel, ha='center', va='top', 
                fontsize=8, color='#666')
    
    return positions, height

def draw_sequence_block(ax, x, y, width, height, num_timesteps, num_features, color, border_color):
    """Draw a 3D block representing sequence data"""
    depth_offset = 0.06
    num_layers = min(num_timesteps // 5, 12)
    
    # Draw back layers first
    for i in range(num_layers - 1, -1, -1):
        layer = FancyBboxPatch(
            (x + depth_offset * i, y + depth_offset * i), 
            width, height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, 
            edgecolor=border_color, 
            linewidth=1.2 if i == 0 else 0.8, 
            alpha=1.0 - i * 0.06,
            zorder=2 - i * 0.1
        )
        ax.add_patch(layer)
    
    return x + width/2, y + height/2

def draw_lstm_unit_detailed(ax, x, y, width, height, layer_num=1, show_labels=True):
    """Draw a detailed LSTM unit with gates, operations, and data flow like standard diagrams"""
    
    # Main cell body - light green background
    rect = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor='#C8E6C9', edgecolor='#2E7D32', 
                          linewidth=2.5, zorder=3)
    ax.add_patch(rect)
    
    # Inner lighter area
    inner_margin = 0.15
    inner_rect = FancyBboxPatch((x + inner_margin, y + inner_margin), 
                                 width - 2*inner_margin, height - 2*inner_margin,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor='#E8F5E9', edgecolor='none', 
                                 linewidth=0, zorder=3.1)
    ax.add_patch(inner_rect)
    
    # === CELL STATE LINE (top horizontal line) ===
    cell_y = y + height * 0.82
    ax.plot([x - 0.3, x + width + 0.3], [cell_y, cell_y], 
            color='#7B1FA2', linewidth=3, zorder=4)
    
    # Cell state input label (c_{t-1})
    if show_labels:
        ax.text(x - 0.5, cell_y + 0.15, '$c_{t-1}$', ha='right', va='center', 
                fontsize=10, color='#7B1FA2', fontweight='bold')
        ax.text(x + width + 0.5, cell_y + 0.15, '$c_t$', ha='left', va='center', 
                fontsize=10, color='#7B1FA2', fontweight='bold')
    
    # === HIDDEN STATE LINE (middle) ===
    hidden_y = y + height * 0.45
    
    # Hidden state input arrow
    ax.annotate('', xy=(x + 0.2, hidden_y), xytext=(x - 0.3, hidden_y),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2), zorder=5)
    
    # Hidden state output arrow  
    ax.annotate('', xy=(x + width + 0.4, hidden_y), xytext=(x + width, hidden_y),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2), zorder=5)
    
    if show_labels:
        ax.text(x - 0.5, hidden_y, '$h_{t-1}$', ha='right', va='center',
                fontsize=10, color='#1565C0', fontweight='bold')
        ax.text(x + width + 0.6, hidden_y, '$h_t$', ha='left', va='center',
                fontsize=10, color='#1565C0', fontweight='bold')
    
    # === INPUT (x_t) - bottom ===
    input_y = y + 0.2
    ax.annotate('', xy=(x + width * 0.3, y + 0.4), xytext=(x + width * 0.3, y - 0.3),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=2), zorder=5)
    if show_labels:
        ax.text(x + width * 0.3, y - 0.5, '$x_t$', ha='center', va='top',
                fontsize=10, color='#E65100', fontweight='bold')
    
    # === GATES (bottom row) ===
    gate_w = width * 0.15
    gate_h = 0.35
    gate_y = y + height * 0.22
    gate_spacing = width * 0.22
    
    gates = [
        ('σ', '#FFAB91', '#E64A19', 'f'),   # Forget gate
        ('σ', '#A5D6A7', '#388E3C', 'i'),   # Input gate  
        ('tanh', '#90CAF9', '#1976D2', 'g'), # Cell candidate (g̃)
        ('σ', '#CE93D8', '#7B1FA2', 'o'),   # Output gate
    ]
    
    gate_positions = []
    for i, (symbol, color, border, label) in enumerate(gates):
        gx = x + width * 0.12 + i * gate_spacing
        gate = FancyBboxPatch((gx, gate_y), gate_w, gate_h,
                              boxstyle="round,pad=0.01,rounding_size=0.05",
                              facecolor=color, edgecolor=border, linewidth=1.5, zorder=4)
        ax.add_patch(gate)
        fontsize = 8 if symbol != 'tanh' else 6
        ax.text(gx + gate_w/2, gate_y + gate_h/2, symbol, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', color='#333')
        # Add gate label below
        ax.text(gx + gate_w/2, gate_y - 0.12, label, ha='center', va='top',
                fontsize=8, fontweight='bold', color=border)
        gate_positions.append((gx + gate_w/2, gate_y + gate_h))
    
    # === OPERATION CIRCLES ===
    op_radius = 0.18
    op_y_mult = y + height * 0.58  # Multiply operations row
    op_y_add = y + height * 0.68   # Add operation
    
    # Multiply circle 1 (forget gate × cell state)
    mult1_x = gate_positions[0][0]
    mult1 = Circle((mult1_x, op_y_mult), op_radius, facecolor='#FFCDD2', 
                   edgecolor='#C62828', linewidth=1.5, zorder=5)
    ax.add_patch(mult1)
    ax.text(mult1_x, op_y_mult, '×', ha='center', va='center', fontsize=12, fontweight='bold', color='#C62828')
    
    # Add circle (forget result + input result)
    add_x = (gate_positions[1][0] + gate_positions[2][0]) / 2
    add_circle = Circle((add_x, op_y_add), op_radius, facecolor='#E1BEE7', 
                        edgecolor='#7B1FA2', linewidth=1.5, zorder=5)
    ax.add_patch(add_circle)
    ax.text(add_x, op_y_add, '+', ha='center', va='center', fontsize=14, fontweight='bold', color='#7B1FA2')
    
    # Multiply circle 2 (input gate × candidate)
    mult2_x = gate_positions[1][0] + gate_spacing * 0.5
    mult2 = Circle((mult2_x, op_y_mult), op_radius, facecolor='#FFCDD2', 
                   edgecolor='#C62828', linewidth=1.5, zorder=5)
    ax.add_patch(mult2)
    ax.text(mult2_x, op_y_mult, '×', ha='center', va='center', fontsize=12, fontweight='bold', color='#C62828')
    
    # Tanh after cell state
    tanh_x = x + width * 0.72
    tanh_y = cell_y - 0.35
    tanh_box = FancyBboxPatch((tanh_x - 0.2, tanh_y - 0.12), 0.4, 0.24,
                              boxstyle="round,pad=0.01,rounding_size=0.03",
                              facecolor='#90CAF9', edgecolor='#1976D2', linewidth=1.5, zorder=5)
    ax.add_patch(tanh_box)
    ax.text(tanh_x, tanh_y, 'tanh', ha='center', va='center', fontsize=6, fontweight='bold', color='#333')
    
    # Multiply circle 3 (output gate × tanh(cell))
    mult3_x = gate_positions[3][0]
    mult3_y = hidden_y + 0.4
    mult3 = Circle((mult3_x, mult3_y), op_radius, facecolor='#FFCDD2', 
                   edgecolor='#C62828', linewidth=1.5, zorder=5)
    ax.add_patch(mult3)
    ax.text(mult3_x, mult3_y, '×', ha='center', va='center', fontsize=12, fontweight='bold', color='#C62828')
    
    # === CONNECTING LINES ===
    line_color = '#666'
    line_width = 1.2
    
    # Gates to multiply operations
    ax.plot([gate_positions[0][0], mult1_x], [gate_positions[0][1], op_y_mult - op_radius], 
            color=line_color, lw=line_width, zorder=3.5)
    ax.plot([gate_positions[1][0], mult2_x], [gate_positions[1][1], op_y_mult - op_radius], 
            color=line_color, lw=line_width, zorder=3.5)
    ax.plot([gate_positions[2][0], mult2_x], [gate_positions[2][1], op_y_mult - op_radius], 
            color=line_color, lw=line_width, zorder=3.5)
    ax.plot([gate_positions[3][0], mult3_x], [gate_positions[3][1], mult3_y - op_radius], 
            color=line_color, lw=line_width, zorder=3.5)
    
    # Cell state connections
    ax.plot([mult1_x, mult1_x], [cell_y, op_y_mult + op_radius], 
            color='#7B1FA2', lw=line_width, zorder=3.5)
    ax.plot([mult1_x, add_x - op_radius], [op_y_mult, op_y_add], 
            color=line_color, lw=line_width, zorder=3.5)
    ax.plot([mult2_x, add_x], [op_y_mult + op_radius, op_y_add - op_radius], 
            color=line_color, lw=line_width, zorder=3.5)
    ax.plot([add_x, add_x], [op_y_add + op_radius, cell_y], 
            color='#7B1FA2', lw=line_width, zorder=3.5)
    
    # Cell state to tanh
    ax.plot([tanh_x, tanh_x], [cell_y, tanh_y + 0.12], 
            color='#7B1FA2', lw=line_width, zorder=3.5)
    
    # Tanh to output multiply
    ax.plot([tanh_x, mult3_x], [tanh_y - 0.12, mult3_y + op_radius], 
            color=line_color, lw=line_width, zorder=3.5)
    
    # Output multiply to hidden state
    ax.plot([mult3_x, mult3_x], [mult3_y - op_radius, hidden_y], 
            color='#1565C0', lw=line_width, zorder=3.5)
    
    return x + width, y + height/2

def draw_lstm_unit(ax, x, y, width, height, layer_num=1):
    """Wrapper for backward compatibility - calls detailed version"""
    return draw_lstm_unit_detailed(ax, x, y, width, height, layer_num, show_labels=True)

def draw_fc_layer_box(ax, x, y, width, height, in_dim, out_dim, activation, 
                      dropout=None, color=None, border=None, label=None):
    """Draw a fully connected layer as a box with details"""
    color = color or COLORS['fc']
    border = border or COLORS['fc_border']
    
    rect = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor=border, linewidth=2, zorder=3)
    ax.add_patch(rect)
    
    # Layer label
    if label:
        ax.text(x + width/2, y + height + 0.2, label, ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # Dimension text
    ax.text(x + width/2, y + height*0.7, f'Linear', ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.text(x + width/2, y + height*0.5, f'{in_dim} → {out_dim}', ha='center', va='center',
            fontsize=9, color='#333')
    
    # Activation
    act_color = '#388E3C' if activation == 'ReLU' else '#1976D2'
    ax.text(x + width/2, y + height*0.28, activation, ha='center', va='center',
            fontsize=8, color=act_color, fontweight='bold')
    
    # Dropout
    if dropout:
        ax.text(x + width/2, y + height*0.1, f'Dropout({dropout})', ha='center', va='center',
                fontsize=7, color=COLORS['dropout'], style='italic')
    
    return x + width/2, y + height/2

def draw_arrow(ax, start, end, color='#666', lw=1.5, style='->', curved=0):
    """Draw an arrow between two points"""
    if curved != 0:
        connectionstyle = f'arc3,rad={curved}'
    else:
        connectionstyle = 'arc3,rad=0'
    
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                              connectionstyle=connectionstyle), zorder=5)

def draw_section_header(ax, x, y, text, width=3):
    """Draw a section header"""
    rect = FancyBboxPatch((x - width/2, y - 0.25), width, 0.5,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='white', edgecolor='#9E9E9E', linewidth=1.5, zorder=10)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', zorder=11)

# =============================================================================
# TITLE
# =============================================================================

ax.text(14, 15.3, 'TemporalChurnPredictor - LSTM Architecture', 
        ha='center', va='center', fontsize=18, fontweight='bold')
ax.text(14, 14.8, 'Multi-Task Learning for Churn Prediction with Dynamic Ap/Tp Thresholds', 
        ha='center', va='center', fontsize=12, color='#555')

# =============================================================================
# SECTION 1: INPUT LAYER
# =============================================================================

draw_section_header(ax, 2, 14, 'INPUT LAYER', width=3.5)

# Event Sequence (50 x 2)
seq_x, seq_y = 0.5, 9.5
seq_center = draw_sequence_block(ax, seq_x, seq_y, 1.8, 3.5, 50, 2, 
                                  COLORS['input_seq'], COLORS['input_seq_border'])
ax.text(seq_x + 0.9, seq_y - 0.5, 'Event Sequence', ha='center', va='bottom', 
        fontsize=10, fontweight='bold')
ax.text(seq_x + 0.9, seq_y - 0.8, '(50 × 2)', ha='center', va='top', 
        fontsize=9, fontweight='bold', color='#333')

# Sequence details
detail_x = seq_x + 0.2
ax.text(detail_x, seq_y + 3.2, '• gain values', ha='left', fontsize=8, color='#1565C0')
ax.text(detail_x, seq_y + 2.9, '• time_delta (days)', ha='left', fontsize=8, color='#1565C0')
ax.text(detail_x, seq_y + 2.5, '50 timesteps', ha='left', fontsize=8, color='#666', style='italic')

# Global Features (4D)
global_y = 5.5
global_pos, _ = draw_labeled_layer(ax, 2, global_y, 4, 4, 0.2, 
                                    COLORS['input_global'], COLORS['input_global_border'],
                                    'Global Features', None, '(4D)', spacing=0.55)

# Global feature labels
feature_labels = ['days_since_last', 'era (0/1/2)', 'month/12', 'year_norm']
for i, (pos, label) in enumerate(zip(global_pos, feature_labels)):
    ax.text(pos[0] + 0.35, pos[1], label, ha='left', va='center', fontsize=7, color='#0277BD')

# Persona Features (10D)
persona_y = 2
persona_pos, _ = draw_labeled_layer(ax, 2, persona_y, 10, 7, 0.18,
                                     COLORS['input_persona'], COLORS['input_persona_border'],
                                     'Persona Features', 'From NMF clusters', '(10D)', spacing=0.42)

# =============================================================================
# SECTION 2: EMBEDDING LAYER  
# =============================================================================

draw_section_header(ax, 7, 7, 'EMBEDDING', width=3)

# Persona embedding layer
embed_x, embed_y = 4.5, 1
embed_w, embed_h = 2, 1.8
draw_fc_layer_box(ax, embed_x, embed_y, embed_w, embed_h, 10, 32, 'ReLU',
                  label='Persona Embedding')

# Embedded output neurons
embed_out_pos, _ = draw_labeled_layer(ax, 7.5, persona_y, 32, 6, 0.15,
                                       COLORS['embed'], COLORS['embed_border'],
                                       '', None, '(32D)', spacing=0.4)

# Arrow from persona to embedding
draw_arrow(ax, (2.5, persona_y), (4.4, embed_y + embed_h/2), COLORS['input_persona_border'], 2)

# Arrow from embedding to output
draw_arrow(ax, (embed_x + embed_w + 0.1, embed_y + embed_h/2), (7.2, persona_y), COLORS['embed_border'], 2)

# =============================================================================
# SECTION 3: LSTM LAYERS
# =============================================================================

draw_section_header(ax, 7, 14, 'LSTM PROCESSING', width=3)

lstm_base_y = 8.2
lstm_w, lstm_h = 2.8, 3.2  # Larger to show internal details

# LSTM Layer 1
lstm1_x = 4
draw_lstm_unit_detailed(ax, lstm1_x, lstm_base_y, lstm_w, lstm_h, layer_num=1, show_labels=True)
ax.text(lstm1_x + lstm_w/2, lstm_base_y + lstm_h + 0.5, 'LSTM Layer 1', 
        ha='center', fontsize=10, fontweight='bold')
ax.text(lstm1_x + lstm_w/2, lstm_base_y + lstm_h + 0.2, 'hidden=128', 
        ha='center', fontsize=8, color='#666')

# LSTM Layer 2
lstm2_x = 8
draw_lstm_unit_detailed(ax, lstm2_x, lstm_base_y, lstm_w, lstm_h, layer_num=2, show_labels=False)
ax.text(lstm2_x + lstm_w/2, lstm_base_y + lstm_h + 0.5, 'LSTM Layer 2',
        ha='center', fontsize=10, fontweight='bold')
ax.text(lstm2_x + lstm_w/2, lstm_base_y + lstm_h + 0.2, 'hidden=128',
        ha='center', fontsize=8, color='#666')

# Dropout annotation between LSTM layers
ax.text((lstm1_x + lstm_w + lstm2_x)/2 + 0.3, lstm_base_y - 0.5, 'dropout=0.2',
        ha='center', fontsize=8, color=COLORS['dropout'], style='italic')

# Arrow from sequence to LSTM1
draw_arrow(ax, (seq_x + 2.2, seq_y + 1.75), (lstm1_x - 0.4, lstm_base_y + lstm_h*0.35), 
           COLORS['input_seq_border'], 2)

# Arrow between LSTM layers (connecting hidden states)
draw_arrow(ax, (lstm1_x + lstm_w + 0.4, lstm_base_y + lstm_h*0.35), 
           (lstm2_x - 0.4, lstm_base_y + lstm_h*0.35), '#1565C0', 2)

# Hidden state output from LSTM
hidden_x = 12
hidden_y = 9.8
hidden_pos, _ = draw_labeled_layer(ax, hidden_x, hidden_y, 128, 8, 0.14,
                                    COLORS['lstm'], COLORS['lstm_border'],
                                    'Hidden State', 'Last layer $h_n$', '(128D)', spacing=0.38)

# Arrow from LSTM2 to hidden
draw_arrow(ax, (lstm2_x + lstm_w + 0.5, lstm_base_y + lstm_h*0.35), 
           (hidden_x - 0.3, hidden_y), '#1565C0', 2)

# =============================================================================
# SECTION 4: FEATURE CONCATENATION
# =============================================================================

draw_section_header(ax, 13.5, 14, 'CONCATENATION', width=3.5)

concat_x, concat_y = 13, 5
concat_w, concat_h = 1.2, 6

# Concatenation box
concat_rect = FancyBboxPatch((concat_x, concat_y), concat_w, concat_h,
                              boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor=COLORS['concat'], edgecolor=COLORS['concat_border'],
                              linewidth=2.5, zorder=3)
ax.add_patch(concat_rect)

# Concatenation symbol and labels
ax.text(concat_x + concat_w/2, concat_y + concat_h/2 + 0.8, '⊕', ha='center', va='center', 
        fontsize=28, fontweight='bold', color='#424242')
ax.text(concat_x + concat_w/2, concat_y + concat_h/2, 'Concat', ha='center', va='center',
        fontsize=10, fontweight='bold')

# Dimension breakdown
ax.text(concat_x + concat_w/2, concat_y + concat_h/2 - 0.8, '164D', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#333')
ax.text(concat_x + concat_w/2, concat_y + 0.5, '128+4+32', ha='center', va='center',
        fontsize=9, color='#666')

# Arrows to concatenation
draw_arrow(ax, (hidden_x + 0.3, hidden_y), (concat_x - 0.1, concat_y + concat_h - 0.8), 
           COLORS['lstm_border'], 2, curved=-0.1)
draw_arrow(ax, (2.5, global_y), (concat_x - 0.1, concat_y + concat_h/2),
           COLORS['input_global_border'], 1.5, curved=0.1)
draw_arrow(ax, (7.8, persona_y + 0.8), (concat_x - 0.1, concat_y + 0.8),
           COLORS['embed_border'], 1.5, curved=0.15)

# =============================================================================
# SECTION 5: SHARED FULLY CONNECTED LAYERS
# =============================================================================

draw_section_header(ax, 18, 14, 'SHARED FC LAYERS', width=3.5)

# FC1: 164 -> 256
fc1_x, fc1_y = 15, 6.5
fc1_w, fc1_h = 1.8, 3.5
draw_fc_layer_box(ax, fc1_x, fc1_y, fc1_w, fc1_h, 164, 256, 'ReLU', 
                  dropout=0.3, label='FC1')

# FC1 neurons visualization
fc1_neurons_x = fc1_x + fc1_w + 0.5
fc1_pos, _ = draw_neuron_layer(ax, fc1_neurons_x, fc1_y + fc1_h/2, 256, 8, 0.12,
                                COLORS['fc'], COLORS['fc_border'], vertical_spacing=0.38)

# FC2: 256 -> 128
fc2_x, fc2_y = 18.5, 6.5
fc2_w, fc2_h = 1.8, 3.5
draw_fc_layer_box(ax, fc2_x, fc2_y, fc2_w, fc2_h, 256, 128, 'ReLU',
                  dropout=0.2, label='FC2')

# FC2 neurons visualization
fc2_neurons_x = fc2_x + fc2_w + 0.5
fc2_pos, _ = draw_neuron_layer(ax, fc2_neurons_x, fc2_y + fc2_h/2, 128, 7, 0.12,
                                COLORS['fc'], COLORS['fc_border'], vertical_spacing=0.4)

# Arrows
draw_arrow(ax, (concat_x + concat_w + 0.1, concat_y + concat_h/2), 
           (fc1_x - 0.1, fc1_y + fc1_h/2), COLORS['concat_border'], 2)
draw_arrow(ax, (fc1_neurons_x + 0.2, fc1_y + fc1_h/2), 
           (fc2_x - 0.1, fc2_y + fc2_h/2), COLORS['fc_border'], 2)

# =============================================================================
# SECTION 6: MULTI-TASK HEADS
# =============================================================================

draw_section_header(ax, 24.5, 14, 'MULTI-TASK HEADS', width=4.5)

head_start_x = 22.5
head_w, head_h = 2.2, 2.5

# Churn Head
churn_y = 11
churn_rect = FancyBboxPatch((head_start_x, churn_y), head_w, head_h,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=COLORS['head_churn'], edgecolor=COLORS['head_churn_border'],
                             linewidth=2, zorder=3)
ax.add_patch(churn_rect)
ax.text(head_start_x + head_w/2, churn_y + head_h - 0.3, 'Churn Head', 
        ha='center', fontsize=10, fontweight='bold')
ax.text(head_start_x + head_w/2, churn_y + head_h/2 + 0.2, 'Linear(128→64)', 
        ha='center', fontsize=8)
ax.text(head_start_x + head_w/2, churn_y + head_h/2 - 0.1, 'ReLU', 
        ha='center', fontsize=8, color='#388E3C')
ax.text(head_start_x + head_w/2, churn_y + head_h/2 - 0.4, 'Linear(64→1)', 
        ha='center', fontsize=8)
ax.text(head_start_x + head_w/2, churn_y + 0.3, 'Sigmoid', 
        ha='center', fontsize=9, fontweight='bold', color='#C62828')

# Ap Head
ap_y = 7.5
ap_rect = FancyBboxPatch((head_start_x, ap_y), head_w, head_h,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=COLORS['head_ap'], edgecolor=COLORS['head_ap_border'],
                          linewidth=2, zorder=3)
ax.add_patch(ap_rect)
ax.text(head_start_x + head_w/2, ap_y + head_h - 0.3, 'Ap Head',
        ha='center', fontsize=10, fontweight='bold')
ax.text(head_start_x + head_w/2, ap_y + head_h/2 + 0.2, 'Linear(128→64)',
        ha='center', fontsize=8)
ax.text(head_start_x + head_w/2, ap_y + head_h/2 - 0.1, 'ReLU',
        ha='center', fontsize=8, color='#388E3C')
ax.text(head_start_x + head_w/2, ap_y + head_h/2 - 0.4, 'Linear(64→1)',
        ha='center', fontsize=8)
ax.text(head_start_x + head_w/2, ap_y + 0.3, 'Sigmoid',
        ha='center', fontsize=9, fontweight='bold', color='#00695C')

# Tp Head
tp_y = 4
tp_rect = FancyBboxPatch((head_start_x, tp_y), head_w, head_h,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=COLORS['head_tp'], edgecolor=COLORS['head_tp_border'],
                          linewidth=2, zorder=3)
ax.add_patch(tp_rect)
ax.text(head_start_x + head_w/2, tp_y + head_h - 0.3, 'Tp Head',
        ha='center', fontsize=10, fontweight='bold')
ax.text(head_start_x + head_w/2, tp_y + head_h/2 + 0.2, 'Linear(128→64)',
        ha='center', fontsize=8)
ax.text(head_start_x + head_w/2, tp_y + head_h/2 - 0.1, 'ReLU',
        ha='center', fontsize=8, color='#388E3C')
ax.text(head_start_x + head_w/2, tp_y + head_h/2 - 0.4, 'Linear(64→1)',
        ha='center', fontsize=8)
ax.text(head_start_x + head_w/2, tp_y + 0.3, 'ReLU',
        ha='center', fontsize=9, fontweight='bold', color='#E65100')

# Arrows from FC2 to heads
fc2_out_x = fc2_neurons_x + 0.2
draw_arrow(ax, (fc2_out_x, fc2_y + fc2_h/2 + 1), (head_start_x - 0.1, churn_y + head_h/2),
           COLORS['fc_border'], 1.5, curved=-0.15)
draw_arrow(ax, (fc2_out_x, fc2_y + fc2_h/2), (head_start_x - 0.1, ap_y + head_h/2),
           COLORS['fc_border'], 1.5)
draw_arrow(ax, (fc2_out_x, fc2_y + fc2_h/2 - 1), (head_start_x - 0.1, tp_y + head_h/2),
           COLORS['fc_border'], 1.5, curved=0.15)

# =============================================================================
# SECTION 7: OUTPUTS
# =============================================================================

output_x = 26
output_r = 0.5

# Churn output
churn_out = Circle((output_x, churn_y + head_h/2), output_r, 
                   facecolor=COLORS['output_churn'], edgecolor=COLORS['head_churn_border'],
                   linewidth=2.5, zorder=4)
ax.add_patch(churn_out)
ax.text(output_x, churn_y + head_h/2, 'P', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='white')
ax.text(output_x + 0.7, churn_y + head_h/2 + 0.2, 'Churn Prob', ha='left', 
        fontsize=9, fontweight='bold')
ax.text(output_x + 0.7, churn_y + head_h/2 - 0.15, '[0, 1]', ha='left', 
        fontsize=8, color='#666')
ax.text(output_x + 0.7, churn_y + head_h/2 - 0.45, 'BCE×2.0', ha='left',
        fontsize=7, color=COLORS['head_churn_border'], style='italic')

# Ap output
ap_out = Circle((output_x, ap_y + head_h/2), output_r,
                facecolor=COLORS['output_ap'], edgecolor=COLORS['head_ap_border'],
                linewidth=2.5, zorder=4)
ax.add_patch(ap_out)
ax.text(output_x, ap_y + head_h/2, 'Ap', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(output_x + 0.7, ap_y + head_h/2 + 0.2, 'Activity Thresh', ha='left',
        fontsize=9, fontweight='bold')
ax.text(output_x + 0.7, ap_y + head_h/2 - 0.15, '[0, 1]', ha='left',
        fontsize=8, color='#666')
ax.text(output_x + 0.7, ap_y + head_h/2 - 0.45, 'MSE×1.0', ha='left',
        fontsize=7, color=COLORS['head_ap_border'], style='italic')

# Tp output
tp_out = Circle((output_x, tp_y + head_h/2), output_r,
                facecolor=COLORS['output_tp'], edgecolor=COLORS['head_tp_border'],
                linewidth=2.5, zorder=4)
ax.add_patch(tp_out)
ax.text(output_x, tp_y + head_h/2, 'Tp', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(output_x + 0.7, tp_y + head_h/2 + 0.2, 'Time Window', ha='left',
        fontsize=9, fontweight='bold')
ax.text(output_x + 0.7, tp_y + head_h/2 - 0.15, '[0, ∞) days', ha='left',
        fontsize=8, color='#666')
ax.text(output_x + 0.7, tp_y + head_h/2 - 0.45, 'MSE×0.1', ha='left',
        fontsize=7, color=COLORS['head_tp_border'], style='italic')

# Arrows to outputs
draw_arrow(ax, (head_start_x + head_w + 0.1, churn_y + head_h/2), 
           (output_x - output_r - 0.1, churn_y + head_h/2), COLORS['head_churn_border'], 2)
draw_arrow(ax, (head_start_x + head_w + 0.1, ap_y + head_h/2),
           (output_x - output_r - 0.1, ap_y + head_h/2), COLORS['head_ap_border'], 2)
draw_arrow(ax, (head_start_x + head_w + 0.1, tp_y + head_h/2),
           (output_x - output_r - 0.1, tp_y + head_h/2), COLORS['head_tp_border'], 2)

# =============================================================================
# LEGEND & INFO
# =============================================================================

legend_y = 1.2
legend_items = [
    (11, 'Sequence Input', COLORS['input_seq'], COLORS['input_seq_border']),
    (13, 'Global Features', COLORS['input_global'], COLORS['input_global_border']),
    (15, 'Persona/Embed', COLORS['embed'], COLORS['embed_border']),
    (17, 'LSTM Layers', COLORS['lstm'], COLORS['lstm_border']),
    (19, 'FC Layers', COLORS['fc'], COLORS['fc_border']),
    (21, 'Churn Head', COLORS['head_churn'], COLORS['head_churn_border']),
    (23, 'Ap Head', COLORS['head_ap'], COLORS['head_ap_border']),
    (25, 'Tp Head', COLORS['head_tp'], COLORS['head_tp_border']),
]

# Legend background
legend_bg = FancyBboxPatch((10, 0.5), 16.5, 1.5,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1)
ax.add_patch(legend_bg)

for x, label, color, border in legend_items:
    rect = Rectangle((x, legend_y - 0.15), 0.4, 0.4, 
                     facecolor=color, edgecolor=border, linewidth=1.5, zorder=5)
    ax.add_patch(rect)
    ax.text(x + 0.55, legend_y, label, va='center', fontsize=8, zorder=5)

# Model specifications
spec_text = "Parameters: ~165K  |  Seq Length: 50  |  LSTM Hidden: 128  |  Batch: 256  |  LR: 0.001  |  Optimizer: Adam"
ax.text(17.5, 0.15, spec_text, ha='center', fontsize=9, color='#666', style='italic')

# =============================================================================
# SAVE
# =============================================================================

plt.tight_layout()
plt.savefig('lstm_architecture_detailed.png', dpi=250, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('lstm_architecture_detailed.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('lstm_architecture_detailed.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("=" * 70)
print("ENHANCED LSTM Architecture Diagram Generated!")
print("=" * 70)
print("\nSaved files:")
print("  📊 lstm_architecture_detailed.png  (250 DPI, for papers)")
print("  📄 lstm_architecture_detailed.pdf  (vector, for printing)")
print("  🎨 lstm_architecture_detailed.svg  (editable in Inkscape/Illustrator)")
print("\n✨ Improvements in this version:")
print("  • Better spacing - no overlaps")
print("  • LSTM gates shown (Forget, Input, Cell, Output)")
print("  • Detailed layer dimensions")
print("  • Loss function weights for each output")
print("  • Dropout annotations")
print("  • Activation functions highlighted")
print("  • Section headers for clarity")
print("  • Legend with all components")
print("=" * 70)

plt.show()
