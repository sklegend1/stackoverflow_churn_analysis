import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
Neural Network Architecture Visualization for LSTM Temporal Model

Creates publication-quality diagrams showing:
1. Layer-by-layer architecture with neuron counts
2. Connections between layers
3. Multi-task output heads

Output formats:
- PNG image (high resolution)
- SVG (scalable vector graphics - fully customizable)
- DOT file (editable in Graphviz tools)
"""

import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

print("=" * 80)
print("LSTM TEMPORAL MODEL - NEURAL NETWORK VISUALIZATION")
print("=" * 80)

# =============================================================================
# Method 1: Graphviz (Professional NN Diagram)
# =============================================================================

print("\n[1/3] Creating Graphviz diagram...")

# Create directed graph
dot = graphviz.Digraph(
    'LSTM_Temporal_Model',
    comment='Temporal LSTM for Ap/Tp Prediction',
    format='png',
    engine='dot'
)

# Configure graph styling
dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
dot.attr('edge', fontname='Arial', fontsize='9', color='#666666')

# =============================================================================
# Input Layer
# =============================================================================

with dot.subgraph(name='cluster_input') as c:
    c.attr(label='INPUT LAYER', style='dashed', color='blue', fontsize='12', fontname='Arial Bold')

    c.node('seq_input',
           'Event Sequence\n[batch, 50, 2]\n━━━━━━━━━━\nGain values\nTime deltas',
           fillcolor='#E3F2FD', shape='box')

    c.node('global_input',
           'Global Features\n[batch, 4]\n━━━━━━━━━━\nDays since last\nEra\nMonth\nYear',
           fillcolor='#E3F2FD', shape='box')

    c.node('persona_input',
           'Persona Features\n[batch, 10]\n━━━━━━━━━━\nReputation\nUpvotes\nAnswers/Questions\nDuration\nCollectivistic\nH/C/T flags',
           fillcolor='#E3F2FD', shape='box')

# =============================================================================
# Persona Embedding
# =============================================================================

with dot.subgraph(name='cluster_embed') as c:
    c.attr(label='EMBEDDING LAYER', style='dashed', color='green', fontsize='12', fontname='Arial Bold')

    c.node('persona_embed',
           'Persona Embedding\n[batch, 32]\n━━━━━━━━━━\nLinear(10 → 32)\n+ ReLU',
           fillcolor='#C8E6C9', shape='box')

# =============================================================================
# LSTM Layer
# =============================================================================

with dot.subgraph(name='cluster_lstm') as c:
    c.attr(label='SEQUENCE PROCESSING', style='dashed', color='orange', fontsize='12', fontname='Arial Bold')

    c.node('lstm',
           'LSTM Layer\n[batch, 50, 128]\n━━━━━━━━━━\nInput size: 2\nHidden size: 128\nNum layers: 2\nDropout: 0.2\n━━━━━━━━━━\nBidirectional: No',
           fillcolor='#FFF9C4', shape='box')

    c.node('lstm_hidden',
           'Hidden State\n[batch, 128]\n━━━━━━━━━━\nLast LSTM output',
           fillcolor='#FFF9C4', shape='box')

# =============================================================================
# Concatenation
# =============================================================================

dot.node('concat',
         'Concatenate\n[batch, 164]\n━━━━━━━━━━\nLSTM (128)\n+ Global (4)\n+ Persona (32)',
         fillcolor='#FFE0B2', shape='box', style='rounded,filled')

# =============================================================================
# Shared FC Layers
# =============================================================================

with dot.subgraph(name='cluster_shared') as c:
    c.attr(label='SHARED PROCESSING', style='dashed', color='red', fontsize='12', fontname='Arial Bold')

    c.node('fc1',
           'FC Layer 1\n[batch, 256]\n━━━━━━━━━━\nLinear(164 → 256)\n+ ReLU\n+ Dropout(0.3)',
           fillcolor='#FFCDD2', shape='box')

    c.node('fc2',
           'FC Layer 2\n[batch, 128]\n━━━━━━━━━━\nLinear(256 → 128)\n+ ReLU\n+ Dropout(0.2)',
           fillcolor='#FFCDD2', shape='box')

# =============================================================================
# Multi-Task Heads
# =============================================================================

# Churn Head
with dot.subgraph(name='cluster_churn') as c:
    c.attr(label='CHURN HEAD', style='dashed', color='purple', fontsize='12', fontname='Arial Bold')

    c.node('churn_fc1',
           'FC Layer\n[batch, 64]\n━━━━━━━━━━\nLinear(128 → 64)\n+ ReLU',
           fillcolor='#E1BEE7', shape='box')

    c.node('churn_fc2',
           'Output Layer\n[batch, 1]\n━━━━━━━━━━\nLinear(64 → 1)\n+ Sigmoid',
           fillcolor='#CE93D8', shape='box')

    c.node('churn_out',
           'Churn Probability\n━━━━━━━━━━\nRange: [0, 1]',
           fillcolor='#BA68C8', shape='ellipse', style='filled')

# Ap Head
with dot.subgraph(name='cluster_ap') as c:
    c.attr(label='AP THRESHOLD HEAD', style='dashed', color='purple', fontsize='12', fontname='Arial Bold')

    c.node('ap_fc1',
           'FC Layer\n[batch, 64]\n━━━━━━━━━━\nLinear(128 → 64)\n+ ReLU',
           fillcolor='#E1BEE7', shape='box')

    c.node('ap_fc2',
           'Output Layer\n[batch, 1]\n━━━━━━━━━━\nLinear(64 → 1)\n+ Sigmoid',
           fillcolor='#CE93D8', shape='box')

    c.node('ap_out',
           'Ap Threshold\n━━━━━━━━━━\nRange: [0, 1]',
           fillcolor='#BA68C8', shape='ellipse', style='filled')

# Tp Head
with dot.subgraph(name='cluster_tp') as c:
    c.attr(label='TP WINDOW HEAD', style='dashed', color='purple', fontsize='12', fontname='Arial Bold')

    c.node('tp_fc1',
           'FC Layer\n[batch, 64]\n━━━━━━━━━━\nLinear(128 → 64)\n+ ReLU',
           fillcolor='#E1BEE7', shape='box')

    c.node('tp_fc2',
           'Output Layer\n[batch, 1]\n━━━━━━━━━━\nLinear(64 → 1)\n+ ReLU',
           fillcolor='#CE93D8', shape='box')

    c.node('tp_out',
           'Tp Window (days)\n━━━━━━━━━━\nRange: [0, ∞)',
           fillcolor='#BA68C8', shape='ellipse', style='filled')

# =============================================================================
# Loss Functions
# =============================================================================

with dot.subgraph(name='cluster_loss') as c:
    c.attr(label='LOSS FUNCTIONS', style='dashed', color='black', fontsize='12', fontname='Arial Bold')

    c.node('loss_churn',
           'BCE Loss\nWeight: 2.0',
           fillcolor='#CFD8DC', shape='box')

    c.node('loss_ap',
           'MSE Loss\nWeight: 1.0',
           fillcolor='#CFD8DC', shape='box')

    c.node('loss_tp',
           'MSE Loss\nWeight: 0.1',
           fillcolor='#CFD8DC', shape='box')

    c.node('loss_total',
           'Combined Loss\n━━━━━━━━━━\n2.0×Churn + Ap + 0.1×Tp',
           fillcolor='#B0BEC5', shape='box', style='rounded,filled')

# =============================================================================
# Connections
# =============================================================================

# Input to processing
dot.edge('seq_input', 'lstm', label='50 timesteps')
dot.edge('lstm', 'lstm_hidden', label='extract last\nhidden state')
dot.edge('persona_input', 'persona_embed', label='10D')
dot.edge('global_input', 'concat', label='4D')

# To concatenation
dot.edge('lstm_hidden', 'concat', label='128D')
dot.edge('persona_embed', 'concat', label='32D')

# Shared layers
dot.edge('concat', 'fc1', label='164D')
dot.edge('fc1', 'fc2', label='256D')

# Multi-task heads
dot.edge('fc2', 'churn_fc1', label='128D')
dot.edge('churn_fc1', 'churn_fc2', label='64D')
dot.edge('churn_fc2', 'churn_out', label='1D')

dot.edge('fc2', 'ap_fc1', label='128D')
dot.edge('ap_fc1', 'ap_fc2', label='64D')
dot.edge('ap_fc2', 'ap_out', label='1D')

dot.edge('fc2', 'tp_fc1', label='128D')
dot.edge('tp_fc1', 'tp_fc2', label='64D')
dot.edge('tp_fc2', 'tp_out', label='1D')

# Loss connections
dot.edge('churn_out', 'loss_churn', style='dashed', color='red')
dot.edge('ap_out', 'loss_ap', style='dashed', color='red')
dot.edge('tp_out', 'loss_tp', style='dashed', color='red')

dot.edge('loss_churn', 'loss_total', style='dashed', color='red')
dot.edge('loss_ap', 'loss_total', style='dashed', color='red')
dot.edge('loss_tp', 'loss_total', style='dashed', color='red')

# Save outputs
dot.render('lstm_architecture_diagram', format='png', cleanup=True)
dot.render('lstm_architecture_diagram', format='svg', cleanup=True)
dot.save('lstm_architecture_diagram.dot')

print("✓ Saved: lstm_architecture_diagram.png (high-res image)")
print("✓ Saved: lstm_architecture_diagram.svg (scalable vector graphics)")
print("✓ Saved: lstm_architecture_diagram.dot (editable source)")

# =============================================================================
# Method 2: Matplotlib Layer Visualization (Simplified)
# =============================================================================

print("\n[2/3] Creating simplified layer diagram with Matplotlib...")

fig, ax = plt.subplots(figsize=(14, 18), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Helper function to draw a layer
def draw_layer(ax, x, y, width, height, neurons, label, color, annotation=''):
    """Draw a neural network layer as a box with circles representing neurons"""

    # Draw box
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(box)

    # Add label
    ax.text(x, y + height/2 + 0.3, label,
            fontsize=11, fontweight='bold', ha='center', va='bottom')

    # Add annotation (dimensions, etc.)
    if annotation:
        ax.text(x, y, annotation,
                fontsize=9, ha='center', va='center', style='italic')

    # Draw neurons as circles (sample representation)
    if neurons > 0:
        max_show = min(neurons, 8)  # Show max 8 neurons
        neuron_spacing = width / (max_show + 1)
        neuron_radius = 0.15

        for i in range(max_show):
            circle = plt.Circle(
                (x - width/2 + neuron_spacing * (i + 1), y - height/2 + 0.2),
                neuron_radius,
                color='white',
                ec='black',
                linewidth=1.5,
                zorder=10
            )
            ax.add_patch(circle)

        # Add "..." if more neurons
        if neurons > max_show:
            ax.text(x, y - height/2 + 0.2, '...',
                    fontsize=12, ha='center', va='center', fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    """Draw an arrow between layers"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        color=color,
        linewidth=2,
        mutation_scale=20,
        zorder=1
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
                fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# =============================================================================
# Draw Layers
# =============================================================================

y_pos = 19

# Input Layers
draw_layer(ax, 2, y_pos, 2.5, 1.2, 2, 'Event Sequence', '#E3F2FD',
           '[batch, 50, 2]\nGain + Time Delta')
draw_layer(ax, 5, y_pos, 2.5, 1.2, 4, 'Global Features', '#E3F2FD',
           '[batch, 4]\nRecency, Era, etc.')
draw_layer(ax, 8, y_pos, 2.5, 1.2, 5, 'Persona Features', '#E3F2FD',
           '[batch, 10]\nStatistics')

y_pos -= 2.5

# Persona Embedding
draw_layer(ax, 8, y_pos, 2.5, 1.0, 5, 'Persona Embedding', '#C8E6C9',
           '[batch, 32]\nLinear + ReLU')
draw_arrow(ax, 8, 19 - 0.6, 8, y_pos + 0.5, 'Embed', '#4CAF50')

y_pos -= 2.5

# LSTM
draw_layer(ax, 2, y_pos, 2.5, 1.5, 8, 'LSTM', '#FFF9C4',
           '[batch, 50, 128]\n2 layers\nDropout: 0.2')
draw_arrow(ax, 2, 19 - 0.6, 2, y_pos + 0.75, 'Sequence', '#FF9800')

y_pos -= 2.0

# LSTM Hidden State
draw_layer(ax, 2, y_pos, 2.5, 1.0, 8, 'LSTM Hidden', '#FFF9C4',
           '[batch, 128]\nLast hidden state')
draw_arrow(ax, 2, y_pos + 2.25, 2, y_pos + 0.5, 'Extract h_n[-1]', '#FF9800')

y_pos -= 2.0

# Concatenate
draw_layer(ax, 5, y_pos, 3.5, 1.2, 8, 'Concatenate', '#FFE0B2',
           '[batch, 164]\n128 + 4 + 32')
draw_arrow(ax, 2, y_pos + 1.5, 4.5, y_pos + 0.6, '128D', '#FF5722')
draw_arrow(ax, 5, 19 - 2.5 - 0.6, 5, y_pos + 0.6, '4D', '#FF5722')
draw_arrow(ax, 8, y_pos + 2.5 + 0.5, 6.5, y_pos + 0.6, '32D', '#FF5722')

y_pos -= 2.5

# FC1
draw_layer(ax, 5, y_pos, 3.5, 1.3, 8, 'FC Layer 1', '#FFCDD2',
           '[batch, 256]\nLinear + ReLU\nDropout: 0.3')
draw_arrow(ax, 5, y_pos + 2.0, 5, y_pos + 0.65, '164D → 256D', '#E91E63')

y_pos -= 2.5

# FC2
draw_layer(ax, 5, y_pos, 3.5, 1.3, 8, 'FC Layer 2', '#FFCDD2',
           '[batch, 128]\nLinear + ReLU\nDropout: 0.2')
draw_arrow(ax, 5, y_pos + 2.15, 5, y_pos + 0.65, '256D → 128D', '#E91E63')

y_pos -= 3.0

# Multi-Task Heads (3 branches)
# Churn Head
draw_layer(ax, 2, y_pos, 2.2, 1.0, 6, 'Churn FC1', '#E1BEE7',
           '[batch, 64]\nLinear + ReLU')
draw_arrow(ax, 4.5, y_pos + 3.15, 2.5, y_pos + 0.5, '128D → 64D', '#9C27B0')

y_pos_churn = y_pos - 1.8
draw_layer(ax, 2, y_pos_churn, 2.2, 1.0, 1, 'Churn Output', '#CE93D8',
           '[batch, 1]\nSigmoid')
draw_arrow(ax, 2, y_pos - 0.5, 2, y_pos_churn + 0.5, '64D → 1D', '#9C27B0')

y_pos_churn -= 1.5
ax.text(2, y_pos_churn, '🎯 Churn Probability [0,1]',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#BA68C8', alpha=0.8))

# Ap Head
draw_layer(ax, 5, y_pos, 2.2, 1.0, 6, 'Ap FC1', '#E1BEE7',
           '[batch, 64]\nLinear + ReLU')
draw_arrow(ax, 5, y_pos + 3.15, 5, y_pos + 0.5, '128D → 64D', '#9C27B0')

y_pos_ap = y_pos - 1.8
draw_layer(ax, 5, y_pos_ap, 2.2, 1.0, 1, 'Ap Output', '#CE93D8',
           '[batch, 1]\nSigmoid')
draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos_ap + 0.5, '64D → 1D', '#9C27B0')

y_pos_ap -= 1.5
ax.text(5, y_pos_ap, '🎯 Ap Threshold [0,1]',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#BA68C8', alpha=0.8))

# Tp Head
draw_layer(ax, 8, y_pos, 2.2, 1.0, 6, 'Tp FC1', '#E1BEE7',
           '[batch, 64]\nLinear + ReLU')
draw_arrow(ax, 5.5, y_pos + 3.15, 7.5, y_pos + 0.5, '128D → 64D', '#9C27B0')

y_pos_tp = y_pos - 1.8
draw_layer(ax, 8, y_pos_tp, 2.2, 1.0, 1, 'Tp Output', '#CE93D8',
           '[batch, 1]\nReLU')
draw_arrow(ax, 8, y_pos - 0.5, 8, y_pos_tp + 0.5, '64D → 1D', '#9C27B0')

y_pos_tp -= 1.5
ax.text(8, y_pos_tp, '🎯 Tp Window [0,∞) days',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#BA68C8', alpha=0.8))

# Add title
ax.text(5, 20.5, 'LSTM Temporal Model Architecture',
        fontsize=16, fontweight='bold', ha='center', va='top')

# Add parameter count
ax.text(5, 0.5, 'Total Parameters: ~1.5M  |  Device: CPU/GPU  |  Framework: PyTorch',
        fontsize=9, ha='center', va='bottom', style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#EEEEEE', alpha=0.8))

plt.tight_layout()
plt.savefig('lstm_architecture_layers.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: lstm_architecture_layers.png (layer visualization)")

# =============================================================================
# Method 3: Create Editable PowerPoint-Style Diagram Guide
# =============================================================================

print("\n[3/3] Creating customization guide...")

guide = """
================================================================================
LSTM ARCHITECTURE DIAGRAM - CUSTOMIZATION GUIDE
================================================================================

Generated Files:
1. lstm_architecture_diagram.png - Full neural network diagram (Graphviz)
2. lstm_architecture_diagram.svg - Scalable vector (editable in Inkscape/Illustrator)
3. lstm_architecture_diagram.dot - Source file (editable in Graphviz tools)
4. lstm_architecture_layers.png - Simplified layer visualization (Matplotlib)

================================================================================
HOW TO CUSTOMIZE
================================================================================

METHOD 1: Edit SVG File (Recommended for presentations)
--------------------------------------------------------
The .svg file can be edited in:
- Adobe Illustrator
- Inkscape (free): https://inkscape.org/
- Figma (import SVG)
- PowerPoint (Insert > Picture > SVG)

You can:
✓ Change colors
✓ Resize text
✓ Move boxes
✓ Add annotations
✓ Change fonts

METHOD 2: Edit DOT File (Best for structure changes)
-----------------------------------------------------
1. Install Graphviz: https://graphviz.org/download/
2. Open lstm_architecture_diagram.dot in text editor
3. Modify node/edge definitions
4. Regenerate: dot -Tpng lstm_architecture_diagram.dot -o output.png

DOT Syntax Quick Reference:
- Add node: node_id [label="Text", fillcolor="#COLOR"];
- Add edge: node1 -> node2 [label="weight"];
- Change color: fillcolor="#RRGGBB"
- Change shape: shape=box|circle|ellipse|polygon

METHOD 3: Edit Python Script
-----------------------------
1. Open visualize_lstm_architecture.py
2. Modify colors, labels, or structure
3. Run: python visualize_lstm_architecture.py

Example Customizations:
- Change layer colors: fillcolor='#NEWCOLOR'
- Add more neurons: Change neuron count in draw_layer()
- Modify dimensions: Change [batch, X] annotations
- Add new layers: Copy existing dot.node() calls

================================================================================
COLOR PALETTE USED
================================================================================

Input Layer:        #E3F2FD (Light Blue)
Embedding Layer:    #C8E6C9 (Light Green)
LSTM Layer:         #FFF9C4 (Light Yellow)
Concatenation:      #FFE0B2 (Light Orange)
FC Layers:          #FFCDD2 (Light Red)
Task Heads:         #E1BEE7, #CE93D8, #BA68C8 (Purple gradient)
Loss Functions:     #CFD8DC, #B0BEC5 (Grey)

To change colors globally, search and replace hex codes in .dot or .py files.

================================================================================
RECOMMENDED TOOLS
================================================================================

1. Graphviz Online Editor: https://dreampuf.github.io/GraphvizOnline/
   - Paste .dot file content
   - Edit in real-time
   - Download as PNG/SVG

2. Draw.io (diagrams.net): https://app.diagrams.net/
   - Import PNG as background
   - Trace over with editable shapes
   - Export to any format

3. Inkscape (Free): https://inkscape.org/
   - Open .svg file
   - Full vector editing
   - Export to PDF/PNG/EPS

4. PowerPoint/Keynote:
   - Insert .svg or .png
   - Add text boxes, arrows, annotations
   - Group elements for easy editing

================================================================================
LAYER DIMENSIONS REFERENCE
================================================================================

Input:
- Event Sequence:     [batch, 50, 2]
- Global Features:    [batch, 4]
- Persona Features:   [batch, 10]

Processing:
- Persona Embedding:  [batch, 32]
- LSTM Hidden:        [batch, 128]
- Concatenate:        [batch, 164]  (128 + 4 + 32)
- FC1:                [batch, 256]
- FC2:                [batch, 128]

Task Heads (each):
- FC1:                [batch, 64]
- Output:             [batch, 1]

Outputs:
- Churn Probability:  [0, 1]
- Ap Threshold:       [0, 1]
- Tp Window:          [0, ∞) days

================================================================================
TIPS FOR PRESENTATIONS
================================================================================

1. For slides: Use .svg file (scales perfectly at any resolution)
2. For papers: Use .png at 300 DPI (high quality)
3. For web: Use .svg or compress .png to <1MB

4. Highlight specific paths:
   - In Inkscape: Select arrows and change color/width
   - In PowerPoint: Draw over with colored shapes

5. Animate in PowerPoint:
   - Add entrance animations to each layer
   - Show data flow step-by-step

================================================================================
QUESTIONS?
================================================================================

To regenerate with different settings:
1. Edit visualize_lstm_architecture.py
2. Run: python visualize_lstm_architecture.py

For custom diagrams, contact the model developer or refer to:
- lstm_temporal_ap_tp.py (model implementation)
- LSTM_ARCHITECTURE_DIAGRAM.md (Mermaid diagrams)
"""

with open('DIAGRAM_CUSTOMIZATION_GUIDE.txt', 'w', encoding='utf-8') as f:
    f.write(guide)

print("✓ Saved: DIAGRAM_CUSTOMIZATION_GUIDE.txt")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print("\n📁 Generated Files:")
print("   1. lstm_architecture_diagram.png  - High-resolution diagram")
print("   2. lstm_architecture_diagram.svg  - Scalable vector (EDITABLE)")
print("   3. lstm_architecture_diagram.dot  - Graphviz source (EDITABLE)")
print("   4. lstm_architecture_layers.png   - Simplified visualization")
print("   5. DIAGRAM_CUSTOMIZATION_GUIDE.txt - How to customize")

print("\n🎨 Next Steps:")
print("   • Open .svg in Inkscape/Illustrator to customize colors and layout")
print("   • Open .dot in Graphviz editor to modify structure")
print("   • Import .png into PowerPoint for presentations")
print("   • Read DIAGRAM_CUSTOMIZATION_GUIDE.txt for detailed instructions")

print("\n✨ The diagrams show:")
print("   ✓ All layers with neuron counts")
print("   ✓ Connections between layers")
print("   ✓ Activation functions (ReLU, Sigmoid)")
print("   ✓ Dropout rates")
print("   ✓ Multi-task learning heads")
print("   ✓ Loss functions")
print("   ✓ Tensor dimensions at each stage")

print("\n" + "=" * 80)
