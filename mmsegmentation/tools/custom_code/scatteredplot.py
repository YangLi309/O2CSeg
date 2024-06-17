import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import matplotlib.lines as mlines
import math
import matplotlib
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams.update({'font.size': 24})

# Model data with log FPS
model = ["SAN", "SAN+O2CD$_{\mathrm{PE+LPC}}$", "STDC+O2CD$_{\mathrm{CMKD}}$", "Segmenter+O2CD$_{\mathrm{CMKD}}$"]
model2desc= {
    "SAN": "SAN: 436.7M",
    "SAN+O2CD$_{\mathrm{PE+LPC}}$": "SAN+O2CD$_{\mathrm{CMKD}}$: 436.7M",
    "STDC+O2CD$_{\mathrm{CMKD}}$": "STDC+O2CD$_{\mathrm{CMKD}}$: 8.3M",
    "Segmenter+O2CD$_{\mathrm{CMKD}}$": "Segmenter+O2CD$_{\mathrm{CMKD}}$: 7.3M"
}

# log_fps = [1.33, 0.123851641, 0.07918124605, 1.738225448, 1.901022173]
fps = [1.33, 1.2, 79.62, 54.73, 79.62, 54.73]
miou = [43.1, 51.52, 47.62, 51.06, 71.61, 73.76]
colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkred']
# Adjusting the plot with bigger circle markers and ensuring text does not overflow the axes. Increased font size.
plt.figure(figsize=(12, 8))
for i, txt in enumerate(model):
    plt.scatter(fps[i], miou[i], s=500, c=colors[i], alpha=0.5, edgecolors='w', linewidths=2)  # Bigger circle markers
    # Adjust text position to prevent overflow
    text_x = fps[i]
    text_y = miou[i]
    if text_x < min(fps) + (max(fps) - min(fps)) * 0.1:
        ha = 'left'
    else:
        ha = 'right'

    if text_y < min(miou) + (max(miou) - min(miou)) * 0.1:
        va = 'bottom'
    else:
        va = 'top'

    # if txt == "Segmenter+O2CD$_{\mathrm{CMKD}}$":
    #     ha = 'left'

    plt.text(text_x, text_y, txt, ha=ha, va=va, color='black')

legend_elements = [mlines.Line2D([0], [0], color=color, marker='o', linestyle='None',
                                  markersize=10, label=model2desc[model_name]) for model_name, color in zip(model, colors)]
plt.legend(handles=legend_elements, loc='lower center', fontsize=26, title='#Params', title_fontsize=26)
# plt.title('FPS vs. mIoU with Model Annotations', fontsize=14)
plt.xlabel('Frames Per Second (FPS)', fontsize=26)
plt.ylabel('Mean IoU (%)', fontsize=26)
plt.grid(True)
plt.tight_layout()  # Adjust layout to make room for the annotation
# plt.show()
# plt.xscale('log')
# adjust_text_positions(log_fps, miou, [ax.texts[i] for i in range(len(model))], fig, ax)

# plt.xlabel('Inference Speed (Log FPS)')
# plt.ylabel('Mean IoU(%)')
# plt.title('Model Performance: Log FPS vs. mIoU')

# plt.show()
plt.savefig('fps_acc.png', dpi=300, bbox_inches='tight')