import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

# Load templates (assumes variable name is "templates" inside the .mat file)
templates = np.load('Coursework/Coursework C/templates.npy', allow_pickle=True)

templates = np.array([tmpl[:150] for tmpl in templates])

# Plot all templates
plt.figure(figsize=(10, 6))
for i, tmpl in enumerate(templates):
    plt.plot(tmpl, label=f"Template {i+1}")

plt.title("All Matched Filter Templates")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()