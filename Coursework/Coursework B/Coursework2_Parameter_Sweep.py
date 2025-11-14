import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Coursework2_Polynomial_Coefficients as GA_poly   # <-- YOUR FILE
import Coursework2_Individual_Number as GA_ind  # <-- YOUR FILE

# ------------------------------------
# Run GA once and return generations
# ------------------------------------
def run_once(mutation_rate, crossover_rate):
   return GA_poly.run_ga(
      mutation_rate=mutation_rate,
      crossover_rate=crossover_rate
   )

# ------------------------------------
# Parameter sweep setup
# ------------------------------------
mutation_values = np.linspace(0.01, 1.0, 20)  # 20 values from 0.01 to 1.00
crossover_values = np.linspace(0.01, 1.0, 20)  # 20 values from 0.01 to 1.00

M_vals = []
C_vals = []
G_vals = []

# ------------------------------------
# Run 10 iterations for each pair
# ------------------------------------
for m in mutation_values:
   for c in crossover_values:
      gens = []
      for iteration in range(10):
         print(f"Running Mutation {m:.2f}, Crossover {c:.2f}, Iteration {iteration+1}")
         gens.append(run_once(m, c))
      avg_gens = np.mean(gens)

      M_vals.append(m)
      C_vals.append(c)
      G_vals.append(avg_gens)

      print(f"Mutation {m:.2f}, Crossover {c:.2f} → Avg Gens = {avg_gens:.1f}")

# ------------------------------------
# 3D Surface Plot
# ------------------------------------

M_vals = np.array(M_vals)
C_vals = np.array(C_vals)
G_vals = np.array(G_vals)

# Convert 1D sweep results into 2D grids
M_grid, C_grid = np.meshgrid(
    mutation_values,
    crossover_values
)

# Reshape Z (avg generations) to match grid
G_grid = G_vals.reshape(len(mutation_values), len(crossover_values))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(M_grid, C_grid, G_grid,
                       cmap="viridis",
                       edgecolor='none')

ax.set_xlabel("Mutation Rate")
ax.set_ylabel("Crossover Rate")
ax.set_zlabel("Average Generations")

fig.colorbar(surf, shrink=0.2, aspect=10)

plt.show()