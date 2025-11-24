from random import randint, random, sample
import numpy as np
import os, sys
plt = None
try:
   import matplotlib
   # If there's no DISPLAY (typical headless), use Agg to allow saving files.
   no_display = os.environ.get('DISPLAY', '') == '' and sys.platform != 'darwin'
   if no_display:
      matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   print(f"matplotlib backend: {matplotlib.get_backend()}")
except Exception as e:
   print(f"matplotlib not available: {e}")
   plt = None

NUM_COEFFS = 6
BIT_LENGTH = 10     # Bits per coefficient (allows values from -512 to 511)
CHROM_LEN = NUM_COEFFS * BIT_LENGTH

TARGET_COEFFS = [25, 18, 31, -14, 7, -19]   # The polynomial coefficients

POP_SIZE = 40
GENERATIONS = 40
MUT_RATE = 0.01
CROSS_RATE = 0.9

def encode_coeff(value):
   """
   Encode integer into binary with sign bit."""
   # shift to fit unsigned representation
   shifted = value + 512
   return format(shifted, f'0{BIT_LENGTH}b')

def decode_coeff(bits):
   """
   Decode binary back into signed integer.
   """
   val = int(bits, 2)
   return val - 512

def encode_individual(coeffs):
   return ''.join(encode_coeff(c) for c in coeffs)

def decode_individual(binary):
   coeffs = []
   for i in range(0, CHROM_LEN, BIT_LENGTH):
      bits = binary[i:i+BIT_LENGTH]
      coeffs.append(decode_coeff(bits))
   return coeffs

def create_individual(length, min, max):
   """Create a member of the population as a binary-encoded individual.
   Each coefficient is an integer in [min, max] then encoded to binary.
   Returns a string of bits of length CHROM_LEN.
   """
   coeffs = [randint(min, max) for _ in range(length)]
   return encode_individual(coeffs)

def create_population(count, length, min, max):
   """
   Create a number of individuals (i.e. a population).
   count: the number of individuals in the population
   length: the number of values per individual
   min: the minimum possible value in an individual's list of values
   max: the maximum possible value in an individual's list of values
   """
   # Population will be a list of binary strings (each string length CHROM_LEN)
   return [create_individual(length, min, max) for x in range(count)]

def fitness(individual, target):
   """
   Determine the fitness of an individual represented as a binary string.
   Fitness approaches 1 as the decoded coefficients get closer to the target list.
   """
   # decode individual bits into coefficient list
   if isinstance(individual, str):
      coeffs = decode_individual(individual)
   else:
      coeffs = individual

   error = sum(abs(coeffs[i] - target[i]) for i in range(len(target)))
   return 1 / (1 + error)

def roulette_wheel_selection(population, target):
   """
   Select one individual proportional to fitness.
   """
   fitnesses = [fitness(individual, target) for individual in population]
   total_fitness = sum(fitnesses)

   #Normalised fitness probabilities
   probabilities = [f / total_fitness for f in fitnesses]

   #Cumulative probabilities
   cumulative = np.cumsum(probabilities)

   #Spin wheel
   r = random()

   #Return first index where cumulative prob >= r
   for i, cumulative_prob in enumerate(cumulative):
      if r <= cumulative_prob:
         return population[i]

   return population[-1]

def tournament_selection(population, target, k=3):
   competitors = sample(population, k)
   def error_for(ind):
      coeffs = decode_individual(ind) if isinstance(ind, str) else ind
      return sum(abs(coeffs[i] - target[i]) for i in range(len(target)))
   return min(competitors, key=error_for)

def crossover(parent1, parent2):
   """
   Single-point crossover between two binary-string parents.
   """
   # parent1/parent2 are binary strings of length CHROM_LEN
   # Use crossover aligned to coefficient boundaries (better preserves genes)
   if NUM_COEFFS > 1:
      gene_cut = randint(1, NUM_COEFFS - 1)
      cut = gene_cut * BIT_LENGTH
   else:
      cut = randint(1, CHROM_LEN - 1)
   return parent1[:cut] + parent2[cut:]

def mutation(individual, mutation_rate):
   """
   Bit-flip mutation on a binary-encoded individual.
   Each bit has independent probability `mutation_rate` to be flipped.
   Returns a new binary string.
   """
   bits = list(individual)
   for i in range(len(bits)):
      if random() < mutation_rate:
         bits[i] = '1' if bits[i] == '0' else '0'
   return ''.join(bits)


def creep_mutation(individual, creep_rate, min_value, max_value):
   """
   Per-coefficient small integer mutation (creep): decode the individual,
   for each coefficient with probability `creep_rate` add or subtract 1,
   clamp to [min_value, max_value], then re-encode and return binary string.
   """
   coeffs = decode_individual(individual)
   for i in range(len(coeffs)):
      if random() < creep_rate:
         # small +/-1 change
         coeffs[i] += -1 if random() < 0.5 else 1
         coeffs[i] = max(min_value, min(max_value, coeffs[i]))
   return encode_individual(coeffs)

# --- Schema-theorem helpers & instrumentation ---
def matches_schema(individual_bits, schema):
   """Return True if binary string `individual_bits` matches `schema` with '*' wildcards."""
   return all(s == '*' or s == b for b, s in zip(individual_bits, schema))

def schema_order(schema):
   """Number of fixed bits in schema (non '*')."""
   return sum(1 for c in schema if c != '*')

def schema_defining_length(schema):
   """Distance between leftmost and rightmost fixed positions (0 if order <=1)."""
   fixed_idx = [i for i, c in enumerate(schema) if c != '*']
   if len(fixed_idx) <= 1:
      return 0
   return fixed_idx[-1] - fixed_idx[0]

def make_single_bit_schemata(encoded_target):
   """Return schemata that fix each single bit to the target bit (order=1)."""
   schemata = []
   for i, bit in enumerate(encoded_target):
      s = ['*'] * len(encoded_target)
      s[i] = bit
      schemata.append(''.join(s))
   return schemata

def make_full_coeff_schemata(encoded_target):
   """Return schemata that fix all bits of each coefficient to their target value."""
   schemata = []
   for coeff in range(NUM_COEFFS):
      start = coeff * BIT_LENGTH
      end = start + BIT_LENGTH
      s = ['*'] * len(encoded_target)
      s[start:end] = list(encoded_target[start:end])
      schemata.append(''.join(s))
   return schemata

def holland_prediction(m_ht, f_h, f_bar, p_c, p_m, d_h, o_h, l=CHROM_LEN):
   """Return Holland's lower bound estimate for m(H, t+1).
   Uses bound: m' >= m * (f_h/f_bar) * (1 - p_c * d/(l-1)) * (1-p_m)^{o}
   """
   if f_bar == 0 or m_ht == 0:
      return 0.0
   term_sel = f_h / f_bar if f_bar != 0 else 0.0
   term_cross = max(0.0, 1.0 - p_c * (d_h / (l - 1))) if l > 1 else 1.0
   term_mut = (1.0 - p_m) ** o_h
   return m_ht * term_sel * term_cross * term_mut

def compute_schema_stats(population, schemata, target):
   """Compute counts and avg fitness for a set of schemata on `population`.
   Returns dict schema->stats
   """
   fitnesses = [fitness(ind, target) for ind in population]
   stats = {}
   for s in schemata:
      matches = [i for i, ind in enumerate(population) if matches_schema(ind, s)]
      m_ht = len(matches)
      f_h = sum(fitnesses[i] for i in matches) / m_ht if m_ht > 0 else 0.0
      o_h = schema_order(s)
      d_h = schema_defining_length(s)
      stats[s] = dict(count=m_ht, avg_fitness=f_h, order=o_h, defining_length=d_h)
   return stats

def print_schema_comparison(stats_curr, stats_next, target, p_c, p_m):
   """Print a compact comparison for tracked schemata between generations.
   `stats_curr` contains counts and avg_fitness for generation t; `stats_next` for t+1.
   This function computes Holland's predicted m' and prints observed vs predicted.
   """
   # compute f_bar from current stats (need population-level average; approximate from schemata?)
   # Better: compute f_bar externally and include in stats if desired. Here we expect caller to compute f_bar.
   print("--- Schemata summary (count_curr, predicted_next, count_next, order,d) ---")
   for s, st in stats_curr.items():
      m_ht = st['count']
      f_h = st.get('avg_fitness', 0.0)
      o_h = st['order']
      d_h = st['defining_length']
      f_bar = st.get('f_bar', None)
      if f_bar is None:
         # if f_bar not provided, skip prediction
         pred = None
      else:
         pred = holland_prediction(m_ht, f_h, f_bar, p_c, p_m, d_h, o_h)
      m_next = stats_next.get(s, {}).get('count', 0)
      if pred is None:
         print(f"o={o_h},d={d_h} curr={m_ht} next={m_next}")
      else:
         print(f"o={o_h},d={d_h} curr={m_ht} pred={pred:.2f} next={m_next}")

def run_ga(mutation_rate, crossover_rate,
         generations=200, population_size=1000, instrument=False, plot=True):
   """
   Provide an external interface to run the GA.
   """

   min_value = -20
   max_value = 40
   target = TARGET_COEFFS
   i_length = len(target)

   # Create initial population of binary strings
   population = create_population(population_size, i_length, min_value, max_value)

   # prepare instrumentation if requested
   if instrument:
      encoded_target = encode_individual(target)
      single_bit_schemata = make_single_bit_schemata(encoded_target)
      coeff_schemata = make_full_coeff_schemata(encoded_target)
      # track only coefficient-level schemata (one schema per coefficient)
      schemata_to_track = coeff_schemata
      schema_history = {s: {'m_curr': [], 'pred': [], 'm_next': []} for s in schemata_to_track}
   else:
      schemata_to_track = []
      schema_history = {}

   for gen in range(generations):

      # Check for exact match (binary-encoded target)
      encoded_target = encode_individual(target)
      if encoded_target in population:
         if instrument and plot and plt is not None and schemata_to_track:
            n = len(schemata_to_track)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            axes = np.atleast_2d(axes)
            for idx, s in enumerate(schemata_to_track):
               r = idx // cols
               c = idx % cols
               ax = axes[r, c]
               hist = schema_history[s]
               gens = list(range(len(hist['m_curr'])))
               ax.plot(gens, hist['m_curr'], label='Curr Gen Count', marker='o')
               ax.plot(gens, hist['pred'], label='Pred Next Gen Count', marker='x')
               ax.plot(gens, hist['m_next'], label='Next Gen Count', marker='s')
               ax.set_title(f'Coeff schema {idx+1}')
               ax.set_xlabel('Generation')
               ax.set_ylabel('Count')
               ax.legend()
            # hide unused axes
            total_axes = rows * cols
            for idx in range(n, total_axes):
               r = idx // cols
               c = idx % cols
               axes[r, c].axis('off')
            plt.tight_layout()
            try:
               outpath = os.path.abspath('schema_theorem_plot.png')
               fig.savefig(outpath)
               print(f"Saved schemata plot to {outpath}")
               # Show only if backend supports interactive display
               try:
                  backend = matplotlib.get_backend()
                  if backend.lower() != 'agg':
                     print(f"Attempting to display plot using backend {backend}")
                     plt.show()
                  else:
                     print(f"Backend is '{backend}', skipping interactive show(). Open the saved file to view the plot.")
               except Exception as e:
                  print(f"Could not display plot window: {e}")
            except Exception as e:
               print(f"Failed to save plot: {e}")
         elif instrument and plot and plt is None:
            print('matplotlib not available — skipping plot.')
         return gen

      # Keep best individual (elitism)
      def error_for(ind):
         coeffs = decode_individual(ind) if isinstance(ind, str) else ind
         return sum(abs(coeffs[i] - target[i]) for i in range(len(target)))

      # Keep top-2 elites to preserve good solutions
      sorted_pop = sorted(population, key=error_for)
      elites = sorted_pop[:2]
      best = elites[0]
      new_population = elites[:]

      # If instrumenting, compute current schema stats and f_bar
      if instrument and schemata_to_track:
         fitnesses = [fitness(ind, target) for ind in population]
         f_bar = sum(fitnesses) / len(fitnesses)
         stats_curr = compute_schema_stats(population, schemata_to_track, target)
         for s in stats_curr:
            stats_curr[s]['f_bar'] = f_bar
      else:
         stats_curr = {}

      # Fill the rest of the population using tournament selection
      for _ in range(population_size - len(new_population)):
         parent1 = tournament_selection(population, target, k=3)
         parent2 = tournament_selection(population, target, k=3)

         # Apply crossover
         if random() < crossover_rate:
               child = crossover(parent1, parent2)
         else:
               child = parent1[:]

         # Apply mutation (bit-flip)
         child = mutation(child, mutation_rate)
         # Apply small per-coefficient creep to help escape single-off errors
         child = creep_mutation(child, creep_rate=0.05, min_value=min_value, max_value=max_value)
         new_population.append(child)

      population = new_population

      # If instrumenting, compute next stats and record predictions/observations
      if instrument and schemata_to_track:
         stats_next = compute_schema_stats(population, schemata_to_track, target)
         for s, st in stats_curr.items():
            m_ht = st['count']
            f_h = st.get('avg_fitness', 0.0)
            o_h = st['order']
            d_h = st['defining_length']
            f_bar = st.get('f_bar', 0.0)
            pred = holland_prediction(m_ht, f_h, f_bar, crossover_rate, mutation_rate, d_h, o_h)
            schema_history[s]['m_curr'].append(m_ht)
            schema_history[s]['pred'].append(pred)
            schema_history[s]['m_next'].append(stats_next.get(s, {}).get('count', 0))

      # Optionally, print best fitness per generation for monitoring
      if gen % 10 == 0 or gen == generations - 1:
         best_err = error_for(best)
         print(f"gen={gen} best_error={best_err}")

   # after run, if instrument and plotting enabled, create a PNG comparing predicted vs observed

   return generations  # If not found

if __name__ == "__main__":
   #Example run
   # Increase generations and mutation rate a bit to improve search
   gens = run_ga(mutation_rate=0.001, crossover_rate=0.9, generations=500, population_size=POP_SIZE, instrument=True, plot=True)
   print(f"GA completed in {gens} generations.")