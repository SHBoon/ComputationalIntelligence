import random
import numpy as np


NUM_COEFFS = 6
BIT_LENGTH = 10     # Bits per coefficient (allows values from -512 to 511)
CHROM_LEN = NUM_COEFFS * BIT_LENGTH

TARGET_COEFFS = [25, 18, 31, -14, 7, -19]   # The polynomial coefficients

POP_SIZE = 40
GENERATIONS = 40
MUT_RATE = 0.01
CROSS_RATE = 0.9

def encode_coeff(value):
   """Encode integer into binary with sign bit."""
   # shift to fit unsigned representation
   shifted = value + 512
   return format(shifted, f'0{BIT_LENGTH}b')

def decode_coeff(bits):
   """Decode binary back into signed integer."""
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

def fitness(binary):
   """
   Fitness = inverse error between decoded coefficients and target.
   """
   coeffs = decode_individual(binary)
   error = sum(abs(coeffs[i] - TARGET_COEFFS[i]) for i in range(NUM_COEFFS))
   return 1 / (1 + error)

def roulette_selection(pop):
   """
   
   """
   fits = [fitness(ind) for ind in pop]
   total = sum(fits)
   probs = [f / total for f in fits]
   cum = np.cumsum(probs)
   r = random.random()
   for i, cp in enumerate(cum):
      if r <= cp:
         return pop[i]
   return pop[-1]

def crossover(parent1, parent2):
   if random.random() > CROSS_RATE:
      return parent1

   point = random.randint(1, CHROM_LEN - 1)
   return parent1[:point] + parent2[point:]

def mutate(binary):
   bits = list(binary)
   for i in range(len(bits)):
      if random.random() < MUT_RATE:
         bits[i] = '1' if bits[i] == '0' else '0'
   return ''.join(bits)

def matches_schema(individual, schema):
   """
   Schema example: 10*1**0*
   """
   return all(schema[i] == '*' or schema[i] == individual[i]
            for i in range(len(schema)))

def count_schema(pop, schema):
   return sum(1 for ind in pop if matches_schema(ind, schema))

def run_schema_demo(schema):
   """
   Run GA and show how schema evolves under selection, crossover, mutation.
   """
   # Initial population
   population = [
      encode_individual([random.randint(-20, 30) for _ in range(NUM_COEFFS)])
      for _ in range(POP_SIZE)
   ]

   print("\n===== SCHEMA THEOREM DEMO =====")
   print(f"Schema: {schema}")
   print("--------------------------------\n")

   for gen in range(GENERATIONS):

      #Selection
      before_selection = count_schema(population, schema)
      matching_before = [ind for ind in population if matches_schema(ind, schema)]
      avg_fit_before = np.mean([fitness(ind) for ind in matching_before]) if matching_before else 0

      mating_pool = [roulette_selection(population) for _ in range(POP_SIZE)]

      after_selection = count_schema(mating_pool, schema)
      matching_sel = [ind for ind in mating_pool if matches_schema(ind, schema)]
      avg_fit_sel = np.mean([fitness(ind) for ind in matching_sel]) if matching_sel else 0

      #Crossover
      offspring = []
      for i in range(0, POP_SIZE, 2):
         p1 = mating_pool[i]
         p2 = mating_pool[(i + 1) % POP_SIZE]
         c1 = crossover(p1, p2)
         c2 = crossover(p2, p1)
         offspring.append(c1)
         offspring.append(c2)

      after_crossover = count_schema(offspring, schema)
      matching_cross = [ind for ind in offspring if matches_schema(ind, schema)]
      avg_fit_cross = np.mean([fitness(ind) for ind in matching_cross]) if matching_cross else 0

      #Mutation
      mutated = [mutate(ind) for ind in offspring]

      after_mutation = count_schema(mutated, schema)
      matching_mut = [ind for ind in mutated if matches_schema(ind, schema)]
      avg_fit_mut = np.mean([fitness(ind) for ind in matching_mut]) if matching_mut else 0

      print(
         f"Generation {gen:02d} |\n"
         f"  Schema Count: Sel {before_selection}->{after_selection}, "
         f"Cross {after_selection}->{after_crossover}, "
         f"Mut {after_crossover}->{after_mutation}\n"
         f"  Avg Fitness:  Sel {avg_fit_before:.4f}->{avg_fit_sel:.4f}, "
         f"Cross {avg_fit_sel:.4f}->{avg_fit_cross:.4f}, "
         f"Mut {avg_fit_cross:.4f}->{avg_fit_mut:.4f}\n"
      )

      population = mutated

EXAMPLE_SCHEMA = "1*0***1***********************************************"

if __name__ == "__main__":
   run_schema_demo(EXAMPLE_SCHEMA)