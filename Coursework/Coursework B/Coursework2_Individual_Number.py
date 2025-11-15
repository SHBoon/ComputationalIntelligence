from random import randint, random
import numpy as np

def create_population(count, min, max):
   """
   Create a number of individuals (i.e. a population).
   """
   return [randint(min, max) for x in range(count)]

def fitness(individual, target):
   """
   Determine the fitness of an individual.
   """
   # Fitness approaches 1 as the individual gets closer to the target
   return 1 / (1 + abs(target - individual))

def roulette_wheel_selection(population, target):
   """
   Select one individual proportional to fitness.
   """
   fitnesses = [fitness(individual, target) for individual in population]
   total_fitness = sum(fitnesses)

   # Normalised fitness probabilities
   probabilities = [f / total_fitness for f in fitnesses]

   # Cumulative probabilities
   cumulative = np.cumsum(probabilities)

   # Spin wheel
   r = random()

   # Return first index where cumulative prob >= r
   for i, cumulative_prob in enumerate(cumulative):
      if r <= cumulative_prob:
         return population[i]

   return population[-1]

def crossover(parent1, parent2):
   """Arithmetic crossover between two numbers."""
   alpha = random()
   child = int(alpha * parent1 + (1 - alpha) * parent2)
   return child

def mutation(individual, mutation_rate, min_value, max_value, pop_range):
   """Add random small noise with some probability."""
   if random() < mutation_rate:
      noise = randint(-pop_range, pop_range) // 2
      individual = max(min_value, min(max_value, individual + noise))
   return individual



# ---- New run_ga function ----
def run_ga(mutation_rate, crossover_rate,
           generations=20, population_size=50):
    min_value = 0
    max_value = 1000
    target = 550
    pop_range = max_value - min_value

    population = create_population(population_size, min_value, max_value)

    for gen in range(generations):
        if target in population:
            return gen

        new_population = [1]
      #   new_population[0] = (min(population, key=lambda ind: abs(ind - target)))

        for _ in range(population_size):
            parent1 = roulette_wheel_selection(population, target)
            parent2 = roulette_wheel_selection(population, target)

            if random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1

            child = mutation(child, mutation_rate, min_value, max_value, pop_range)
            new_population.append(child)

        population = new_population

    return generations

if __name__ == "__main__":
    # Example run
    gens = run_ga(mutation_rate=0.1, crossover_rate=0.7)
    print(f"GA completed in {gens} generations.")