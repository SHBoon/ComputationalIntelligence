from random import randint, random
import numpy as np

def create_individual(length, min, max):
   """
   Create a single individual.
   """

   individual = [randint(min, max) for x in range(length)]

   return individual

def create_population(count, length, min, max):
   """
   Create a number of individuals (i.e. a population).
   """

   return [create_individual(length, min, max) for x in range(count)]

def fitness(individual, target):
   """
   Determine the fitness of an individual.
   """

   #Fitness approaches 1 as the individual gets closer to the target
   diff = abs(target - sum(individual))
   
   return 1 / (1 + diff)

def roulette_wheel_selection(population, target):
   """
   Select one individual proportional to fitness.
   """

   fitnesses = [fitness(individual, target) for individual in population]
   total_fitness = sum(fitnesses)

   probabilities = [f / total_fitness for f in fitnesses]

   cumulative = np.cumsum(probabilities)

   r = random()

   for i, cumulative_prob in enumerate(cumulative):
      if r <= cumulative_prob:
         return population[i]

   return population[-1]

def crossover(parent1, parent2):
   """
   Arithmetic crossover between two numbers.
   """

   return [parent1[i] if random() < 0.5 else parent2[i] for i in range(len(parent1))]

def mutation(individual, mutation_rate, min_value, max_value, pop_range):
   """
   Add random small noise with some probability.
   """
   for i in range(len(individual)):
      if random() < mutation_rate:
         noise = randint(-pop_range, pop_range) // 10
         individual[i] = max(min_value, min(max_value, individual[i] + noise))
   return individual


def run_ga(mutation_rate, crossover_rate, generations=100, population_size=50):
   """
   Provide an external interface to run the GA.
   """
   min_value = 0
   max_value = 1000
   target = 550
   i_length = 1
   pop_range = max_value - min_value

   population = create_population(population_size, i_length, min_value, max_value)

   for gen in range(generations):
      if any(sum(ind) == target for ind in population):
         return gen

      new_population = []

      #Elitism
      best_individual = max(population, key=lambda ind: fitness(ind, target))
      new_population.append(best_individual.copy())

      for _ in range(population_size - 1):
         parent1 = roulette_wheel_selection(population, target)
         parent2 = roulette_wheel_selection(population, target)

         if random() < crossover_rate:
               child = crossover(parent1, parent2).copy()
         else:
               child = parent1.copy()

         child = mutation(child, mutation_rate, min_value, max_value, pop_range)
         new_population.append(child)

      population = new_population

   return generations

if __name__ == "__main__":
   #Internal run
   gens = run_ga(mutation_rate=0.4, crossover_rate=0.99)
   print(f"GA completed in {gens} generations.")