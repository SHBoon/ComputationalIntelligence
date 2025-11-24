from random import randint, random, sample
import numpy as np

def create_individual(length, min, max):
   """ Create a member of the population."""
   return [randint(min, max) for x in range(length)]

def create_population(count, length, min, max):
   """
   Create a number of individuals (i.e. a population).
   count: the number of individuals in the population
   length: the number of values per individual
   min: the minimum possible value in an individual's list of values
   max: the maximum possible value in an individual's list of values
   """
   return [create_individual(length, min, max) for x in range(count)]

def fitness(individual, target):
   """
   Determine the fitness of an individual.
   """
   #Fitness approaches 1 as the individual gets closer to the target
   error = sum(abs(individual[i] - target[i]) for i in range(len(target)))
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
   return min(competitors, key=lambda ind:
      sum(abs(ind[i] - target[i]) for i in range(len(target))))

def crossover(parent1, parent2):
   """
   Single point crossover between two parents.
   """
   #Choose random crossover point
   cut = randint(1, len(parent1)-1)
   return parent1[:cut] + parent2[cut:]

def mutation(individual, mutation_rate, min_value, max_value, pop_range):
   """
   Mutate an individual by adding small random noise to one gene.
   """
   #Mutate with given probability
   if random() < mutation_rate:
      pos = randint(0, len(individual)-1)
      noise = randint(-pop_range, pop_range) // 5
      individual[pos] += noise
      individual[pos] = max(min_value, min(max_value, individual[pos]))
   return individual

def run_ga(mutation_rate, crossover_rate,
         generations=200, population_size=100):
   """
   Provide an external interface to run the GA.
   """

   min_value = -20
   max_value = 40
   target = [25, 18, 31, -14, 7, 19]
   i_length = len(target)
   pop_range = max_value - min_value

   population = create_population(population_size, i_length, min_value, max_value)

   for gen in range(generations):

      #Check for exact match
      if target in population:
         return gen

      new_population = [1]
      new_population[0] = (min(population, key=lambda ind: sum(abs(ind[i] - target[i]) for i in range(len(target)))))

      for _ in range(population_size):
         parent1 = roulette_wheel_selection(population, target)
         parent2 = roulette_wheel_selection(population, target)

         #Apply crossover
         if random() < crossover_rate:
               child = crossover(parent1, parent2)
         else:
               child = parent1[:]

         #Apply mutation
         child = mutation(child, mutation_rate, min_value, max_value, pop_range)
         new_population.append(child)

      population = new_population

   return generations  #If not found

if __name__ == "__main__":
   #Example run
   gens = run_ga(mutation_rate=0.4, crossover_rate=0.65)
   print(f"GA completed in {gens} generations.")