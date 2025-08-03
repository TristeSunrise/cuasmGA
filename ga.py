import random
from sass_kernel import SassKernel
from sassgen import extract_kernel_sass, write_sass_file

POP_SIZE = 10   #population size

class Individual:
    def __init__(self, kernel_section):
        self.sass = kernel_section  # List of instructions (strings)
        self.fitness = None

# def evaluate_fitness(individual):
#     cubin = write_sass_file(individual.sass)  # write back to cubin
#     run_benchmark(cubin)
# def run_benchmark(cubin):
#     return 


def initialize_population(original_kernel_section): 
    # Initialize population
    population = []
    for _ in range(POP_SIZE):
        shuffled = random.sample(original_kernel_section, len(original_kernel_section))
        population.append(Individual(shuffled))
    return population

def tournament_selection(population, k = 4):  #to reduce the noise, try rank-based selection later
    contenders = random.sample(population, k)
    best = min(contenders, key=lambda x: x.fitness)
    return best

def crossover(parent1 : Individual, parent2 : Individual):
    size = parent1.sass.size
    #create selection point
    point1 = random.randint(0, size-2)
    point2 = random.randint(point1+1, size-1)
    #create child1
    child1_sass = [None] * size
    child1_sass[point1:point2] = parent1.sass[point1:point2]
    remainingp2 = [x for x in parent2.sass if x not in child1_sass[point1:point2]]
    fill_idx = 0
    for i in range(size):
        if child1_sass[i] is None:
            child1_sass[i] = remainingp2[fill_idx]
            fill_idx += 1
    #create child2
    child2_sass = [None] * size
    child2_sass[point1:point2] = parent2.sass[point1:point2]
    remainingp1 = [x for x in parent1.sass if x not in child2_sass[point1:point2]]
    fill_idx = 0
    for i in range(size):
        if child2_sass[i] is None:
            child2_sass[i] = remainingp1[fill_idx]
            fill_idx += 1


    return Individual(child1_sass), Individual(child2_sass)


if __name__ == '__main__':
    sass,kernel_section = extract_kernel_sass("cuasmrl_kernel_0d1d2_0.pkl")
    pure_kernel = SassKernel(sass,kernel_section)._get_kernel()
    print(len(pure_kernel))


