import numpy as np
from numpy import random

# Generic Genetic Algorithm from scratch, very general with simple objective function
# STEP 1: Create a population of random bitstrings
# STEP 2: Enumerate over a fixed number of iterations
# STEP 3: Use an objective function to get a fitness score
# STEP 4: Select parents to create children; create selected parent list
# STEP 5: Use this to create child list
# To do this, perform crossovers and mutations and create the new generation
# Keep looping these generations for total iterations

# Hyperparamters 
n_bits = 20 # binary bitstrings of length 20
n_pop = 100 # 100 bitstrings
n_iter = 100 # 100 generations
r_cross = 0.9 # crossover rate
r_mut = 1.0 / float(n_bits) # mutation rate

# tournament selection
# choosing k is a bit trial and error, just check the amount of gens
# other kinds of selection exist, this is the most basic
def selection(population, scores, k=3):
    # first random selection; pick a random bitstring from population
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        # check if better (e.g. perform a tournament)
        # do this k amount of times; k is low because this is efficient for large problems
        # basically, just compare a couple random inviduals and then, if ix is better, 
        # set selection_ix = to it to improve the quality of our population
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

# crossover two parents to create two children
# sometimes other methods of crossing the lists can be used, this is the most basic
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    # .rand() gives a probability between [0, 1)
    if np.random.rand() < r_cross: 
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]
    
# mutation operator
# usually doesn't need to be adapted, pretty straightfoward
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        # 10% chance for .rand() to be less than r_mut
        if random.rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

# objective function
def onemax(x):
    return -sum(x)
    
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring; n_pop bitstrings of length n_bits
    pop = [random.randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # keep track of best solution
    # get a baseline evaluation from first individual
    best, best_eval = 0, objective(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        # look at every bitstring, compare score with current best eval
        for i in range(n_pop):
            if scores[i] < best_eval: # minimizing
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        #select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2): # iterates by 2 each time; adjacent parents could be too similar 
            # get selected parents in pairs, so 50 iterations for 2 selections each = new gen of 100
            p1, p2 = selected[i], selected[i+1]
            #p1.tolist()
            #p2.tolist() 
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

if __name__ == "__main__":
    # perform the genetic algorithm search
    best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
    print('Done!')
    print('f(%s) = %f' % (best, score))