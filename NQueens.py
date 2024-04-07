import numpy as np
from numpy import random
from gen_alg_scratch import selection 


# PSEUDOCODE
    # Use permutations instead of bistrings
    # i.e. pass in the number of queens (N) to the gen alg function, and call .permutation to
    # create random lists of permutations up to N, where each number in the row vector represents
    # the row of the queens placement, where each column of the row vector represents the column
    # EXAMPLE: [2, 3, 0, 1] for N = 4: 
        # Queen in first column, 2nd row
        # Queen in second column, 4th row
        # Queen in third column, first row
        # queen in fourth column, 2nd row   
        #""" board = [[0, 0, 1, 0],
        #             [0, 0, 0, 1],
        #             [1, 0, 0, 0],
        #             [0, 1, 0, 0]
        #             ] """
        # here, there is a lot of collisions obv, so this is a bad perm
    # Our goal is to evolve the gens by minimizing collisions; that is the obj function

# THINGS TO EXPERIMENT WITH:
# Once convergence is reached, RESET the gene pool, or a large part, to try and find a different solution
# It tends to get stuck on certain solutions and then purely relies on the low mutation chance for finding
# new solutions, for n > 10
# Also, parameters in terms of n might be better than hard coding
# Maybe some recursion would be possible, not sure
# At the very least, find a way to reset converged solutions so that more can be found

# Hyperparamters 
# NOTES
    # For higher N values, higher mutation is great
    # >=100 pop is almost necessary
    # 1000 generations seems to be find up to n=20
n = 7 # amount of queens, size of chessboard, and size of perms
n_pop = 100 # perms
n_iter = 1000 # generations
r_cross = 0.9 # crossover rate
r_mut = 1.0 / float(n/2) # mutation rate

def construct_board(x, n):
    for sol in x:
        print("------------------------")
        print("Possible Solution:")
        board = [[0] * n for _ in range(n)]
        for col, row in enumerate(sol):
            board[row][col] = 1
        for row in board:
            print(row)


def mutation(x, r_mut):
    for i in range(len(x)):
        # check for a mutation
        # 10% chance for .rand() to be less than r_mut
        if random.rand() < r_mut:
            # pick a random possible number
            x[i] = np.random.randint(0, n)


def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    # .rand() gives a probability between [0, 1)
    if np.random.rand() < r_cross: 
        n = len(p1)
        pt = random.randint(0, n-1)

        # addition isn't possible for normal lists of different sizes, so use np.concatenate
        c1 = np.concatenate((p1[0:pt], p2[pt:n]))
        c2 = np.concatenate((p2[0:pt], p1[pt:n]))
        
    return c1, c2

def count_collisions(x, n):
    # x is an array representing a random N permutation
    # Now evaluate this permutation for how many collisions there are for the queens
    collisions = 0    
    for i in range(n):
        # iterate over pairs of queens
        # i is a queen from x, j is every queen that I could be touching
        # iterate through the perm so that no collisions are counted twice
        # i is only ever compared to j, which is always greater, due to the range
        # so, we don't check queens to the left of i
        for j in range(i+1, n):
            # if i and j are in the same row
            if (x[i] == x[j]):
                collisions += 1
            # trick to check diagonals
            if (abs(x[i] - x[j]) == j - i):
                collisions += 1

    return collisions

def gen_alg(objective, n, n_pop, n_iter, r_cross, r_mut):
    # Generate initial population using permutations of N
    pop = [random.permutation(n) for _ in range(n_pop)]
    solutions = []

    # initialize best to array of zeros in case it tries to mutate on the first iteration
    best, best_eval = [[0] * n], objective(pop[0], n)

    for gen in range(n_iter):
        # obtain number of collisions for each perm in pop
        scores = [objective(c, n) for c in pop]

        # check for new best
        for i in range(n_pop):
            if scores[i] <= best_eval: # minimzing collisions
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
                if best_eval == 0 and not any(np.array_equal(best, s) for s in solutions):
                    solutions.append(best.copy())
                    mutation(best, 1) # ensure mutation to find new solutions

                    # pop[i] = [0] * full resets like this work ok, but not great
                    # best = [0] * n 

        # select parents with generic selection function
        selected = [selection(pop, scores) for _ in range(n_pop)]

        # create next gen
        children = list()
        for i in range(0, n_pop, 2): 
            p1, p2 = selected[i], selected[i+1]

            # iterate over each returned child
            for c in crossover(p1, p2, r_cross):
                # mutate 
                mutation(c, r_mut)
                children.append(c)

        pop = children
    return [solutions, best_eval]

if __name__ == "__main__":
    best, best_eval = gen_alg(count_collisions, n, n_pop, n_iter, r_cross, r_mut)
    print("Done!")

    print("Number of Solutions Found: %d" % (len(best)))

    construct_board(best, n)
