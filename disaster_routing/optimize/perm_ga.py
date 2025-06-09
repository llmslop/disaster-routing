import random


def permutation_genetic_algorithm(
    n, objective, population_size=100, generations=500, mutation_rate=0.2
):
    def random_perm():
        perm = list(range(n))
        random.shuffle(perm)
        return perm

    def mutate(perm):
        a, b = random.sample(range(n), 2)
        perm[a], perm[b] = perm[b], perm[a]

    def crossover(p1, p2):
        # Order Crossover (OX)
        start, end = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[start:end] = p1[start:end]
        fill = [x for x in p2 if x not in child]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[j]
                j += 1
        return child

    # Initial population
    population = [random_perm() for _ in range(population_size)]

    best_perm = None
    best_cost = float("inf")

    for gen in range(generations):
        population.sort(key=objective)
        if objective(population[0]) < best_cost:
            best_cost = objective(population[0])
            best_perm = population[0][:]

        # Select elites (top 20%)
        elites = population[: population_size // 5]

        # Generate new population
        new_population = elites[:]
        while len(new_population) < population_size:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                mutate(child)
            new_population.append(child)

        population = new_population

    return best_perm, best_cost
