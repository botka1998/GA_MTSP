import math
import abb
import json
import numpy as np
import random
import matplotlib.pyplot as plt



def get_random_targets(num):
    random_targets = np.empty([num,3])
    for i in range(num):
        random_targets[i] = [round(random.uniform(0,300),3), round(random.uniform(0,300),3), 0]
    return random_targets

def random_chromosome():
    rob1 = random.randint(0, Globals.num_targets)
    rob2 = Globals.num_targets - rob1
    length = 2 + rob1 + rob2
    random_ruta = np.arange(1, Globals.num_targets+1, 1, dtype=int)
    np.random.shuffle(random_ruta)
    chromosome = np.empty(length,dtype=int)
    chromosome[0] = rob1
    chromosome[1] = rob2
    for i in range(2,length):
        chromosome[i] = random_ruta[i - 2]

    return chromosome

def initial_population():
    initial_popopulation = np.empty([Globals.population_size, Globals.num_targets + 2], dtype=int)
    for i in range(Globals.population_size):
        initial_popopulation[i] = random_chromosome()
    return initial_popopulation
def mutate_chromosome(chromosome):
    base = chromosome[2:]
    if random.random() < Globals.mutation_rate:
        indx1, indx2 = random.randint(0, len(base)-1), random.randint(0, len(base)-1)
        temp = base[indx1]
        base[indx1] = base[indx2]
        base[indx2] = temp
    if random.random() < Globals.mutation_rate:
        rob1 = random.randint(0, len(base))
        rob2 = len(base) - rob1
        chromosome[0] = rob1
        chromosome[1] = rob2
    chromosome[2:] = base
    return chromosome
def get_route_distance(home,route):
    route_distance = 0
    if home == 1:
        prev_target = Globals.home1
    elif home == 2:
        prev_target = Globals.home2
    for i in range(len(route)):
        curr_target = Globals.targets[route[i]-1]
        route_distance += math.dist(prev_target, curr_target)
        prev_target = curr_target
    if home == 1:
        route_distance += math.dist(prev_target, Globals.home1)
    elif home == 2:
        route_distance += math.dist(prev_target, Globals.home2)

    return route_distance
def get_distance(chromosome):
    distance = 0
    base = chromosome[2:]
    if chromosome[0] != 0:
        route1 = base[:chromosome[0]]
        distance += get_route_distance(1,route1)
    if chromosome[1] != 0:
        route2 = base[chromosome[0]:]
        distance += get_route_distance(2, route2)

    return distance

def get_fitness(chromosome):
    rob1 = chromosome[0]
    route = chromosome[2:]
    distance1 = get_route_distance(1, route[:rob1])
    distance2 = get_route_distance(2, route[rob1:])
    fitness = 1 / ((distance1 / Globals.speed1) + (distance2 / Globals.speed2))
    return fitness

def crossover(p1,p2):
    child = np.zeros_like(p1)
    child_base = child[2:]
    base1 = p1[2:] #vadimo rute za oba roditelja
    base2 = p2[2:]
    length = len(base1)
    #generisemo random pocetak i kraj dela hromozoma koji ce dete naslediti od roditelja1
    start, end = 0 , 0
    while start == end:
        start, end = random.randint(0, length-1), random.randint(0, length-1)

    if start > end:
        temp = start
        start = end
        end = temp
    #DETE direktno nasledjuje deo hromozoma roditelja1
    for i in range(start, end):
        child_base[i] = base1[i]

    if start > 0:
        i = 0
        j = 0
        while i < start:
            if not np.isin(base2[j],child_base):
                child_base[i] = base2[j]
                i += 1
            j += 1
    if end < (length):
        i = end
        j = 0
        while i < length:
            if not np.isin(base2[j], child_base):
                child_base[i] = base2[j]
                i += 1
            j += 1

    rob1 = random.randint(0, Globals.num_targets)
    rob2 = Globals.num_targets - rob1
    child[0] = rob1
    child[1] = rob2
    child[2:] = child_base

    return child

def pick_parrent(population):
    number_of_rows = population.shape[0]
    random_slice = np.random.choice(number_of_rows,size=Globals.pick_size,replace=False)
    pick_population = population[random_slice, :]
    parrent = get_best(pick_population)
    return parrent

def evolve_population(population):
    next_population = np.empty_like(population)
    for i in range(Globals.population_size):
        p1 = pick_parrent(population)
        p2 = pick_parrent(population)

        new_chromosome = crossover(p1,p2)
        next_population[i] = mutate_chromosome(new_chromosome)
    return next_population
def get_best(population):
    best = population[0]
    best_fitness = get_fitness(population[0])
    size = population.size / (Globals.num_targets + 2)
    for i in range(int(size)):
        if get_fitness(population[i]) > best_fitness:
            best = population[i]
            best_fitness = get_fitness(best)
    return best
def plot_route(home, route):
    prev = home
    plt.scatter(prev[0], prev[1])

    for i in range(len(route)):
        curr = Globals.targets[route[i]-1]
        plt.scatter(curr[0],curr[1])
        plt.plot([prev[0],curr[0]], [prev[1],curr[1]])
        prev = curr
    plt.plot([prev[0], home[0]], [prev[1], home[1]])
    plt.show()

def plot(chromosome):
    base = chromosome[2:]
    if chromosome[0] != 0:
        route1 = base[:chromosome[0]]
        plot_route(Globals.home1, route1)
    if chromosome[1] != 0:
        route2 = base[chromosome[0]:]
        plot_route(Globals.home2, route2)

class Globals:
    num_targets = 12
    num_generations = 100
    population_size = 100
    mutation_rate = 0.4
    pick_size = 10
    speed1 = 800
    speed2 = 750
    home1 = np.array([438.495, -401.141, 619.5])
    home2 = np.array([438.495, 678.859, 619.5])
    targets = get_random_targets(num_targets)