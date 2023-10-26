import random
import numpy as np
import pandas as pd
import seaborn as sns
from deap import base
from deap import tools
from deap import creator
import matplotlib.pyplot as plt


def upload_data(n1, n2):
    print("Distances between Customers and Warehouse:")
    df_dist_aux = upload_csv(n1)  # reads the data from the file
    df_dist_aux.rename(columns={'Distances between Customers and Warehouse': 'Distances'}, inplace=True)
    print(df_dist_aux.head(5))

    print("Number of Customers orders:")
    df_ord_aux = upload_csv('CustOrd.csv')  # reads the data from the file

    print("Position from Customers and Warehouse:")
    df_XY_aux = upload_csv(n2)  # reads the data from the file
    df_XY_aux['Plot Color'] = df_XY_aux['Customer XY'].apply(lambda x1: 1 if x1 == 0 else 0)  # creates new column to
    # create color on scatterplot
    if files_to_open != 1:
        df_XY_aux.rename(columns={'x': 'X', 'y': 'Y'}, inplace=True)
    df_XY_aux["Orders"] = df_ord_aux['Orders'].copy()
    print(df_XY_aux.head(5))
    print("Number total of orders: ", df_XY_aux["Orders"].sum())

    return df_dist_aux, df_ord_aux, df_XY_aux


def upload_csv(name):
    ######  Data understanding and cleaning  #####
    dfaux = pd.read_csv(name, sep=',')  # open the file
    return dfaux


def scatter_plot_comparison(dfaux):
    sns.relplot(data=dfaux, x='X', y='Y', size='Orders', hue='Plot Color')
    plt.show()
    plt.close()


##############################################
#####   Data understanding and cleaning  #####
##############################################

files_to_open = 2  # if 1 wharehouse is in the Central position else is in the Corner
if files_to_open == 1:
    name1 = 'CustDist_WHCentral.csv'
    name2 = 'CustXY_WHCentral.csv'
    plot_coordinates = 50  # position of the wharehouse
else:
    name1 = 'CustDist_WHCorner.csv'
    name2 = 'CustXY_WHCorner.csv'
    plot_coordinates = 0

df_dist, df_ord, df_XY = upload_data(name1, name2)  # reads the data from the file

# scatter_plot_comparison(df_XYCentral)  # shows the position of the Customers and wharehouse on a plot


###################################################
#####  Particle swarm optimization algorithm  #####
###################################################


random.seed(42)
number_customers = 30  # change the number of customers for the salesman to go to
truck_capacity = 1000  # change the orders' capacity of the truck
yes_heuristic = 1  # put 1 to use our heuristic, 2 to use project heuristic, 3 to both heuristics, other to none

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Customer", list, fitness=creator.FitnessMin)


def generate(number):  # function responsible for creating the population members
    global yes_heuristic

    if yes_heuristic == 1 or yes_heuristic == 3:  # creates one member with a heuristic function
        yes_heuristic -= 1
        cust_aux = heuristic()
        cust = creator.Customer(cust_aux)  # creates the member of the population
    elif yes_heuristic == 2:  # creates one member with a heuristic function
        yes_heuristic = 4
        cust_aux = heuristic2()
        cust = creator.Customer(cust_aux)  # creates the member of the population
    else:  # create a random member
        cust = creator.Customer(random.sample(range(number), number))
    return cust


def heuristic():  # Salesman parts from the current location and checks which distance is the smallest and heads to that
    # customer (function derives from: Steepest Ascent Hill Climbing)

    sub = np.ones(number_customers)  # used to subtract at the end cust result, so it stays the same as the others
    visited = np.ones(number_customers + 1)  # keeps track of the visited customers
    cust = []  # stores the salesman path
    visited[0] = 2
    soma_dist = 0
    soma_ord = 0
    for j in range(1, number_customers+1):
        minimo_dist = 0
        minimo_ord = 0
        atual = np.where(visited == 2)  # the current location of the salesman
        atual = atual[0][0]
        for e in range(1, number_customers+1):  # sees which customer is better to visit based on current location
            if visited[e] == 1:  # a customer that hasn't been visited
                orders = df_ord["Orders"].iloc[e]
                dist = df_dist.iloc[atual, e + 1]
                soma_ord_aux = soma_ord + orders
                if soma_ord_aux > truck_capacity:  # order is bigger than truck capacity needs to go back to wharehouse
                    dist = df_dist.iloc[0, cust[- 1] + 1] + df_dist.iloc[0, e + 1]
                if minimo_dist == 0 or dist < minimo_dist:  # stores the best customer to go next
                    minimo_dist = dist
                    minimo_ord = orders
                    next_ = e
            else:  # it already visited this customer
                continue
        # finds the best customer to go next and updates all values
        soma_ord = soma_ord + minimo_ord
        if soma_ord > truck_capacity:  # the salesman goes to load the van again
            soma_ord = minimo_ord
        soma_dist = soma_dist + minimo_dist
        visited[atual] = 0
        visited[next_] = 2
        cust.append(next_)
    cust = np.subtract(np.array(cust), sub)
    cust = cust.astype(int).tolist()
    return cust


def heuristic2():
    sub = np.ones(number_customers)  # used to subtract at the end cust result, so it stays the same as the others
    df_aux1 = df_XY[['Customer XY', 'X', 'Y']][(0 < df_XY['Customer XY']) & (df_XY['Customer XY'] <=
                                                                             number_customers)].copy()
    df_aux = df_aux1.sort_values(by='Y', ascending=False)
    rightdata_heuristic = df_aux['Customer XY'][df_aux['X'] > plot_coordinates].values

    df_aux = df_aux1.sort_values(by='Y', ascending=True)
    leftdata_heuristic = df_aux['Customer XY'][df_aux['X'] < plot_coordinates].values

    cust = np.concatenate((leftdata_heuristic, rightdata_heuristic))
    cust = np.subtract(np.array(cust), sub)
    cust = cust.astype(int).tolist()
    return cust


toolbox = base.Toolbox()

toolbox.register("customer", generate, number=number_customers)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.customer)


# the goals ('fitness') function to be minimized: evaluate_ord_file takes the number of orders from the files;
# evaluate_ord assumes that all orders values are 50

def evaluate_ord_file(individual):  # each individual needs to be added

    orders = df_ord["Orders"].iloc[individual[0] + 1]  # takes the order value for the first individual
    soma = df_dist.iloc[0, individual[0] + 2]  # takes the distance value from wharehouse to first individual

    for j in range(1, len(individual)):
        dist = df_dist.iloc[
            individual[j - 1] + 1, individual[j] + 2]  # takes the distance value from previous to
        # current individual
        orders = df_ord["Orders"].iloc[individual[j] + 1] + orders  # takes the order value for the current individual

        if orders > truck_capacity:  # order is bigger than truck capacity needs to go back to wharehouse
            orders = df_ord["Orders"].iloc[individual[j] + 1]  # takes the order value for the current individual
            # because didn't deliver
            dist = df_dist.iloc[0, individual[j - 1] + 2] + df_dist.iloc[0, individual[j] + 2]  # distance from
            # previous individual to wharehouse and then wharehouse to next individual

        soma = soma + dist

    dist = df_dist.iloc[0, individual[- 1] + 2]
    soma = soma + dist
    return soma,


def evaluate_ord(individual):  # same as evaluate_ord_file but with a fixed orders value at 50
    orders = 50
    soma = df_dist.iloc[0, individual[0] + 2]

    for j in range(1, len(individual)):
        dist = df_dist.iloc[individual[j - 1] + 1, individual[j] + 2]
        orders = 50 + orders
        if orders > truck_capacity:
            orders = 50
            dist = df_dist.iloc[0, individual[j - 1] + 2] + df_dist.iloc[0, individual[j] + 2]
        soma = soma + dist

    dist = df_dist.iloc[0, individual[- 1] + 2]
    soma = soma + dist

    return soma,


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", evaluate_ord_file)  # use either evaluate_ord_file or evaluate_ord

# register the crossover operator
toolbox.register("mate", tools.cxOrdered)  # we choose this, because it keeps the values of the arrays, so it doesn't
# repeat numbers

# register a mutation operator with a probability to shufle attributes
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of x individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=18)

# ----------

# Initialize statistics object
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

# create an initial population of 40 possible outcomes
pop = toolbox.population(n=40)

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.6, 0.3

print("Start of evolution")

# Evaluate the entire population

fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0

# Compile statistics about the population
record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
print(logbook.stream)

# Begin the evolution
while g <= 250:
    # A new generation
    g = g + 1

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):  # offspring[::2] faz com que salte a cada 2

        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]  # takes the individuals that either mutated
    # or where crossed over

    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    # Compile statistics about the new population
    record = stats.compile(pop)
    logbook.record(gen=g, evals=len(invalid_ind), **record)
    print(logbook.stream)

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
ones = np.ones(len(best_ind))
fit_final = best_ind.fitness.values
best_ind = np.add(ones, best_ind).astype(int)
print("Best individual is %s, %s" % (best_ind, fit_final))

### plots grafic with the values obtained in each generation
x, avg, max_, min_ = logbook.select("gen", "avg", "max", "min")
# plt.plot(x, avg, label="avg", linestyle="--")
# plt.plot(x, max_, label="max", linestyle=":")
plt.plot(x, min_, label="min", linestyle="-.")
plt.legend("Distances in each generation")
plt.show()
plt.close()

### creates the final path of the salesman taking account the times he had to get back to the wharehouse
soma_final = 0
best_ind_plot = np.append(0, best_ind)

for i in range(1, len(best_ind_plot)):
    ords = df_ord["Orders"].iloc[best_ind_plot[i]]  # takes the orders value for each customer
    soma_final = soma_final + ords
    if soma_final > truck_capacity:  # needs to get back to the wharehouse, so adds wharehouse location
        soma_final = ords
        aux = np.append(best_ind_plot[:i], 0)
        best_ind_plot = np.append(aux, best_ind_plot[i:])
best_ind_plot = np.append(best_ind_plot, 0)  # at the end he goes back to the wharehouse

### plots the salesman path
plt.xlim(0, 100)
plt.ylim(0, 100)


def draw_arrow(first, arr_start, arr_end):  # draws the arrows connecting each location
    dx = arr_end[0] - arr_start[0]
    dy = arr_end[1] - arr_start[1]
    if first == 1:  # the first time he leaves the wharehouse the arrow is dark green
        if arr_start[0] == plot_coordinates and arr_start[1] == plot_coordinates:
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,
                      color='darkgreen', linestyle="--")
            plt.plot(arr_start[0], arr_start[1], color='orange', marker='s')  # plot x and y using orange circle markers
    elif first == len(best_ind_plot)-1:  # the last time he returs to the wharehouse the arrow is red
        if arr_end[0] == plot_coordinates and arr_end[1] == plot_coordinates:
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,
                      color='red', linestyle="--")
            plt.plot(arr_start[0], arr_start[1], color='black', marker='o')  # plot x and y using blue circle markers
    else:
        if arr_start[0] == plot_coordinates and arr_start[1] == plot_coordinates:  # every time he leaves the wharehouse
            # the arrow is lime green
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,
                      color='lime')
        elif arr_end[0] == plot_coordinates and arr_end[1] == plot_coordinates:  # every he returs to the wharehouse the
            # arrow is salmon
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,
                      color='salmon')
            plt.plot(arr_start[0], arr_start[1], color='black', marker='o')  # plot x and y using blue circle markers
        else:  # every he goes from customer to customer the arrow is blue
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,
                      color='blue')
            plt.plot(arr_start[0], arr_start[1], color='black', marker='o')  # plot x and y using blue circle markers


for i in range(1, len(best_ind_plot)):  # retrieves map locations of customers and plots them + arrows
    draw_arrow(i, [df_XY["X"].iloc[best_ind_plot[i - 1]], df_XY["Y"].iloc[best_ind_plot[i - 1]]],
               [df_XY["X"].iloc[best_ind_plot[i]], df_XY["Y"].iloc[best_ind_plot[i]]])
plt.show()
plt.close()
