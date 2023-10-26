# %%
import random
import numpy as np
import pandas as pd
import seaborn as sns
from deap import base
from deap import tools
from deap import creator
import matplotlib.pyplot as plt


def upload_data(n1, n2):  # uploads data from files
    # print("Distances between Customers and Warehouse:")
    df_dist_aux = upload_csv(n1)  # reads the data from the file
    df_dist_aux.rename(columns={'Distances between Customers and Warehouse': 'Distances'}, inplace=True)
    # print(df_dist_aux.head(5))

    # print("Number of Customers orders:")
    df_ord = upload_csv('CustOrd.csv')  # reads the data from the file
    # print("Position from Customers and Warehouse:")
    df_XY_aux = upload_csv(n2)  # reads the data from the file
    df_XY_aux['Plot Color'] = df_XY_aux['Customer XY'].apply(lambda x1: 1 if x1 == 0 else 0)  # creates new column to
    # create color on scatterplot
    # if files_to_open != 1:

    df_XY_aux.rename(columns={'x': 'X', 'y': 'Y'}, inplace=True)
    df_XY_aux["Orders"] = df_ord['Orders'].copy()
    # print(df_XY_aux.head(5))
    # print("Number total of orders: ", df_XY_aux["Orders"].sum())

    return df_dist_aux, df_ord, df_XY_aux


def upload_csv(name):
    dfaux = pd.read_csv(name, sep=',')  # open the file
    return dfaux


def scatter_plot_comparison(dfaux):  # shows the position of the Customers and wharehouse on a plot
    sns.relplot(data=dfaux, x='X', y='Y', size='Orders', hue='Plot Color')
    plt.show()
    plt.close()


##############################################
#####   Data understanding and cleaning  #####
##############################################

dist_central = 'CustDist_WHCentral.csv'
xy_central = 'CustXY_WHCentral.csv'
dist_corner = 'CustDist_WHCorner.csv'
xy_corner = 'CustXY_WHCorner.csv'


# scatter_plot_comparison(df_XYCentral)  # shows the position of the Customers and wharehouse on a plot

def select_best(pop):  # retrieves the best salesman path and it's value
    best_ind = tools.selBest(pop, 1)[0]
    ones = np.ones(len(best_ind))
    fit_final = best_ind.fitness.values
    best_ind = np.add(ones, best_ind).astype(int)

    return best_ind, fit_final


def evaluate_ord_file(individual):  # creates the distance, by adding the path distances between customers

    orders = df_ord["Orders"].iloc[individual[0] + 1]  # takes the order value for the first individual
    soma = df_dist.iloc[0, individual[0] + 2]  # takes the distance value from wharehouse to first individual

    for j in range(1, len(individual)):
        dist = df_dist.iloc[
            individual[j - 1] + 1, individual[j] + 2]  # takes the distance value from previous to
        # current individual
        orders = df_ord["Orders"].iloc[individual[j] + 1] + orders  # takes the order value for the current individual

        if orders > TRUCKCAPACITY:  # order is bigger than truck capacity needs to go back to wharehouse
            orders = df_ord["Orders"].iloc[individual[j] + 1]  # takes the order value for the current individual
            # because didn't deliver
            dist = df_dist.iloc[0, individual[j - 1] + 2] + df_dist.iloc[0, individual[j] + 2]  # distance from
            # previous individual to wharehouse and then wharehouse to next individual

        soma = soma + dist

    dist = df_dist.iloc[0, individual[- 1] + 2]  # in the end the salesman needs to go back to the wharehouse
    soma = soma + dist
    return soma,


def evaluate_ord(individual):  # same as evaluate_ord_file but with a fixed orders value at 50
    orders = 50
    soma = df_dist.iloc[0, individual[0] + 2]

    for j in range(1, len(individual)):
        dist = df_dist.iloc[individual[j - 1] + 1, individual[j] + 2]
        orders = 50 + orders
        if orders > TRUCKCAPACITY:
            orders = 50
            dist = df_dist.iloc[0, individual[j - 1] + 2] + df_dist.iloc[0, individual[j] + 2]
        soma = soma + dist

    dist = df_dist.iloc[0, individual[- 1] + 2]
    soma = soma + dist

    return soma,


def generate(yes_heuristic, number):  # function responsible for creating the population members
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

    sub = np.ones(NUMCUSTOMERS)  # used to subtract at the end cust result, so it stays the same as the others
    visited = np.ones(NUMCUSTOMERS + 1)  # keeps track of the visited customers
    cust = []  # stores the salesman path
    visited[0] = 2
    soma_dist = 0
    soma_ord = 0
    for j in range(1, NUMCUSTOMERS + 1):
        minimo_dist = 0
        minimo_ord = 0
        atual = np.where(visited == 2)  # the current location of the salesman
        atual = atual[0][0]
        for e in range(1, NUMCUSTOMERS + 1):  # sees which customer is better to visit based on current location
            if visited[e] == 1:  # a customer that hasn't been visited
                orders = df_ord["Orders"].iloc[e]
                dist = df_dist.iloc[atual, e + 1]
                soma_ord_aux = soma_ord + orders
                if soma_ord_aux > TRUCKCAPACITY:  # order is bigger than truck capacity needs to go back to wharehouse
                    dist = df_dist.iloc[0, cust[- 1] + 1] + df_dist.iloc[0, e + 1]
                if minimo_dist == 0 or dist < minimo_dist:  # stores the best customer to go next
                    minimo_dist = dist
                    minimo_ord = orders
                    next_ = e
            else:  # it already visited this customer
                continue
        # finds the best customer to go next and updates all values
        soma_ord = soma_ord + minimo_ord
        if soma_ord > TRUCKCAPACITY:  # the salesman goes to load the van again
            soma_ord = minimo_ord
        soma_dist = soma_dist + minimo_dist
        visited[atual] = 0
        visited[next_] = 2
        cust.append(next_)
    cust = np.subtract(np.array(cust), sub)
    cust = cust.astype(int).tolist()
    return cust


def heuristic2():  # the project heuristic
    sub = np.ones(NUMCUSTOMERS)  # used to subtract at the end cust result, so it stays the same as the others

    # it finds all customers that the salesman is going to visit (10, 30 or 50)
    df_aux1 = df_XY[['Customer XY', 'X', 'Y']][(0 < df_XY['Customer XY']) & (df_XY['Customer XY'] <=
                                                                             NUMCUSTOMERS)].copy()

    df_aux = df_aux1.sort_values(by='Y', ascending=False)  # the customers are ordered from higher Y to lowest
    rightdata_heuristic = df_aux['Customer XY'][df_aux['X'] > PLOTCOORDINATES].values  # it finds all customers to the
    # right of the wharehouse

    df_aux = df_aux1.sort_values(by='Y', ascending=True)  # the customers are ordered from lowest Y to higher
    leftdata_heuristic = df_aux['Customer XY'][df_aux['X'] < PLOTCOORDINATES].values  # it finds all customers to the
    # left of the wharehouse

    # creates an individual for the population
    cust = np.concatenate((leftdata_heuristic, rightdata_heuristic))
    cust = np.subtract(np.array(cust), sub)
    cust = cust.astype(int).tolist()
    return cust


###################################################
#####            Genetic Algorithm            #####
###################################################

fit_score = np.zeros((3, 2, 2, 30))
min_score = np.full((3, 2, 2), np.inf)
log_score = np.zeros((3, 2, 2, 2, 252))

TRUCKCAPACITY = 1000  # change the orders' capacity of the truck
yes_heuristic = 2  # put 1 to use our heuristic version, 2 to use project heuristic, 3 to both heuristics, other to none

customers = [10, 30, 50]
positions = ['central', 'corner']
order_function = ['file', '50 each']

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Customer", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

## It retrieves the right files and the right number of members for testing to cretae the graphs asked for the delivery

# first for is to change number of customers
for cust in customers:
    NUMCUSTOMERS = cust  # change the number of customers for the salesman to go to

    # chages the Genetic algorithm variables depending on the number of costumers. The values where achieved by trial
    # and error
    if cust == 10:
        TOURNSIZE = 3
        NCUST = 0
    elif cust == 30:
        TOURNSIZE = 7
        NCUST = 1
    elif cust == 50:
        TOURNSIZE = 15
        NCUST = 2

    toolbox.register("customer", generate, yes_heuristic, number=NUMCUSTOMERS)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.customer)

    print('Customers:', cust)

    # second for is to change the files that are opened
    for pos in positions:
        print('Pos:', pos)
        if pos == 'central':
            df_dist, df_ord, df_XY = upload_data(dist_central, xy_central)  # reads the data from the file
            PLOTCOORDINATES = 50
            NPOS = 0

        elif pos == 'corner':
            df_dist, df_ord, df_XY = upload_data(dist_corner, xy_corner)  # reads the data from the file
            PLOTCOORDINATES = 0
            NPOS = 1

        for order in order_function:
            print('Order:', order)
            if order == 'file':
                ord_function = evaluate_ord_file
                NORD = 0
            elif order == '50 each':
                ord_function = evaluate_ord
                NORD = 1

            # third for does the 30 trials needed
            for iterat in range(0, 30):
                random.seed(iterat)

                # the goals ('fitness') function to be minimized: evaluate_ord_file takes the number of orders from the
                # files;
                # evaluate_ord assumes that all orders values are 50

                # Operator registration
                # register the goal / fitness function
                toolbox.register("evaluate", ord_function)  # use either evaluate_ord_file or evaluate_ord

                # register the crossover operator
                toolbox.register("mate",
                                 tools.cxOrdered)  # we choose this, because it keeps the values of the arrays, so it doesn't
                # repeat numbers

                # register a mutation operator with a probability to shufle attributes
                toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

                # operator for selecting individuals for breeding the next
                # generation: each individual of the current generation
                # is replaced by the 'fittest' (best) of x individuals
                # drawn randomly from the current generation.
                toolbox.register("select", tools.selTournament,
                                 tournsize=TOURNSIZE)  # Ordenated with 50 -> best First:6 Second:7, Third:10
                # Ordenated with file -> best First:9 Second:10, Third:7
                # tournsize: 3 para 10, 7 para 30, 18 para 50

                # Initialize statistics object

                logbook = tools.Logbook()
                logbook.header = "gen", "std", "min", "avg"

                # create an initial population of 40 possible outcomes
                pop = toolbox.population(n=40)

                # CXPB  is the probability with which two individuals
                #       are crossed
                #
                # MUTPB is the probability for mutating an individual
                CXPB, MUTPB = 0.7, 0.3

                # print("Start of evolution")

                # Evaluate the entire population

                fitnesses = list(map(toolbox.evaluate, pop))

                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit

                # Extracting all the fitnesses of
                fits = [ind.fitness.values[0] for ind in pop]

                # Compile statistics about the population
                record = stats.compile(pop)
                logbook.record(gen=0, **record)
                # print(logbook.stream)

                # Variable keeping track of the number of generations
                g = 0
                # Begin the evolution
                while g <= 250:
                    # A new generation
                    g = g + 1
                    # print('Generation', g, 'Iteration', iterat)

                    # Select the next generation individuals
                    offspring = toolbox.select(pop, len(pop))

                    # Clone the selected individuals
                    offspring = list(map(toolbox.clone, offspring))

                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2],
                                              offspring[1::2]):  # offspring[::2] faz com que salte a cada 2

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
                    invalid_ind = [ind for ind in offspring if
                                   not ind.fitness.valid]  # takes the individuals that either mutated
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
                    logbook.record(gen=g, **record)
                    # print(logbook.stream)

                # print("-- End of evolution --")

                ind, fit_final = select_best(pop)

                fit_score[NCUST][NPOS][NORD][iterat] = fit_final[0]

                if fit_final[0] < min_score[NCUST][NPOS][NORD]:
                    min_score[NCUST][NPOS][NORD] = fit_final[0]
                    log_score[NCUST][NPOS][NORD][0] = logbook.select('gen')
                    log_score[NCUST][NPOS][NORD][1] = logbook.select('min')

                print('Fit:', fit_final[0], ' | ', 'Iter:', ' | ', iterat, ' | ', cust, ' | ', pos, ' | ', order)


def score(fit, min_value):
    dist = min_value
    mean = np.mean(fit)
    std = np.std(fit)
    print('Dist:', dist)
    print('Mean:', mean)
    print('STD:', std)
    return dist, mean, std


### The graphs will be created for the values achieved on the previous run. For each of the 12 case studies the graphs
# showed are from the best run
# %%
dist = np.empty((3, 2, 2))
mean = np.empty((3, 2, 2))
std = np.empty((3, 2, 2))

NCUST = 0
print('***** 10 Customers *****')
NPOS = 0
print('------- Central -------')
print('>>>>> 50 orders each <<<<<')
dist[NCUST][NPOS][0], mean[NCUST][NPOS][0], std[NCUST][NPOS][0] = score(fit_score[NCUST][NPOS][0],
                                                                        min_score[NCUST][NPOS][0])

print('------- Central -------')
print('>>>>> Orders from file <<<<<')
dist[NCUST][NPOS][1], mean[NCUST][NPOS][1], std[NCUST][NPOS][1] = score(fit_score[NCUST][NPOS][1],
                                                                        min_score[NCUST][NPOS][1])

NPOS = 1
print('------- Corner -------')
print('>>>>> 50 orders each <<<<<')
dist[NCUST][NPOS][0], mean[NCUST][NPOS][0], std[NCUST][NPOS][0] = score(fit_score[NCUST][NPOS][0],
                                                                        min_score[NCUST][NPOS][0])

print('------- Corner -------')
print('>>>>> Orders from file <<<<<')
dist[NCUST][NPOS][1], mean[NCUST][NPOS][1], std[NCUST][NPOS][1] = score(fit_score[NCUST][NPOS][1],
                                                                        min_score[NCUST][NPOS][1])

NCUST = 1
print('***** 30 Customers *****')
NPOS = 0
print('------- Central -------')
print('>>>>> 50 orders each <<<<<')
dist[NCUST][NPOS][0], mean[NCUST][NPOS][0], std[NCUST][NPOS][0] = score(fit_score[NCUST][NPOS][0],
                                                                        min_score[NCUST][NPOS][0])

print('------- Central -------')
print('>>>>> Orders from file <<<<<')
dist[NCUST][NPOS][1], mean[NCUST][NPOS][1], std[NCUST][NPOS][1] = score(fit_score[NCUST][NPOS][1],
                                                                        min_score[NCUST][NPOS][1])

NPOS = 1
print('------- Corner -------')
print('>>>>> 50 orders each <<<<<')
dist[NCUST][NPOS][0], mean[NCUST][NPOS][0], std[NCUST][NPOS][0] = score(fit_score[NCUST][NPOS][0],
                                                                        min_score[NCUST][NPOS][0])

print('------- Corner -------')
print('>>>>> Orders from file <<<<<')
dist[NCUST][NPOS][1], mean[NCUST][NPOS][1], std[NCUST][NPOS][1] = score(fit_score[NCUST][NPOS][1],
                                                                        min_score[NCUST][NPOS][1])

NCUST = 2
print('***** 50 Customers *****')
NPOS = 0
print('------- Central -------')
print('>>>>> 50 orders each <<<<<')
dist[NCUST][NPOS][0], mean[NCUST][NPOS][0], std[NCUST][NPOS][0] = score(fit_score[NCUST][NPOS][0],
                                                                        min_score[NCUST][NPOS][0])

print('------- Central -------')
print('>>>>> Orders from file <<<<<')
dist[NCUST][NPOS][1], mean[NCUST][NPOS][1], std[NCUST][NPOS][1] = score(fit_score[NCUST][NPOS][1],
                                                                        min_score[NCUST][NPOS][1])

NPOS = 1
print('------- Corner -------')
print('>>>>> 50 orders each <<<<<')
dist[NCUST][NPOS][0], mean[NCUST][NPOS][0], std[NCUST][NPOS][0] = score(fit_score[NCUST][NPOS][0],
                                                                        min_score[NCUST][NPOS][0])

print('------- Corner -------')
print('>>>>> Orders from file <<<<<')
dist[NCUST][NPOS][1], mean[NCUST][NPOS][1], std[NCUST][NPOS][1] = score(fit_score[NCUST][NPOS][1],
                                                                        min_score[NCUST][NPOS][1])

plt.figure()
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

x = ['Central, 50 each', 'Central, File', 'Corner, 50 each', 'Corner, File']
br1 = np.arange(len(x))
br2 = [y + barWidth for y in br1]
br3 = [y + barWidth for y in br2]
plt.title('Minimum Distance of each Case Study')
plt.bar(br1, [dist[0][0][0], dist[0][0][1], dist[0][1][0], dist[0][1][1]],
        width=barWidth, color='r', label='10 customers')
plt.bar(br2, [dist[1][0][0], dist[1][0][1], dist[1][1][0], dist[1][1][1]],
        width=barWidth, color='b', label='30 customers')
plt.bar(br3, [dist[2][0][0], dist[2][0][1], dist[2][1][0], dist[2][1][1]],
        width=barWidth, color='g', label='50 customers')
plt.legend()
plt.xlabel('Case Studies', fontweight='bold', fontsize=15)
plt.ylabel('Total Distance', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(x))], x)
plt.show()
plt.close()

plt.figure()
barWidth = 0.25
x = ['Central, 50 each', 'Central, File', 'Corner, 50 each', 'Corner, File']
y0 = [dist[0][0][0], dist[0][0][1], dist[0][1][0], dist[0][1][1]]
y1 = [dist[1][0][0], dist[1][0][1], dist[1][1][0], dist[1][1][1]]
y2 = [dist[2][0][0], dist[2][0][1], dist[2][1][0], dist[2][1][1]]

plt.plot(np.arange(0, len(x)), y0, 'b', label='10 cust', marker='o')
plt.plot(np.arange(0, len(x)), y1, 'r', label='30 cust', marker='o')
plt.plot(np.arange(0, len(x)), y2, 'g', label='50 cust', marker='o')
plt.title('Total Distance in each Case Study')
plt.xlabel('Case Study')
plt.ylabel('Total Distance')
plt.xticks([r + barWidth for r in range(len(x))], x)
plt.legend()
plt.show()

plt.figure()
plt.plot(log_score[0][0][0][0], log_score[0][0][0][1], 'b', label='Central, 50 each')
plt.plot(log_score[0][0][1][0], log_score[0][0][1][1], 'r', label='Central, file')
plt.plot(log_score[0][1][0][0], log_score[0][1][0][1], 'g', label='Corner, 50 each')
plt.plot(log_score[0][1][1][0], log_score[0][1][1][1], 'y', label='Corner, file')
plt.title('Convergence with 10 customers')
plt.xlabel('Generations')
plt.ylabel('Score')
plt.legend()
plt.show()

plt.figure()
plt.plot(log_score[1][0][0][0], log_score[1][0][0][1], 'b', label='Central, 50 each')
plt.plot(log_score[1][0][1][0], log_score[1][0][1][1], 'r', label='Central, file')
plt.plot(log_score[1][1][0][0], log_score[1][1][0][1], 'g', label='Corner, 50 each')
plt.plot(log_score[1][1][1][0], log_score[1][1][1][1], 'y', label='Corner, file')
plt.title('Convergence with 30 customers')
plt.xlabel('Generations')
plt.ylabel('Score')
plt.legend()
plt.show()

plt.figure()
plt.plot(log_score[2][0][0][0], log_score[2][0][0][1], 'b', label='Central, 50 each')
plt.plot(log_score[2][0][1][0], log_score[2][0][1][1], 'r', label='Central, file')
plt.plot(log_score[2][1][0][0], log_score[2][1][0][1], 'g', label='Corner, 50 each')
plt.plot(log_score[2][1][1][0], log_score[2][1][1][1], 'y', label='Corner, file')
plt.title('Convergence with 50 customers')
plt.xlabel('Generations')
plt.ylabel('Score')
plt.legend()
plt.show()
