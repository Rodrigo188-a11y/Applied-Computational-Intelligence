import random
import numpy as np
import pandas as pd
import seaborn as sns
from deap import base
from deap import tools
from deap import creator
import matplotlib.pyplot as plt
from deap.benchmarks.tools import hypervolume

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

    df_XY_aux.rename(columns={'x': 'X', 'y': 'Y'}, inplace=True)

    df_XY_aux["Orders"] = df_ord_aux['Orders'].copy()
    print(df_XY_aux.head(5))

    print('Sum of orders:',df_XY_aux["Orders"].sum())

    return df_dist_aux, df_ord_aux, df_XY_aux


def upload_csv(name):
    ######  Data understanding and cleaning  #####
    dfaux = pd.read_csv(name, sep=',')  # open the file

    return dfaux


def heuristic():  # Salesman parts from the current location and checks which distance is the smallest and heads to that
    # customer (function derives from: Steepest Ascent Hill Climbing)

    sub = np.ones(NUMCUSTOMERS)  # used to subtract at the end cust result, so it stays the same as the others
    visited = np.ones(NUMCUSTOMERS + 1)  # keeps track of the visited customers
    cust = []  # stores the salesman path
    visited[0] = 2
    soma_dist = 0
    soma_ord = 0
    for j in range(1, NUMCUSTOMERS+1):
        minimo_dist = 0
        minimo_ord = 0
        atual = np.where(visited == 2)  # the current location of the salesman
        atual = atual[0][0]
        for e in range(1, NUMCUSTOMERS+1):  # sees which customer is better to visit based on current location
            if visited[e] == 1:  # a customer that hasn't been visited
                orders = df_ord["Orders"].iloc[e]
                dist = df_dist.iloc[atual, e + 1]
                soma_ord_aux = soma_ord + orders
                if soma_ord_aux > TRUCKCAPACITY:  # order is bigger than truck capacity needs to go back to warehouse
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

def scatter_plot_comparison(df_aux):
    sns.relplot(data=df_aux, x='X', y='Y', size='Orders', hue='Plot Color')
    plt.show()
    plt.close()

##############################################
#####   Data understanding and cleaning  #####
##############################################

name1 = 'CustDist_WHCentral.csv'
name2 = 'CustXY_WHCentral.csv'
PLOTCOORDINATES = 50  # position of the warehouse


df_dist, df_ord, df_XY = upload_data(name1, name2)  # reads the data from the file

scatter_plot_comparison(df_XY)  # shows the position of the Customers and warehouse on a plot

def generate(number):  # function responsible for creating the population members
    global yes_heuristic
    if yes_heuristic == 1:  # creates one member with a heuristic function
        yes_heuristic = 2
        customer_aux = heuristic()
        customer = creator.Customer(customer_aux)  # creates the member of the population
    else:  # create a random member
        customer = creator.Customer(random.sample(range(number), number))

    return customer


def evaluate_ord_file(individual):  # each individual needs to be added
    orders = df_ord["Orders"].iloc[individual[0] + 1]  # takes the order value for the first individual
    soma = df_dist.iloc[0, individual[0] + 2]  # takes the distance value from warehouse to first individual

    for j in range(1, len(individual)):
        dist = df_dist.iloc[individual[j - 1] + 1, individual[j] + 2]  # takes the distance value from previous to
        # current individual
        orders = df_ord["Orders"].iloc[individual[j] + 1] + orders  # takes the order value for the current individual

        if orders > TRUCKCAPACITY:  # order is bigger than truck capacity needs to go back to warehouse
            orders = df_ord["Orders"].iloc[individual[j] + 1]  # takes the order value for the current individual
            # because didn't deliver
            dist = df_dist.iloc[0, individual[j - 1] + 2] + df_dist.iloc[0, individual[j] + 2]  # distance from
            # previous individual to warehouse and then warehouse to next individual

        soma = soma + dist

    dist = df_dist.iloc[0, individual[-1] + 2]
    soma = soma + dist

    return soma

def evaluate_cost_file(individual):  # calculates the cost of each load
    orders = df_ord["Orders"].iloc[individual[0] + 1]
    cost = 0

    for j in range(1, len(individual)):
        dist = df_dist.iloc[individual[j - 1] + 1, individual[j] + 2]
        orders = df_ord["Orders"].iloc[individual[j] + 1] + orders

        if orders > TRUCKCAPACITY:
            dist = df_dist.iloc[0, individual[j - 1] + 2] + df_dist.iloc[0, individual[j] + 2]
            cost += (TRUCKCAPACITY - orders)*dist
            orders = df_ord["Orders"].iloc[individual[j] + 1]
        else:
            cost += orders*dist

    return cost


def f(individual):
    dist = evaluate_ord_file(individual)
    cost = evaluate_cost_file(individual)

    return cost, dist,

def select_best(pop):
    best_ind = tools.selBest(pop, 1)[0]
    ones = np.ones(len(best_ind))
    fit_final = best_ind.fitness.values
    best_ind = np.add(ones, best_ind).astype(int)
    # print("Best individual is %s, %s" % (best_ind, fit_final))
    return best_ind

def plot_convergence(logbook):
    x, avg, min_ = logbook.select("gen", "avg", "min")
    plt.plot(x, avg, label="avg", linestyle="--")
    plt.plot(x, min_, label="min", linestyle="-.")
    plt.title("Distances in each generation")
    plt.xlabel('Generations')
    plt.ylabel('Convergence')
    plt.legend()
    plt.show()
    plt.close()
    
def plot_hypervolume(hyper):
    plt.figure()
    plt.plot(hyper,'bo', label = 'Hypervolume')
    plt.title('Hypervolume')
    plt.xlabel('Distance Objective')
    plt.ylabel('Cost Objective')
    plt.show()
    plt.close()
    
def plot_pareto(pareto):
    x, y = zip(*[ind.fitness.values for ind in pareto])
    
    print('Minimum Cost:')
    print('Cost:', x[0],'Distance:', y[0])
    print('\nMinimum Distance')
    print('Cost:', x[-1],'Distance:', y[-1])

    plt.figure()
    plt.scatter(x, y)
    plt.title('Pareto Front')
    plt.xlabel('Distance Objective')
    plt.ylabel('Cost Objective')
    plt.show()

def draw_arrow(first, arr_start, arr_end, idx):  # draws the arrows connecting each location
    dx = arr_end[0] - arr_start[0]
    dy = arr_end[1] - arr_start[1]
    if first == 1:  # the first time he leaves the warehouse the arrow is dark green
        if arr_start[0] == PLOTCOORDINATES and arr_start[1] == PLOTCOORDINATES:
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,
                      color='darkgreen', linestyle="--")
            plt.plot(arr_start[0], arr_start[1], color='orange', marker='s')  # plot x and y using orange circle markers
    elif first == len(best_ind_plot)-1:  # the last time he returs to the warehouse the arrow is red
        if arr_end[0] == PLOTCOORDINATES and arr_end[1] == PLOTCOORDINATES:
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,color='red', linestyle="--")
            plt.plot(arr_start[0], arr_start[1], color='black', marker='o')  # plot x and y using blue circle markers
    else:
        if arr_start[0] == PLOTCOORDINATES and arr_start[1] == PLOTCOORDINATES:  # every time he leaves the warehouse
            # the arrow is lime green
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,color='lime')
        elif arr_end[0] == PLOTCOORDINATES and arr_end[1] == PLOTCOORDINATES:  # every he returs to the warehouse the
            # arrow is salmon
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,color='salmon')
            plt.plot(arr_start[0], arr_start[1], color='black', marker='o')  # plot x and y using blue circle markers
        else:  # every he goes from customer to customer the arrow is blue
            plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=2, head_length=2, length_includes_head=True,color='blue')
            plt.plot(arr_start[0], arr_start[1], color='black', marker='o')  # plot x and y using blue circle markers
            plt.annotate(text=str(df_ord["Orders"].loc[idx]), xy=(arr_start[0],arr_start[1]), xytext=(arr_start[0], arr_start[1]-3))

def draw_map(best_ind):
    soma_final = 0
    best_ind_plot = np.append(0, best_ind)

    for i in range(1, len(best_ind_plot)):
        ords = df_ord["Orders"].iloc[best_ind_plot[i]]  # takes the orders value for each customer
        soma_final = soma_final + ords

        if soma_final > TRUCKCAPACITY:  # needs to get back to the warehouse, so adds warehouse location
            soma_final = ords
            aux = np.append(best_ind_plot[:i], 0)
            best_ind_plot = np.append(aux, best_ind_plot[i:])

    best_ind_plot = np.append(best_ind_plot, 0)  # at the end he goes back to the warehouse
    return best_ind_plot

def draw_full_path(best_ind_plot):
    for i in range(1, len(best_ind_plot)):  # retrieves map locations of customers and plots them + arrows
        x_start = df_XY["X"].iloc[best_ind_plot[i - 1]]
        y_start = df_XY["Y"].iloc[best_ind_plot[i - 1]]
        
        x_end = df_XY["X"].iloc[best_ind_plot[i]]
        y_end = df_XY["Y"].iloc[best_ind_plot[i]]

        draw_arrow(i, [x_start, y_start], [x_end, y_end], best_ind_plot[i-1])

    plt.title('Travelling Salesman Path')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()
    plt.close()


###################################################
#####            Genetic Algorithm            #####
###################################################

random.seed(42)
NUMCUSTOMERS = 30  # change the number of customers for the salesman to go to
TRUCKCAPACITY = 1000  # change the orders' capacity of the truck
yes_heuristic = 2  # put 1 to use heuristic to create first population member, else put other number
hyper = list()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,))
creator.create("Customer", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("customer", generate, number=NUMCUSTOMERS)
toolbox.register("population", tools.initRepeat, list, toolbox.customer)


# the goals ('fitness') function to be minimized: evaluate_ord_file takes the number of orders from the files;
# evaluate_ord assumes that all orders values are 50


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", f)  # use either evaluate_ord_file or evaluate_ord

# register the crossover operator
toolbox.register("mate", tools.cxOrdered)  # we choose this, because it keeps the values of the arrays, so it doesn't repeat numbers

# register a mutation operator with a probability to shufle attributes
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# operator for selecting individuals for breeding the next generation: each individual of the current generation is replaced by the 
# 'fittest' (best) of x individuals drawn randomly from the current generation.
toolbox.register("select", tools.selNSGA2)

pareto = tools.ParetoFront()
hyper = list()

# Initialize statistics object
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)

logbook = tools.Logbook()
logbook.header = "gen", "std", "min", "avg"

# create an initial population of 40 possible outcomes
NGEN = 250
MU = 40
CXPB = 0.9

# print("Start of evolution")

pop = toolbox.population(n=MU)
fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

pop = toolbox.select(pop,len(pop))

# Compile statistics about the population
record = stats.compile(pop)
logbook.record(gen=0, **record)

# Begin the evolution
g = 0
while g <= NGEN:
    # A new generation
    g = g + 1

    # Select the next generation individuals
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]

    # Apply crossover and mutation on the offspring
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):  # offspring[::2] faz com que salte a cada 2
        if random.random() < CXPB:
            toolbox.mate(ind1, ind2)
    
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        del ind1.fitness.values
        del ind2.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Gather all the fitnesses in one list and print the stats
    pop = toolbox.select(pop + offspring, MU)

    pareto.update(pop)
    
    hyper.append(hypervolume(pareto, [2e7, 2000]))
    
    record = stats.compile(pop)
    logbook.record(gen=g, **record)
    print(logbook.stream)

# print("-- End of (successful) evolution --")

best_ind = select_best(pop)

### plots grafic with the values obtained in each generation
plot_convergence(logbook)
plot_hypervolume(hyper)
plot_pareto(pareto)


### creates the final path of the salesman taking account the times he had to get back to the warehouse
best_ind_plot = draw_map(best_ind)

### plots the salesman path
draw_full_path(best_ind_plot)