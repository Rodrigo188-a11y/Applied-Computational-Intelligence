TESTME.py
	- solution for the single objective problem
	- integrates the heuristic solution
	
	At line 49 change files_to_open = 1 to open the Warehouse in Central position; files_to_open = 2 to open the Warehouse in Corner position
	At line 70 change number_customers to 10, 30 or 50 as intended, or any other number ultil 50 
	At line 72 change heuristic functions:
		- To run with our version, use yes_heuristic = 1
		- To run with the project required version, use yes_heuristic = 2
		- To run with both, yes_heuristic = 3
		- None, anything else

	To get good results at line 220 change:	(this values need to be tuned to get better results for different populations, but for testing purposes the values below work fine)
		- tournsize = 3 if population 10
		- tournsize = 7 if population 30
		- tournsize = 18 if population 50
	

singleobjective.py 
	- solution for the single objective problem
	- integrates the heuristic solution

	Our Genetic Algorithm is inside 4 for's, where we automatically change all the parameters and do all the runs needed. 
	In our computers, it took aproximatelly 40min.

	We developed 2 heuristic functions. 
		- To run with our version, use yes_heuristic = 1
		- To run with the project required version, use yes_heuristic = 2
		- To run with both, yes_heuristic = 3
		- None, anything else

	However, to run a single #Customer, position and Order origin, you can just change the arrays the contain the values
		customers - for the number of customers the TSP has to solve
		positions - 'central' or 'corner'
		order_function - 'file' or '50 each'

multiobjective.py
	- solution for the multiobjective problem
	
	We present the solution of the minimization of cost and distance. After each generation we print the logbook, with 
the average, minimum and standard deviation values of the population.
	After the evolution, it's printed the minimum values that minimize each objective and are presented the pareto front 
and the hypervolume
	To finish, it's presented the path that the Travelling Salesman will make.