all: search cluster

search: search.o help_functions.o calculations.o calculations_lsh.o
	g++ search.o help_functions.o calculations.o calculations_lsh.o -o search

cluster: cluster.o help_functions.o calculations.o calculations_cluster.o
	g++ cluster.o help_functions.o calculations.o calculations_cluster.o -o cluster

search.o: search.cpp
	g++ -c search.cpp

cluster.o: cluster.cpp
	g++ -c cluster.cpp

help_functions.o: help_functions.cpp
	g++ -c help_functions.cpp

calculations.o: calculations.cpp
	g++ -c calculations.cpp
	
calculations_lsh.o: calculations_lsh.cpp
	g++ -c calculations_lsh.cpp

calculations_cluster.o: calculations_cluster.cpp
	g++ -c calculations_cluster.cpp

clean:
	rm -f search cluster help_functions calculations calculations_lsh calculations_cluster search.o cluster.o help_functions.o calculations.o calculations_lsh.o calculations_cluster.o
