HW04

Problem 1 - Generate task1.png
sbatch problem1.sh

Problem 2 - task2.cpp, generates nbody-cpp.png
sbatch problem2.sh

Problem 3 - task3.cpp, modify task2.cpp to use pragma for directives; timing report -> timing_problem3.txt
sbatch problem3.sh

(I plotted the graph for problem 3 using task3.m and it is written to task3.pdf)


Problem 4 - modify task3.cpp to run omp schedulers

a) run static scheduler; task3_static.cpp; timing report -> timing_problem4_static.txt
sbatch problem4_static.sh

b) run dynamic scheduler; task3_dynamic.cpp; timing report -> timing_problem4_dynamic.txt
sbatch problem4_dynamic.sh

c) run guided scheduler; task3_guided.cpp; timing report -> timing_problem4_guided.txt
sbatch problem4_guided.sh

(I plotted the graph for problem 4 using task3.m and it is written to task4.pdf)


