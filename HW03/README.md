**Assignment 3**
Using Openmd to parallelize functions from assignment 2

**to run question 1**
sbatch task1.sh
-> Will produce a timing_data.txt file containing all outputs, which can be grpahed using task1.m
-> Graph file - task1.pdf

**to run question 2**
sbatch task2.sh
-> Will produce a timing_data_2.txt file containing all outputs, which can be grpahed using task2.m
-> Graph file - task2.pdf

**to run question 3**
Part1
sbatch task3_ts.sh
-> Will produce a timing_ts.txt file containing all outputs, which can be grpahed using task3_ts.m
-> Graph file - task3_ts.pdf

Part2
Compare the lowest execution time and threshold (in my case 256) and replace ts with that value in tas3_t.sh
sbatch task3_t.sh
-> Will produce a timing_t.txt file containing all outputs, which can be grpahed using task3_t.m
-> Graph file - task3_t.pdf



