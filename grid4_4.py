from mpi4py import MPI
import numpy as np
import sys


GRID_SIZE = 4

SIZE = GRID_SIZE * GRID_SIZE

node_dict = {}

size = 0



#Fills up all information necessary for node transactions
#Each node will contain its position i j and an array with 4 variables
#that contain values from neighbours

#In case of 4x4 we will have 1 dictionary assigned since one dot is assigned to a process
#Otehrwise, we will have a many variables due to the fact that we assign a cluster of dots
#to a process. Dictionary has (i,j) position as key and [n2,n1,n0] as value
#indicating previous-previous value, previous value, current value
#STRIKE value is set to be n2 at beginning at node N/2,N/2 --> [0,0,1]
def fill_node_dict(dot_amount_per_process, rank):

    # case N=4
    if dots_per_process == 1:
        for z in range (0, dots_per_process):
            rank_counter = -1
            for j in range (0,GRID_SIZE):
                for i in range(0, GRID_SIZE):
                    rank_counter += 1
                    if rank_counter == rank:
                        if (i == GRID_SIZE/2) and (j == GRID_SIZE/2):
                            node_dict[(i,j)] = [0, 0, 1]
                        else:
                            node_dict[(i,j)] = [0, 0, 0]
    #case N=512
    else:
        offset = GRID_SIZE/size
        beginning = (offset * rank)
        end = beginning + offset
        for j in range(beginning, end):
            for i in range(0, GRID_SIZE):
                if (i == GRID_SIZE / 2) and (j == GRID_SIZE / 2):
                    node_dict[(i, j)] = [1, 0, 0]
                else:
                    node_dict[(i, j)] = [0, 0, 0]



#simulation iteration method. Will update nodes and chose nodes based on condition provided
#default is non but specified from calls from main. Can be: EDGE, CENTER, CORNER
def simulate_iteration(condition = None):





if __name__ == "__main__":

    iterations = int(sys.argv[1])

    #MPI inits
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    dots_per_process = (GRID_SIZE*GRID_SIZE / size)

    #setup dict for each node. The dict will contain (i,j) tuple as key and array [n2,n1,n0]
    #for t-2,t-1,t values
    fill_node_dict(dots_per_process,rank)

     # print "I'm rank " + str(rank) + " and my grid dict is" + str(node_dict) + '\n\n'

    # For each of iteration, do 3 updates. First center, then edges, then corners.
    # Then print the value at N/2 N/2 where strike was applied
    for iteration in range(0,iterations + 1):

        simulate_iteration(condition = "CENTER")

        simulate_iteration(condition = "EDGE")

        simulate_iteration(condition = "CORNER")

        #STRIKE PRINT
        key_list = node_dict.keys()
        if((GRID_SIZE/2,GRID_SIZE/2) in key_list):
            #strike value is the value we get from node dict. The value is the array with n2,n1,n0. We display n0
            #the latest (current) strike value
            strike_value = node_dict[(GRID_SIZE/2,GRID_SIZE/2)][2]
            print('Strike value at iteration ' + str(iteration) + ' is ' + str(strike_value))

        iterations -= 1



