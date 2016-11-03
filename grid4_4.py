#ECSE420
#LAB2
#ANDONI ROMAN
#YORDAN NESHEV

#PYTHONIC DRUM 1.0


from mpi4py import MPI
import numpy as np
import sys

#GLOBALS
GRID_SIZE = 4
SIZE = GRID_SIZE * GRID_SIZE

#Node dictionary containing (i,j) of node as key and [t2,t1,t] as time values for each node
node_dict = {}
size = 0

#multiplicant constants
ETA = float(0.0002)
RHO = float(0.5)
G =	float(0.75)

#Dict that will get received by each node. Will contain all neigbouring values with their i and j
neigbour_nodes_and_their_values = {}

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
                            node_dict[(i,j)] = [0.0, 0.0, 0.0]
                        else:
                            node_dict[(i,j)] = [0.0, 0.0, 0.0]
    #case N=512
    else:
        offset = GRID_SIZE/size
        beginning = (offset * rank)
        end = beginning + offset
        for j in range(beginning, end):
            for i in range(0, GRID_SIZE):
                if (i == GRID_SIZE / 2) and (j == GRID_SIZE / 2):
                    node_dict[(i, j)] = [0.0, 0.0, 0.0]
                else:
                    node_dict[(i, j)] = [0.0, 0.0, 0.0]



#simulation iteration method. Will update nodes and chose nodes based on condition provided
#default is non but specified from calls from main. Can be: EDGE, CENTER, CORNER
def simulate_iteration(condition = None):

    #All received values from the neighbours will be stored here
    #will always have 4 structures with  values in it:
        #UPPER NEIGBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
        #LOWER NEIGBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
        #LEFT NEIGHBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
        #RIGHT NEIGHBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
    #depending on condition, there will be special cases (ex: corner cases)

    #exchange data with neighbours code here (MPI)

    #receive data from neigbours code here (MPI)

    #Now update depending on condition
    #get now list of all keys of nodes  that belong to process
    #Based on this key list, each process will update only those nodes that belong to it
    #So we check the key first, if its in processes pool of keys.
    #If the key is in the pool, check the received neighbor values.
    #Take the value (t2, t1 or t0) basing on the case (corner, middle or edges)
    # ex neighbor_nodes_and_their_values[(1,0)][2] means we get the most recent value (position 2 in value array)
    #  of  neighbor that is situated at 1,0

    key_list = node_dict.keys()

    neigbour_key_list = neigbour_nodes_and_their_values.keys()

    if(condition == "CORNER"):
        #case upper left corner. depends on previous value of node i=1,j=0
        if (0,0) in key_list:
            if ((1,0) in neigbour_key_list):
                update_value = G * neigbour_nodes_and_their_values[(1,0)][2]
                node_dict[(0,0)][2] = update_value

        # case upper right corner. depends on previous value of node i=N-2,j=0
        elif (GRID_SIZE-1, 0) in key_list:
            if ((GRID_SIZE-2, 0) in neigbour_key_list):
                update_value = G * neigbour_nodes_and_their_values[(GRID_SIZE-2, 0)][2]
                node_dict[(GRID_SIZE-1, 0)][2] = update_value

        # case lower  left corner. depends on previous value of node i=0,j=N-2
        elif (0, GRID_SIZE-1) in key_list:
            if ((0, GRID_SIZE-2) in neigbour_key_list):
                update_value = G * neigbour_nodes_and_their_values[(0, GRID_SIZE-2)][2]
                node_dict[(0,GRID_SIZE-1)][2] = update_value

        # case lower right corner. depends on previous value of node i=N-1,j=N-2
        elif (GRID_SIZE-1, GRID_SIZE-1) in key_list:
            if ((GRID_SIZE-1, GRID_SIZE-2) in neigbour_key_list):
                update_value = G * neigbour_nodes_and_their_values[(GRID_SIZE-1, GRID_SIZE-2)][2]
                node_dict[(GRID_SIZE-1,GRID_SIZE-1)][2] = update_value


    elif(condition == "EDGE"):
        #left most edge case --> get all nodes with i at 0 and j starting from 1 till N-2
        for j in range(1,GRID_SIZE-1):
            if (0,j) in key_list:
                if(1,j) in neigbour_key_list:
                    update_value = G * neigbour_nodes_and_their_values[(1,j)][2]
                    node_dict[(0,j)][2] = update_value

        #right  most edge case --> get all nodes with i at N-1 j starting from 1 till N-2
        for j in range(1,GRID_SIZE-1):
            if (GRID_SIZE-1,j) in key_list:
                if (GRID_SIZE-2,j) in neigbour_key_list:
                    update_value = G * neigbour_nodes_and_their_values[(GRID_SIZE-2,j)][2]
                    node_dict[(GRID_SIZE-1,j)][2] = update_value

        #right top edge case --> get all nodes with j at 0 and x starting from 1 till N-2
        for i in range(1,GRID_SIZE-1):
            if (i,0) in key_list:
                if (i,1) in neigbour_key_list:
                    update_value = G * neigbour_nodes_and_their_values[(i,1)][2]
                    node_dict[(i,0)][2] = update_value

        #bottom top edge case --> get all nodes with j at 0 and x starting from 1 till N-2
        for i in range(1,GRID_SIZE-1):
            if (i,GRID_SIZE-1) in key_list:
                if (i,GRID_SIZE-2) in neigbour_key_list:
                    update_value = G * neigbour_nodes_and_their_values[(i,GRID_SIZE-2)][2]
                    node_dict[(i,GRID_SIZE-1)][2] = update_value

    elif(condition == "CENTER"):
        #CENTER CASE
        #All i's are situated between 1 and N-2
        #All j's are situated between 1 and N-2
        #Get all 4 neigbours values and then update the value using the formula
        for i in range(1,GRID_SIZE-1):
            for j in range(1,GRID_SIZE-1):
                if(i,j) in key_list:

                    left_neigbour_prev_val = 0
                    right_neighbour_prev_val = 0
                    lower_neighbour_prev_val = 0
                    upper_neighbour_prev_val = 0
                    #now treat every neighbour separately
                    if(i-1,j) in neigbour_key_list:
                        left_neigbour_prev_val = neigbour_nodes_and_their_values[(i-1,j)][1]
                    if(i+1,j) in neigbour_key_list:
                        right_neighbour_prev_val = neigbour_nodes_and_their_values[(i+1,j)][1]
                    if(i, j-1) in neigbour_key_list:
                        lower_neighbour_prev_val = neigbour_nodes_and_their_values[(i, j-1)][1]
                    if(i, j+1) in neigbour_key_list:
                        upper_neighbour_prev_val = neigbour_nodes_and_their_values[(i, j+1)][1]

                    glorious_center_value = (RHO * (left_neigbour_prev_val + right_neighbour_prev_val + \
                            upper_neighbour_prev_val + lower_neighbour_prev_val - (4*node_dict[(i,j)][1])) + \
                            2*(node_dict[(i,j)][1]) - (1 - ETA)*(node_dict[(i,j)][0]))/(1-ETA)



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


    # For each of iteration, do 3 updates. First center, then edges, then corners.
    # Then print the value at N/2 N/2 where strike was applied
    for iteration in range(0,iterations + 1):

        simulate_iteration(condition = "CENTER")

        simulate_iteration(condition = "EDGE")

        simulate_iteration(condition = "CORNER")

        print "I'm rank " + str(rank) + " and my grid dict is" + str(node_dict) + '\n\n'

        #STRIKE PRINT
        key_list = node_dict.keys()
        if((GRID_SIZE/2,GRID_SIZE/2) in key_list):
            #strike value is the value we get from node dict. The value is the array with n2,n1,n0. We display n0
            #the latest (current) strike value
            strike_value = node_dict[(GRID_SIZE/2,GRID_SIZE/2)][2]
            print('Strike value at iteration ' + str(iteration) + ' is ' + str(strike_value))

        iterations -= 1

        #at each iteration also shift previous values.
    MPI.Finalize()

