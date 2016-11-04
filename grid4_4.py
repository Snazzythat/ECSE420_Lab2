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
dots_per_process = 0

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

#Returns the number of the node in 2D array starting from 0 to N-1*N-1
def get_node_num(coord_tuple):
    #eliminate the boundary conditions first
    num = 0

    if(coord_tuple[0] < 0 or coord_tuple[1]  < 0):
        num = -1
    elif(coord_tuple[0] >= GRID_SIZE or coord_tuple[0] >= GRID_SIZE):
        num = -1
    else:
        rank_counter = -1
        for j in range(0, GRID_SIZE):
            for i in range(0, GRID_SIZE):
                rank_counter += 1
                if(coord_tuple[0]==i and coord_tuple[1]==j):
                    num = rank_counter
    return num


#Returns the ranks of the process that contains the specified neighbor coordinate
def get_neighbor_rank(coord_tuple, dots_per_process):
    #eliminate the boundary conditions first
    rank = 0

    if(coord_tuple[0] < 0 or coord_tuple[1]  < 0):
        rank = -1
    elif(coord_tuple[0] >= GRID_SIZE or coord_tuple[0] >= GRID_SIZE):
        rank = -1
    else:
        rank_counter = -1
        for j in range(0, GRID_SIZE):
            for i in range(0, GRID_SIZE):
                rank_counter += 1
                if(coord_tuple[0]==i and coord_tuple[1]==j):
                    rank = rank_counter
                    rank = int(rank/dots_per_process)
    return rank

#Returns neigbour coordinate based on current coordinate
# k ==>  0:left 1:right 2: up 3: down
def get_neighbor_coord_based_on_current_coordinate(i,j,k):

    neighbor_coordinate = (-1,-1)

    if k == 0:
        neighbor_coordinate = (i-1,j)
    elif k == 1:
        neighbor_coordinate = (i+1,j)
    elif k == 2:
        neighbor_coordinate = (i, j+1)
    elif k == 3:
        neighbor_coordinate = (i, j-1)

    return neighbor_coordinate

#Based on conditions verifies if the node provided the node number needs update
#Returns true if update needed
def check_if_needs_update(condition, node_number):

    #CORNER CASE
    if (node_number == 0 or node_number == GRID_SIZE - 1 or node_number == GRID_SIZE * (GRID_SIZE - 1) or node_number == ( GRID_SIZE * GRID_SIZE - 1)):
        if(condition != "CORNER"):
            return False
    elif (node_number % GRID_SIZE == 0 or node_number % GRID_SIZE == GRID_SIZE - 1 or  node_number < GRID_SIZE or  node_number > GRID_SIZE * (GRID_SIZE - 1)):
        if(condition != "EDGE"):
            return False
    else:
        if(condition != "CENTER"):
            return False
    return True



def exchange_data_with_neighbors(rank, buffer, operation=None, condition=None):

    counter = 0
    key_list = node_dict.keys()

    for i in range (0, GRID_SIZE):
        for j in range (0, GRID_SIZE):

            if ((i,j) in key_list):
                kcount = 0
                #check for neighbors 0:left 1:right 2: up 3: down
                for k in range (0,4):

                    #getting neighbors coordinate
                    neighbour_coord = get_neighbor_coord_based_on_current_coordinate(i,j,k)

                    #SENDING TO NEIGHBORS
                    if operation == 'send':
                        #Getting neighbors rank
                        rank_of_neighbor_to_send_to = get_neighbor_rank(neighbour_coord,dots_per_process)

                        #if not outside the grid and not same rank as current process
                        #then send
                        #Applies to one process per dot or many dots per process
                        if ((rank_of_neighbor_to_send_to != -1) and (rank_of_neighbor_to_send_to != rank)):

                            neighbour_node_num = get_node_num(neighbour_coord)

                            if (check_if_needs_update(condition, neighbour_node_num) == False):
                                continue
                            #otherwise, the neighbour belongs to other process. Need to send.
                            #get our own value and then send to the neighbor
                            #send the previous value (t-1) value
                            my_value_to_send = node_dict[(i,j)][1]
                            my_tag = get_node_num((i,j))
                            #MPI SEND
                            comm.send(my_value_to_send, dest=rank_of_neighbor_to_send_to, tag=my_tag);
                            counter += 1

                    # RECEIVING FROM NEIGHBORS
                    elif operation == 'recv':
                        rank_of_neighbor_to_receive_from = get_neighbor_rank(neighbour_coord,dots_per_process)
                        # if not outside the grid and not same rank as current process
                        # then receive
                        # Applies to one process per dot or many dots per process
                        if ((rank_of_neighbor_to_receive_from != -1) and (rank_of_neighbor_to_receive_from != rank)):

                            my_node_number = get_node_num((i,j))
                            if (check_if_needs_update(condition, my_node_number) == False):
                                continue
                            #otherwise we NEED update so need to receive from process
                            neighbours_tag = get_node_num((neighbour_coord))
                            value_to_receive = comm.recv(source =rank_of_neighbor_to_receive_from, tag = neighbours_tag)
                            #updating neighbour dict with t-1 value of neighbour at (i,j)
                            neigbour_dict_keys = neigbour_nodes_and_their_values.keys()

                            #will create a new entry at first run if neighbour coordinate and value array is not
                            #in neighbour dictionary yet
                            if(not(neighbour_coord in neigbour_dict_keys)):
                                neigbour_nodes_and_their_values[neighbour_coord] = [0.0,0.0,0.0]

                            neigbour_nodes_and_their_values[neighbour_coord][1]=value_to_receive

                        #Means those nodes belong to US (1 process with many dots). Simply distribute between neighbours
                        #without MPI sending.
                        elif rank_of_neighbor_to_receive_from == rank:
                            value_to_receive = node_dict[neighbour_coord][1]








#simulation iteration method. Will update nodes and chose nodes based on condition provided
#default is non but specified from calls from main. Can be: EDGE, CENTER, CORNER
def simulate_iteration(rank,condition = None):

    #All received values from the neighbours will be stored here
    #will always have 4 structures with  values in it:
        #UPPER NEIGBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
        #LOWER NEIGBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
        #LEFT NEIGHBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
        #RIGHT NEIGHBOURS i,j as key and their array with PREV-PREV, PREV, CURRENT values
    #depending on condition, there will be special cases (ex: corner cases)

    exchange_data_with_neighbors(rank, {}, operation= 'send', condition = condition)

    exchange_data_with_neighbors(rank, {},  operation = 'recv', condition = condition)

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
                    #update to most recent value
                    node_dict[(i,j)][2] = glorious_center_value



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

        simulate_iteration(rank,condition = "CENTER")

        simulate_iteration(rank,condition = "EDGE")

        simulate_iteration(rank,condition = "CORNER")

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

