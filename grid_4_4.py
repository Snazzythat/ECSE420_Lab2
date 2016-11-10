#ECSE420
#LAB2
#ANDONI ROMAN
#EXTRA PYTHON IMPLEMENTATION OF 4x4 GRID

#PYTHONIC DRUM 5.0

from mpi4py import MPI
import numpy as np
import sys

#GLOBALS
GRID_SIZE = 4
SIZE = GRID_SIZE * GRID_SIZE

#Node dictionary containing (i,j) of node as key and [t2,t1,t] as time values for each node
node_dict = {}
size = 0
rank = 0

#multiplicant constants
ETA = float(0.0002)
RHO = float(0.5)
G =	float(0.75)

#Dict that will get received by each node. Will contain all neigbouring values with their i and j
neigbour_nodes_and_their_values = {}
dots_per_process = 0
a=10

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
                            node_dict[(i,j)] = [0.0, 0.0, 0.1]
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
                    node_dict[(i, j)] = [0.0, 0.0, 0.1]
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
    elif(coord_tuple[0] >= GRID_SIZE or coord_tuple[1] >= GRID_SIZE):
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


#verify special case to distinguish the sending of current value or previous. Example, corners
#depend on current values of edges, hence edges need to send current value.
#Hence, when sending to EDGES or CORNERS, we need to send present value and not previous like
#in the middle case.
def if_corner(node_number):
    if node_number == 0 or node_number == GRID_SIZE - 1 or node_number == GRID_SIZE * (GRID_SIZE - 1) or node_number == (GRID_SIZE * GRID_SIZE) - 1:
        return True

    return False

def if_edge(node_number):
    if (node_number % GRID_SIZE == 0) or (node_number % GRID_SIZE == GRID_SIZE - 1) or ( node_number < GRID_SIZE) or  (node_number > GRID_SIZE * (GRID_SIZE - 1)):
        return True
    return False

def if_center(node_number):
    if node_number == 0 or node_number == GRID_SIZE - 1 or node_number == GRID_SIZE * (GRID_SIZE - 1) or node_number == (GRID_SIZE * GRID_SIZE) - 1:
        return False
    elif (node_number % GRID_SIZE == 0) or (node_number % GRID_SIZE == GRID_SIZE - 1) or ( node_number < GRID_SIZE) or  (node_number > GRID_SIZE * (GRID_SIZE - 1)):
        return False
    return True


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ITERATION + UPGRADE FOR CENTER

def do_center_iteration():

    key_list = node_dict.keys()

    #SENDING
    for node in key_list:

        i = node[0]
        j = node[1]

        my_node_number = get_node_num((i, j))

        if if_center(my_node_number) and size !=1:
            for k in range(0, 4):

                neighbour_coord = get_neighbor_coord_based_on_current_coordinate(i, j, k)
                neighbor_node_number = get_node_num(neighbour_coord)
                rank_of_neighbor_to_send_to = get_neighbor_rank(neighbour_coord, dots_per_process)

                #Dont send outside grid or to neighbors of same rank.
                if (rank_of_neighbor_to_send_to != -1):
                    if (rank_of_neighbor_to_send_to != rank):

                        #Also If its corner or edge also dont send
                        if (if_corner(neighbor_node_number) or if_edge(neighbor_node_number)):
                            continue
                        #DEBUG
                        #print('I am center with rank ' + str(rank) + ' at position ' + str(node) + ' and I send value to rank ' + str(rank_of_neighbor_to_send_to) + ' at position ' + str(neighbour_coord))
                        my_value_to_send = node_dict[(i, j)][2]
                        comm.send(my_value_to_send, dest=rank_of_neighbor_to_send_to, tag=neighbor_node_number)

    #RECEIVING
    for node in key_list:

        i = node[0]
        j = node[1]

        my_node_number = get_node_num((i, j))

        if if_center(my_node_number) and size !=1:
            for k in range(0, 4):

                neighbour_coord = get_neighbor_coord_based_on_current_coordinate(i, j, k)
                neighbor_node_number = get_node_num(neighbour_coord)
                rank_of_neighbor_to_receive_from = get_neighbor_rank(neighbour_coord, dots_per_process)

                # Dont send outside grid or to neighbors of same rank.
                if (rank_of_neighbor_to_receive_from != -1):
                    if (rank_of_neighbor_to_receive_from != rank):

                        # Do not receive values from corners, only U1 from edges and other nodes
                        if (if_corner(neighbor_node_number)):
                            continue
                        my_tag = get_node_num((i, j))
                        value_to_receive = comm.recv(source=rank_of_neighbor_to_receive_from, tag=my_tag)

                        neigbour_dict_keys = neigbour_nodes_and_their_values.keys()

                        # will create a new entry at first run if neighbour coordinate and value array is not
                        # in neighbour dictionary yet
                        if (not (neighbour_coord in neigbour_dict_keys)):
                            neigbour_nodes_and_their_values[neighbour_coord] = [0.0, 0.0, 0.0]
                            neigbour_nodes_and_their_values[neighbour_coord][2] = value_to_receive
                        else:
                            neigbour_nodes_and_their_values[neighbour_coord][2] = value_to_receive


    #UPGRADE PROCESS
    for node in key_list:

        i = node[0]
        j = node[1]

        my_node_number = get_node_num((i, j))

        neigbour_key_list = neigbour_nodes_and_their_values.keys()

        if if_center(my_node_number):
            # DEBUG
            # print('I,J belonging to center: ' + str((i,j)))
            left_neighbor_prev_val = 0.0
            right_neighbor_prev_val = 0.0
            lower_neighbor_prev_val = 0.0
            upper_neighbor_prev_val = 0.0
            # now treat every neighbour separately. Same analogy as with corners and edges
            # if neighbor is in neighbor dict, that means we received it from MPI
            # otherwise, the neighbor is in node_dict (belongs to same rank process)
            # shift first existing

            # check if neighbour is an edge or not If it is, then send after the newly updated value
            edges_that_need_new_value = []

            if (i - 1, j) in neigbour_key_list:
                left_neighbor_prev_val = neigbour_nodes_and_their_values[(i - 1, j)][2]
                neighbour_node_num = get_node_num((i - 1, j))
                if if_edge(neighbour_node_num):
                    edges_that_need_new_value.append((i - 1, j))
            else:
                left_neighbor_prev_val = node_dict[(i - 1, j)][2]

            if (i + 1, j) in neigbour_key_list:
                right_neighbor_prev_val = neigbour_nodes_and_their_values[(i + 1, j)][2]
                neighbour_node_num = get_node_num((i + 1, j))
                if if_edge(neighbour_node_num):
                    edges_that_need_new_value.append((i + 1, j))
            else:
                right_neighbor_prev_val = node_dict[(i + 1, j)][2]

            if (i, j - 1) in neigbour_key_list:
                lower_neighbor_prev_val = neigbour_nodes_and_their_values[(i, j - 1)][2]
                neighbour_node_num = get_node_num((i, j - 1))
                if if_edge(neighbour_node_num):
                    edges_that_need_new_value.append((i, j - 1))
            else:
                lower_neighbor_prev_val = node_dict[(i, j - 1)][2]

            if (i, j + 1) in neigbour_key_list:
                upper_neighbor_prev_val = neigbour_nodes_and_their_values[(i, j + 1)][2]
                neighbour_node_num = get_node_num((i, j + 1))
                if if_edge(neighbour_node_num):
                    edges_that_need_new_value.append((i, j + 1))
            else:
                upper_neighbor_prev_val = node_dict[(i, j + 1)][2]

            glorious_center_value = (RHO * (left_neighbor_prev_val + right_neighbor_prev_val +
                                            upper_neighbor_prev_val + lower_neighbor_prev_val - (
                                            4 * node_dict[(i, j)][2])) +
                                     2 * (node_dict[(i, j)][2]) - (1 - ETA) * (node_dict[(i, j)][1])) / ((1 + ETA))

            node_dict[(i, j)][0] = node_dict[(i, j)][1]
            node_dict[(i, j)][1] = node_dict[(i, j)][2]
            node_dict[(i, j)][2] = glorious_center_value

            if len(edges_that_need_new_value) != 0:
                for neighbor in edges_that_need_new_value:
                    rank_of_edge = get_neighbor_rank(neighbor, dots_per_process)
                    if (rank_of_edge != -1):
                        if (rank_of_edge != rank):
                            neighbour_node_num = get_node_num(neighbor)
                            comm.send(glorious_center_value, dest=rank_of_edge, tag=neighbour_node_num)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ITERATION + UPGRADE FOR EDGES


def do_edge_iteration():

    key_list = node_dict.keys()

    #SENDING
    for node in key_list:

        i = node[0]
        j = node[1]

        my_node_number = get_node_num((i, j))

        if if_edge(my_node_number) and size !=1:
            for k in range(0, 4):

                neighbour_coord = get_neighbor_coord_based_on_current_coordinate(i, j, k)
                neighbor_node_number = get_node_num(neighbour_coord)
                rank_of_neighbor_to_send_to = get_neighbor_rank(neighbour_coord, dots_per_process)

                #Dont send outside grid or to neighbors of same rank.
                if (rank_of_neighbor_to_send_to != -1):
                    if (rank_of_neighbor_to_send_to != rank):

                        #Also If its corner or edge, we dont send. We only send to center adjacent node.
                        if (if_corner(neighbor_node_number) or if_edge(neighbor_node_number)):
                            continue

                        rank_of_neighbor_to_send_to = get_neighbor_rank(neighbour_coord, dots_per_process)
                        my_value_to_send = node_dict[(i, j)][2]
                        comm.send(my_value_to_send, dest=rank_of_neighbor_to_send_to, tag=neighbor_node_number)

    #RECEIVING
    for node in key_list:

        i = node[0]
        j = node[1]

        my_node_number = get_node_num((i, j))

        if if_edge(my_node_number) and size !=1:
            for k in range(0, 4):

                neighbour_coord = get_neighbor_coord_based_on_current_coordinate(i, j, k)
                neighbor_node_number = get_node_num(neighbour_coord)
                rank_of_neighbor_to_receive_from = get_neighbor_rank(neighbour_coord, dots_per_process)

                # Dont receive from outside the grid
                if (rank_of_neighbor_to_receive_from != -1):
                    if (rank_of_neighbor_to_receive_from != rank):

                        # Do not receive values from corners or edges nodes. ONLY RECEIVE NEW VALUE from CENTER NODES
                        if (if_corner(neighbor_node_number) or if_edge(neighbor_node_number)):
                            continue

                        my_tag = get_node_num((i, j))
                        value_to_receive = comm.recv(source=rank_of_neighbor_to_receive_from, tag=my_tag)

                        neigbour_dict_keys = neigbour_nodes_and_their_values.keys()

                        # will create a new entry at first run if neighbour coordinate and value array is not
                        # in neighbour dictionary yet
                        if (not(neighbour_coord in neigbour_dict_keys)):
                            neigbour_nodes_and_their_values[neighbour_coord] = [0.0, 0.0, 0.0]
                            neigbour_nodes_and_their_values[neighbour_coord][2] = value_to_receive
                        else:
                            neigbour_nodes_and_their_values[neighbour_coord][2] = value_to_receive


    #UPGRADE PROCESS

    # left most edge case --> get all nodes with i at 0 and j starting from 1 till N-2
    for j in range(1, GRID_SIZE - 1):

        neigbour_key_list = neigbour_nodes_and_their_values.keys()

        if (0, j) in key_list:

            # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
            if (1, j) in neigbour_key_list:
                update_value = G * neigbour_nodes_and_their_values[(1, j)][2]
            else:
                # if neighbor is not in neighbor_key_list that means it belongs to current process
                update_value = G * node_dict[(1, j)][2]

            # shift first existing, then update the latest value
            node_dict[(0, j)][0] = node_dict[(0, j)][1]
            node_dict[(0, j)][1] = node_dict[(0, j)][2]
            node_dict[(0, j)][2] = update_value

            # now need to send to lower left right corner the update value
            if (j == GRID_SIZE - 2):
                rank_of_neighbor_to_send_to = get_neighbor_rank((0, GRID_SIZE - 1), dots_per_process)
                if (rank_of_neighbor_to_send_to != -1):
                    if (rank_of_neighbor_to_send_to != rank):
                        neighbour_node_num = get_node_num((0, GRID_SIZE - 1))
                        #verify again if corner
                        if if_corner(neighbour_node_num):
                            comm.send(update_value, dest=rank_of_neighbor_to_send_to, tag=neighbour_node_num)
                            #print('Im an edge node with rank ' + str(rank) + ' and I sent values to corner with rank ' + str(rank_of_neighbor_to_send_to))

        # right  most edge case --> get all nodes with i at N-1 j starting from 1 till N-2
        if (GRID_SIZE - 1, j) in key_list:

            # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
            if (GRID_SIZE - 2, j) in neigbour_key_list:
                update_value = G * neigbour_nodes_and_their_values[(GRID_SIZE - 2, j)][2]
            else:
                # if neighbor is not in neighbor_key_list that means it belongs to current process
                update_value = G * node_dict[(GRID_SIZE - 2, j)][2]

            # shift first existing, then update the latest value
            node_dict[(GRID_SIZE - 1, j)][0] = node_dict[(GRID_SIZE - 1, j)][1]
            node_dict[(GRID_SIZE - 1, j)][1] = node_dict[(GRID_SIZE - 1, j)][2]
            node_dict[(GRID_SIZE - 1, j)][2] = update_value

            # now need to send to corner lower right corner the update value
            if (j == GRID_SIZE - 2):
                rank_of_neighbor_to_send_to = get_neighbor_rank((GRID_SIZE - 1, GRID_SIZE - 1), dots_per_process)
                if (rank_of_neighbor_to_send_to != -1):
                    if (rank_of_neighbor_to_send_to != rank):
                        neighbour_node_num = get_node_num((GRID_SIZE - 1, GRID_SIZE - 1))
                        if if_corner(neighbour_node_num):
                            comm.send(update_value, dest=rank_of_neighbor_to_send_to, tag=neighbour_node_num)
                            #print('Im an edge node with rank ' + str(rank) + ' and I sent values to corner with rank ' + str(rank_of_neighbor_to_send_to))

    # right top edge case --> get all nodes with j at 0 and x starting from 1 till N-2
    for i in range(1, GRID_SIZE - 1):

        neigbour_key_list = neigbour_nodes_and_their_values.keys()

        if (i, 0) in key_list:

            # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
            if (i, 1) in neigbour_key_list:
                update_value = G * neigbour_nodes_and_their_values[(i, 1)][2]
            else:
                # if neighbor is not in neighbor_key_list that means it belongs to current process
                update_value = G * node_dict[(i, 1)][2]

            # shift already existing, then update the latest value
            node_dict[(i, 0)][0] = node_dict[(i, 0)][1]
            node_dict[(i, 0)][1] = node_dict[(i, 0)][2]
            node_dict[(i, 0)][2] = update_value

            # now need to send to corner upper left corner the update value
            if (i == 1):
                rank_of_neighbor_to_send_to = get_neighbor_rank((0, 0), dots_per_process)
                if (rank_of_neighbor_to_send_to != -1):
                    if (rank_of_neighbor_to_send_to != rank):
                        neighbour_node_num = get_node_num((0, 0))
                        if if_corner(neighbour_node_num):
                            comm.send(update_value, dest=rank_of_neighbor_to_send_to, tag=neighbour_node_num)
                            #print('Im an edge node with rank ' + str(rank) + ' and I sent values to corner with rank ' + str(rank_of_neighbor_to_send_to))

                        # now need to send to corner upper left corner the update value

            if (i == GRID_SIZE - 2):
                rank_of_neighbor_to_send_to = get_neighbor_rank((GRID_SIZE - 1, 0), dots_per_process)
                if (rank_of_neighbor_to_send_to != -1):
                    if (rank_of_neighbor_to_send_to != rank):
                        neighbour_node_num = get_node_num((GRID_SIZE - 1, 0))
                        if if_corner(neighbour_node_num):
                            comm.send(update_value, dest=rank_of_neighbor_to_send_to, tag=neighbour_node_num)
                        #print('Im an edge node with rank ' + str(rank) + ' and I sent values to corner with rank ' + str(rank_of_neighbor_to_send_to))

        # bottom top edge case --> get all nodes with j at 0 and x starting from 1 till N-2
        if (i, GRID_SIZE - 1) in key_list:

            # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
            if (i, GRID_SIZE - 2) in neigbour_key_list:
                update_value = G * neigbour_nodes_and_their_values[(i, GRID_SIZE - 2)][2]
            else:
                # if neighbor is not in neighbor_key_list that means it belongs to current process
                update_value = G * node_dict[(i, GRID_SIZE - 2)][2]

            # shift already existing, then update the latest value
            node_dict[(i, GRID_SIZE - 1)][0] = node_dict[(i, GRID_SIZE - 1)][1]
            node_dict[(i, GRID_SIZE - 1)][1] = node_dict[(i, GRID_SIZE - 1)][2]
            node_dict[(i, GRID_SIZE - 1)][2] = update_value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ITERATION + UPGRADE FOR CORNERS


#each corner only receives one value and never sends.
def do_corner_iteration():

    key_list = node_dict.keys()

    for node in key_list:

        i = node[0]
        j = node[1]

        my_node_number = get_node_num((i, j))

        if if_corner(my_node_number)and size !=1:

            for k in range(0, 4):

                neighbour_coord = get_neighbor_coord_based_on_current_coordinate(i, j, k)
                neighbor_node_number = get_node_num(neighbour_coord)
                rank_of_neighbor_to_receive_from = get_neighbor_rank(neighbour_coord, dots_per_process)

                # DEBUG
                #print('Im a corner node with node number ' + str(my_node_number) + ' and one of my neighbors has rank ' + str(rank_of_neighbor_to_receive_from))
                # Dont receive from outside the grid
                if (rank_of_neighbor_to_receive_from != -1):
                    if (rank_of_neighbor_to_receive_from != rank):

                        # Do not receive values from corners or center, only from edges
                        if (if_corner(neighbor_node_number) or if_center(neighbor_node_number)):
                            continue

                        my_tag = get_node_num(node)

                        #each corner will receive from specific edge, not from both. See the formulas for restriction
                        #upper left right receive from upper edge
                        #lower left right receive from left and right edges
                        if ((my_tag ==0  and neighbor_node_number ==1) or \
                            (my_tag == (GRID_SIZE - 1)  and neighbor_node_number == GRID_SIZE - 2) or \
                            (my_tag == GRID_SIZE * (GRID_SIZE - 1) and neighbor_node_number ==  (GRID_SIZE * (GRID_SIZE - 1))-GRID_SIZE) or \
                            (my_tag == (GRID_SIZE * GRID_SIZE) - 1 and neighbor_node_number ==  GRID_SIZE * (GRID_SIZE - 1) -1)):

                                value_to_receive = comm.recv(source=rank_of_neighbor_to_receive_from, tag=my_tag)
                                #DEBUG
                                #print('Im a corner node with rank ' + str(rank) + ' and I received values from edge with rank ' + str(rank_of_neighbor_to_receive_from))
                                neigbour_dict_keys = neigbour_nodes_and_their_values.keys()

                                # will create a new entry at first run if neighbour coordinate and value array is not
                                # in neighbour dictionary yet
                                if (not (neighbour_coord in neigbour_dict_keys)):
                                    neigbour_nodes_and_their_values[neighbour_coord] = [0.0, 0.0, 0.0]
                                    neigbour_nodes_and_their_values[neighbour_coord][2] = value_to_receive
                                else:
                                    neigbour_nodes_and_their_values[neighbour_coord][2] = value_to_receive

    neigbour_key_list = neigbour_nodes_and_their_values.keys()

    if (0, 0) in key_list:

        # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
        if ((1, 0) in neigbour_key_list):
            update_value = G * neigbour_nodes_and_their_values[(1, 0)][2]
            # if neighbor is not in neighbor_key_list that means it belongs to current process
        else:
            update_value = G * node_dict[(1, 0)][2]

        # shift first existing
        node_dict[(0, 0)][0] = node_dict[(0, 0)][1]
        node_dict[(0, 0)][1] = node_dict[(0, 0)][2]
        node_dict[(0, 0)][2] = update_value

        # case upper right corner. depends on previous value of node i=N-2,j=0
    elif (GRID_SIZE - 1, 0) in key_list:

        # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
        if ((GRID_SIZE - 2, 0) in neigbour_key_list):
            update_value = G * neigbour_nodes_and_their_values[(GRID_SIZE - 2, 0)][2]
            # if neighbor is not in neighbor_key_list that means it belongs to current process
        else:
            update_value = G * node_dict[(GRID_SIZE - 2, 0)][2]

        # shift first existing
        node_dict[(GRID_SIZE - 1, 0)][0] = node_dict[(GRID_SIZE - 1, 0)][1]
        node_dict[(GRID_SIZE - 1, 0)][1] = node_dict[(GRID_SIZE - 1, 0)][2]
        node_dict[(GRID_SIZE - 1, 0)][2] = update_value

        # case lower  left corner. depends on previous value of node i=0,j=N-2
    elif (0, GRID_SIZE - 1) in key_list:

        # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
        if ((0, GRID_SIZE - 2) in neigbour_key_list):
            update_value = G * neigbour_nodes_and_their_values[(0, GRID_SIZE - 2)][2]
            # if neighbor is not in neighbor_key_list that means it belongs to current process
        else:
            update_value = G * node_dict[(0, GRID_SIZE - 2)][2]

        # shift first existing
        node_dict[(0, GRID_SIZE - 1)][0] = node_dict[(0, GRID_SIZE - 1)][1]
        node_dict[(0, GRID_SIZE - 1)][1] = node_dict[(0, GRID_SIZE - 1)][2]
        node_dict[(0, GRID_SIZE - 1)][2] = update_value

        # case lower right corner. depends on previous value of node i=N-1,j=N-2
    elif (GRID_SIZE - 1, GRID_SIZE - 1) in key_list:

        # update the latest value based if the value came from MPI (in neigbors dict) or it is in same node_dict
        if ((GRID_SIZE - 1, GRID_SIZE - 2) in neigbour_key_list):
            update_value = G * neigbour_nodes_and_their_values[(GRID_SIZE - 1, GRID_SIZE - 2)][2]
        else:
            # if neighbor is not in neighbor_key_list that means it belongs to current process
            update_value = G * node_dict[(GRID_SIZE - 1, GRID_SIZE - 2)][2]

        # shift first existing
        node_dict[(GRID_SIZE - 1, GRID_SIZE - 1)][0] = node_dict[(GRID_SIZE - 1, GRID_SIZE - 1)][1]
        node_dict[(GRID_SIZE - 1, GRID_SIZE - 1)][1] = node_dict[(GRID_SIZE - 1, GRID_SIZE - 1)][2]
        node_dict[(GRID_SIZE - 1, GRID_SIZE - 1)][2] = update_value



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN


if __name__ == "__main__":

    iterations = int(sys.argv[1])

    #MPI inits
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    rank = rank

    dots_per_process = (GRID_SIZE*GRID_SIZE / size)


    #setup dict for each node. The dict will contain (i,j) tuple as key and array [n2,n1,n0]
    #for t-2,t-1,t values
    fill_node_dict(dots_per_process,rank)

    # For each of iteration, do 3 updates. First center, then edges, then corners.
    # Then print the value at N/2 N/2 where strike was applied

    output_string = ''
    for iteration in range(1,iterations+1):

        do_center_iteration()

        do_edge_iteration()

        do_corner_iteration()

        #STRIKE PRINT
        #Only process that has it will print
        key_list = node_dict.keys()
        if((GRID_SIZE/2,GRID_SIZE/2) in key_list):
            #strike value is the value we get from node dict. The value is the array with n2,n1,n0. We display n0
            #the latest (current) strike value
            strike_value = a*node_dict[(GRID_SIZE/2,GRID_SIZE/2)][2]
            output_string = str(strike_value)
            if iteration != iterations:
                output_string = output_string + ','
            print(output_string)

        #shift now all for next iteration
    comm.barrier()

    MPI.Finalize()

