#ECSE420
#LAB2 --> EXTRA PYTHON IMPLEMENTATION OF 512x512 GRID
#ANDONI ROMAN

#PYTHONIC DRUM 1.0

from mpi4py import MPI
import numpy as np
import sys
import output

######################################################################################## GLOBALS
GRID_SIZE = 512
UNIVERSAL_TAG = 0
mpi_world_size = 0
rank = 0

# multiplicant constants
ETA = 0.0002
RHO = 0.5
G = 0.75
cnt = 0

# define the necessary grids such as the current grid,
# the new value grid, the previous value and prev-prev value
# grids to keep track of values.
u = []
u1 = []
u2 = []


received_all = False

######################################################################################## SETS up a fresh gris
def setup_value_grid():
    for i in range (0,GRID_SIZE):
        list = []
        for j in range(0,GRID_SIZE):
            list.append(0.0)
        u.append(list)
        u1.append(list)
        u2.append(list)


######################################################################################## 'PUNCHES' the surface

def insert_perturbation():
    u[GRID_SIZE/2][GRID_SIZE/2] = 1

    #TESTING PURPOSES
    # if rank == 0:
    #     u1[first_row_in_cluster][0] = 99999
    #     u1[last_row_in_cluster][0] = 7887
    # if rank == 1:
    #     u1[first_row_in_cluster][0] = 111
    #     u1[last_row_in_cluster][0] = 1
    #
    # if rank == 2:
    #     u1[first_row_in_cluster][0] = 222
    #     u1[last_row_in_cluster][0] = 2
    #
    # if rank == 3:
    #     u1[first_row_in_cluster][0] = 333
    #     u1[last_row_in_cluster][0] = 3


######################################################################################## UPDATE function

######################################################################################## SEND+RECV function

#Send Receive function responsible to send and receive rows as buffers and
#insert them into the grid as currently received values. Depending on the rank of the process,
# the sending and receiving is done for specific cases.
#The process with rank 0 and size-1 will send to rank 1 and size-2 ranks respectively.
#The processes in the middle between 0 and last rank will send both to upper and lowe rank processes.
def send_receive(first_row_in_cluster, last_row_in_cluster):

    upper_neighbor_row = []
    lower_neighbor_row = []
    # all processes but the one in last lank send their first rows
    if (rank != mpi_world_size - 1):
        dest_above = rank + 1
        obj_to_send = {'ghost_row':u1[last_row_in_cluster] }
        comm.send(obj_to_send, dest=rank+1, tag=UNIVERSAL_TAG)

        #print('I am rank ' + str(rank) + 'and I send my last row'+ str(u1[last_row_in_cluster]) + ' to rank ' + str(rank +1))


    # all processes but the firts one receive rows in the upper ghost rows
    if (rank != 0):
        source_below = rank - 1
        data = comm.recv(source=rank-1, tag=UNIVERSAL_TAG)
        ghost_row = data.get('ghost_row')
        #u1[first_row_in_cluster - 1] = ghost_row
        upper_neighbor_row = ghost_row

        #print('I am rank' + str(rank) + ' and I received from '+ str(rank-1) + 'row  ' + str(ghost_row))

    # send all first rows in the clusters but the first rank (only sends last one)
    if (rank != 0):
        dest_above = rank - 1
        obj_to_send = {'ghost_row': u1[first_row_in_cluster]}
        comm.send(obj_to_send, dest=rank-1, tag=UNIVERSAL_TAG)
        #print('I am rank ' + str(rank) + 'and I send my first row' + str(u1[first_row_in_cluster]) + ' to rank ' + str(rank - 1))

    # all processes but the one in last lank receive in lower ghost rows
    if (rank != mpi_world_size - 1):
        source_above = rank + 1
        data = comm.recv(source=rank+1, tag=UNIVERSAL_TAG)
        ghost_row = data.get('ghost_row')
        #u1[last_row_in_cluster + 1] = ghost_row
        #print('I am rank' + str(rank) + ' and I received from ' + str(rank + 1) + 'row  ' + str(ghost_row))
        lower_neighbor_row = ghost_row
    comm.barrier()

    return (upper_neighbor_row, lower_neighbor_row)

######################################################################################## MEAN SQUARE ERROR FN
# gets the value from the output array, does square error comparison and sees if greater than (0.00001)
# as provided in lab
def do_error_calc(first_row_in_cluster,last_row_in_cluster,cnt):
    if ((first_row_in_cluster <= GRID_SIZE/2) and (last_row_in_cluster >= GRID_SIZE/2)):
        if (((u[GRID_SIZE/2][GRID_SIZE/2] - comparable_output[t]) * (u[GRID_SIZE/2][GRID_SIZE/2] - comparable_output[t])) > 0.00001):
            cnt += 1
            print(str(comparable_output[t]) + ',')
            return cnt

######################################################################################## PRESENCE CHECKER
#Checks if current node is in the current process
def any_in_current_process(i):

    for m in range(first_row_in_cluster, last_row_in_cluster+1):
        if m ==i:
            return True
    return False



######################################################################################## MAIN
if __name__ == "__main__":

    #init all MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_world_size = comm.Get_size()
    status = MPI.Status()

    received_all = False

    #get number of iterations
    iterations = int(sys.argv[1])

    #define constants necessary (offsets)
    process_row_amount = GRID_SIZE / mpi_world_size
    #print('I am rank %s and I have %s rows'%(rank, process_row_amount))
    first_row_in_cluster = rank * process_row_amount
    last_row_in_cluster = first_row_in_cluster + process_row_amount - 1 #need a ghost row
    #print('I am rank %s and my first row is %s and my last row is %s') % (rank, first_row_in_cluster, last_row_in_cluster)

    #setup an empty value grid
    setup_value_grid()

    #insert punch at N/2,N/2
    insert_perturbation()

    #iterate
    for t in range (0,iterations):

        upper_received_row = []
        lower_received_row = []
        #send and receive for center nodes. We only send rows without edges since edge value is
        if mpi_world_size != 1:
            transaction_result = send_receive(first_row_in_cluster, last_row_in_cluster)
            upper_received_row = transaction_result[0]
            lower_received_row = transaction_result[1]

        center_done = False
        #Center update
        for i in range (first_row_in_cluster, last_row_in_cluster+1):
            if (i != 0) and (i != GRID_SIZE - 1):
                for j in range (1,GRID_SIZE-1):

                    left_neighbor = u1[i][j-1]
                    right_neighbor = u1[i][j+1]
                    lower_neighbor = 0.0
                    upper_neighbor = 0.0

                    if any_in_current_process(i+1):
                        lower_neighbor = u1[i+1][j]
                    else:
                        lower_neighbor = lower_received_row[j]

                    if any_in_current_process(i-1):
                        upper_neighbor = u1[i-1][j]
                    else:
                        upper_neighbor = upper_received_row[j]

                    #since neighbors are in lower and upper, center neighbors are above and below

                    u[i][j] = ((RHO* (upper_neighbor+ lower_neighbor + right_neighbor + left_neighbor - 4*u1[i][j]) + 2*u1[i][j] - (1-ETA)*u2[i][j])/(1+ETA))
                    u2[i][j] = u1[i][j]
                    u1[i][j] = u[i][j]

                    # if(i==GRID_SIZE/2 and j == GRID_SIZE/2):
                    #
                    #     print(str(u[GRID_SIZE / 2][GRID_SIZE / 2]))

            if i == last_row_in_cluster:
                center_done = True

        if center_done:
            #Edges update
            for i in range(first_row_in_cluster, last_row_in_cluster+1):
                for j in range(0,GRID_SIZE):

                    #upper edge  first
                    if (i==0) and (j!=0) and (j!=GRID_SIZE-1):
                        u[i][j] = G * (u[1][j])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]

                    #update lower edge
                    elif (i == GRID_SIZE-1 and j!=0 and j!= GRID_SIZE-1):
                        u[i][j] = G * (u[GRID_SIZE - 2][j])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]

                    #update left  edge
                    elif (j == 0 and i != 0 and i != GRID_SIZE-1):
                        u[i][j] = G * (u[i][1])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]

                    # update right edge
                    elif(j == GRID_SIZE-1 and i !=0 and i != GRID_SIZE-1):
                        u[i][j] = G *( u[i][GRID_SIZE - 2])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]

            #Corner update
            for i in range(first_row_in_cluster, last_row_in_cluster+1):
                for j in range(0,GRID_SIZE):
                    if i == 0 and j == 0:
                        u[i][j] = G * (u[0][1])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]

                    elif i == GRID_SIZE-1 and  j == 0:
                        u[i][j] = G * (u[GRID_SIZE-2][0])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]

                    elif i == 0 and j == GRID_SIZE-1:
                        u[i][j] = G * (u[0][GRID_SIZE-2])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]

                    elif i == GRID_SIZE-1 and j == GRID_SIZE-1:
                        u[i][j] = G * (u[GRID_SIZE-2][GRID_SIZE-1])
                        u2[i][j] = u1[i][j]
                        u1[i][j] = u[i][j]




            #Do error calculations
            comparable_output = output.get_output()
            do_error_calc(first_row_in_cluster,last_row_in_cluster,cnt)

    MPI.Finalize()

