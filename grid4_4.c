#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <math.h>

#define SEND_TAG 0
#define ANSWER_TAG 1
#define COMMUNICATE_TAG 2
#define GRID_SIZE 4
#define UPDATE_INTERIOR 0
#define UPDATE_BOUNDARY 1
#define UPDATE_CORNER 2
#define P 0.5
#define N 0.0002
#define G 0.75

typedef struct {
	int rank;
    float value;
	float prev_value;
	float prev_prev_value;
} gridNode;

//Current Grid
gridNode* grid[GRID_SIZE][GRID_SIZE];
//[0]=top [1]=right [2]=bottom [3]=left
float neighbor_nodes_values[4];
int numberOfSlaves;
//If printComment==1, then print some debug comments
int printComments = -1;


// Each node in the 4x4 grid is an independent process
void setupGrid(int rank){
	//Allocate memory to the global array of gridNode pointers (16 pointers)
	//grid = malloc(sizeof(gridNode *) * GRID_SIZE * GRID_SIZE);
	int row;
	int column;
	for (row=0;row<GRID_SIZE;row++){
		for(column=0;column<GRID_SIZE;column++){
			// Allocate memory to each gridNode pointer that will contain the info relative to a node
			grid[row][column] = malloc(sizeof(gridNode));
			
			//Fill gridNodes
			if(row == GRID_SIZE/2 && column == GRID_SIZE/2){
				//Center node (Set prev_value to 1.0 to simulate a hit on the drum)
				grid[row][column]->rank = rank;
				grid[row][column]->value = 1.0;
				grid[row][column]->prev_value = 0.0;
				grid[row][column]->prev_prev_value = 0.0;
			}else{
				grid[row][column]->rank = rank;
				grid[row][column]->value = 0.0;
				grid[row][column]->prev_value = 0.0;
				grid[row][column]->prev_prev_value = 0.0;
			}
		}
	}
}

int getRowFromRank(int rank) {
	int row = floor(rank / GRID_SIZE);
	return row;
}

int getColumnFromRank(int rank) {
	int column = rank % GRID_SIZE;
	return column;
}

int getRankFromRowColumn(int row, int column){
	if(row<0 || row>GRID_SIZE-1 || column<0 || column>GRID_SIZE-1){
		//Wrong coordinates
		return -1;
	}else{
		return row*GRID_SIZE+column;
	}
}

//Return 1 if coordinates point to corner node, 0 otherwise
int isCornerNode(int rank){
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	if((row==0 && column==0) || (row==0 && column==GRID_SIZE-1) || (row==GRID_SIZE-1 && column==0) || (row==GRID_SIZE-1 && column==GRID_SIZE-1)){
		return 1;
	}else{
		return 0;
	}
}

//Return 1 if coordinates point to boundary node, 0 otherwise
int isBoundaryNode(int rank){
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	//Boundary, but not corner
	if((row==0 || row==GRID_SIZE-1 || column==0 || column==GRID_SIZE-1) && (isCornerNode(rank))!=1){
		return 1;
	}else{
		return 0;
	}
}

//Return 1 if coordinates point to interior node, 0 otherwise
int isInteriorNode(int rank){
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	if((row>0 && row<GRID_SIZE-1) && (column>0 && column<GRID_SIZE-1)){
		return 1;
	}else{
		return 0;
	}
}

//Position: [0]=top [1]=right [2]=bottom [3]=left
int getRankNeighborNode(int myRow, int myColumn, int position){
	if(position==0){
		//Top
		return getRankFromRowColumn(myRow-1, myColumn);
	}else if(position==1){
		//Right
		return getRankFromRowColumn(myRow, myColumn+1);
	}else if(position==2){
		//Bottom
		return getRankFromRowColumn(myRow+1, myColumn);
	}else if(position==3){
		//Left
		return getRankFromRowColumn(myRow, myColumn-1);
	}else {
		//Wrong position value, no valid neighbor
		return -1;
	}
}

void performDataExchangesForInteriorHelper(int rank, int operation){
	//Get previous values from all boundary nodes
	//getRankNeighborNode [0]=top [1]=right [2]=bottom [3]=left
	int k;
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	if(operation==SEND_TAG){
		if(isBoundaryNode(rank)){
			for(k=0;k<4;k++){
				int neighborRankToSendTo = getRankNeighborNode(row, column, k);
				if(isInteriorNode(neighborRankToSendTo)==1){
					//Boundary node sends its previous value to a neighbor interior node
					MPI_Send(&(grid[row][column]->prev_value), 1, MPI_FLOAT, neighborRankToSendTo, COMMUNICATE_TAG, MPI_COMM_WORLD);
					//printf("I am rank %d and sent value %f to neigbor rank %d.\n", rank, grid[row][column]->prev_value, neighborRankToSendTo);
				}
			}
		}else if(isInteriorNode(rank)==1){
			for(k=0;k<4;k++){
				int neighborRankToSendTo = getRankNeighborNode(row, column, k);
				if(isInteriorNode(neighborRankToSendTo)==1){
					//Interior node send its previous value to another interior node
					MPI_Send(&(grid[row][column]->prev_value), 1, MPI_FLOAT, neighborRankToSendTo, COMMUNICATE_TAG, MPI_COMM_WORLD);
					//printf("I am rank %d and sent value %f to neigbor rank %d.\n", rank, grid[row][column]->prev_value, neighborRankToSendTo);
				}
			}
		}
	}else if(operation==ANSWER_TAG && isInteriorNode(rank)==1){
		//Interior nodes receive values from all of their neighbors
		for(k=0;k<4;k++){
			int neighborRankToReceiveFrom = getRankNeighborNode(row, column, k);
			MPI_Recv(&(neighbor_nodes_values[k]), 1, MPI_FLOAT, neighborRankToReceiveFrom, COMMUNICATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//printf("I am rank %d and received value %f from neigbor rank %d.\n", rank, neighbor_nodes_values[k], neighborRankToReceiveFrom);
		}
	}
}

void performDataExchangesForInterior(int rank){
	performDataExchangesForInteriorHelper(rank, SEND_TAG);
	MPI_Barrier(MPI_COMM_WORLD);
	performDataExchangesForInteriorHelper(rank, ANSWER_TAG);
}

void performDataExchangesForBoundaryHelper(int rank, int operation){
	//Send to boundary nodes needed values from neighbor interior nodes
	//getRankNeighborNode [0]=top [1]=right [2]=bottom [3]=left
	int k;
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	if(operation==SEND_TAG && isInteriorNode(rank)==1){
		for(k=0;k<4;k++){
			int neighborRankToSendTo = getRankNeighborNode(row, column, k);
			if(isBoundaryNode(neighborRankToSendTo)==1){
				//Interior node sends its current value to a neighbor boundary node
				MPI_Send(&(grid[row][column]->value), 1, MPI_FLOAT, neighborRankToSendTo, COMMUNICATE_TAG, MPI_COMM_WORLD);
				//printf("I am rank %d and sent value %f to neigbor rank %d.\n", rank, grid[row][column]->value, neighborRankToSendTo);
			}
		}
	}else if(operation==ANSWER_TAG && isBoundaryNode(rank)==1){
		//Boundary nodes receive values from neighbor interior nodes
		for(k=0;k<4;k++){
			int neighborRankToReceiveFrom = getRankNeighborNode(row, column, k);
			if(isInteriorNode(neighborRankToReceiveFrom)==1){
				MPI_Recv(&(neighbor_nodes_values[k]), 1, MPI_FLOAT, neighborRankToReceiveFrom, COMMUNICATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("I am rank %d and received value %f from neigbor rank %d.\n", rank, neighbor_nodes_values[k], neighborRankToReceiveFrom);
			}
		}
	}
}

void performDataExchangesForBoundary(int rank){
	performDataExchangesForBoundaryHelper(rank, SEND_TAG);
	MPI_Barrier(MPI_COMM_WORLD);
	performDataExchangesForBoundaryHelper(rank, ANSWER_TAG);
}

void performDataExchangesForCornersHelper(int rank, int operation){
	//Send to boundary nodes needed values from neighbor interior nodes
	//getRankNeighborNode [0]=top [1]=right [2]=bottom [3]=left
	int k;
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	if(operation==SEND_TAG && isBoundaryNode(rank)==1){
		//Specific boundary nodes send their current value to neighbor corner nodes
		int neighborRankToSendTo=-1;
		if(row==1 && column==0){
			//Send my current value to [0,0] corner
			neighborRankToSendTo = getRankFromRowColumn(0,0);
		}else if(row==GRID_SIZE-2 && column==0){
			//Send my current value to [GRID_SIZE-1,0] corner
			neighborRankToSendTo = getRankFromRowColumn(GRID_SIZE-1,0);
		}else if(row==0 && column==GRID_SIZE-2){
			//Send my current value to [0,GRID_SIZE-1] corner
			neighborRankToSendTo = getRankFromRowColumn(0,GRID_SIZE-1);
		}else if(row==GRID_SIZE-1 && column==GRID_SIZE-2){
			//Send my current value to [GRID_SIZE-1,GRID_SIZE-1] corner
			neighborRankToSendTo = getRankFromRowColumn(GRID_SIZE-1,GRID_SIZE-1);
		}
		if(neighborRankToSendTo!=-1){
			MPI_Send(&(grid[row][column]->value), 1, MPI_FLOAT, neighborRankToSendTo, COMMUNICATE_TAG, MPI_COMM_WORLD);
			//printf("I am rank %d and sent value %f to neigbor rank %d.\n", rank, grid[row][column]->value, neighborRankToSendTo);
		}
	}else if(operation==ANSWER_TAG && isCornerNode(rank)==1){
		//Corner nodes receive values from specific neighbor boundary nodes
		int neighborRankToReceiveFrom=-1;
		if(row==0 && column==0){
			//Receive current value from bottom [1,0] boundary node
			neighborRankToReceiveFrom = getRankFromRowColumn(1,0);
			k=2;
		}else if(row==GRID_SIZE-1 && column==0){
			//Receive current value from top [GRID_SIZE-2,0] boundary node
			neighborRankToReceiveFrom = getRankFromRowColumn(GRID_SIZE-2,0);
			k=0;
		}else if(row==0 && column==GRID_SIZE-1){
			//Receive current value from left [0,GRID_SIZE-2] boundary node
			neighborRankToReceiveFrom = getRankFromRowColumn(0,GRID_SIZE-2);
			k=3;
		}else if(row==GRID_SIZE-1 && column==GRID_SIZE-1){
			//Receive current value from left [GRID_SIZE-1,GRID_SIZE-2] boundary node
			neighborRankToReceiveFrom = getRankFromRowColumn(GRID_SIZE-1,GRID_SIZE-2);
			k=3;
		}
		if(neighborRankToReceiveFrom!=-1){
			MPI_Recv(&(neighbor_nodes_values[k]), 1, MPI_FLOAT, neighborRankToReceiveFrom, COMMUNICATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//printf("I am rank %d and received value %f from neigbor rank %d.\n", rank, neighbor_nodes_values[k], neighborRankToReceiveFrom);
		}
	}
}

void performDataExchangesForCorners(int rank){
	performDataExchangesForCornersHelper(rank, SEND_TAG);
	MPI_Barrier(MPI_COMM_WORLD);
	performDataExchangesForCornersHelper(rank, ANSWER_TAG);
}

void setNewValue(float newValue, int row, int column){
	grid[row][column]->prev_prev_value = grid[row][column]->prev_value;
	grid[row][column]->prev_value = grid[row][column]->value;
	grid[row][column]->value = newValue;
}

void perform_iterationHelper(int rank, int operation){
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	
	//Update Interior Nodes
	if(operation == UPDATE_INTERIOR){
		performDataExchangesForInterior(rank);
		if(isInteriorNode(rank)==1){
			float newValue = (P*(neighbor_nodes_values[0]+neighbor_nodes_values[1]+neighbor_nodes_values[2]+neighbor_nodes_values[3]-4*grid[row][column]->prev_value)+2*grid[row][column]->prev_value-(1-N)*grid[row][column]->prev_prev_value)/(1+N);
			setNewValue(newValue, row, column);
		}
	}
	
	//Update Boundary Nodes
	if(operation == UPDATE_BOUNDARY){
		performDataExchangesForBoundary(rank);
		if(isBoundaryNode(rank)==1){
			//[0]=top [1]=right [2]=bottom [3]=left
			float newValue;
			if(row==0){
				newValue = G*neighbor_nodes_values[2];
			}else if(row==GRID_SIZE-1){
				newValue = G*neighbor_nodes_values[0];
			}else if(column==0){
				newValue = G*neighbor_nodes_values[1];
			}else if(column==GRID_SIZE-1){
				newValue = G*neighbor_nodes_values[3];
			}
			setNewValue(newValue, row, column);
		}
	}
	
	//Update Corner Nodes
	if(operation == UPDATE_CORNER){
		performDataExchangesForCorners(rank);
		if(isCornerNode(rank)==1){
			//[0]=top [1]=right [2]=bottom [3]=left
			float newValue;
			if(row==0 && column==0){
				newValue = G*neighbor_nodes_values[2];
			}else if(row==GRID_SIZE-1 && column==0){
				newValue = G*neighbor_nodes_values[0];
			}else if(row==0 && column==GRID_SIZE-1){
				newValue = G*neighbor_nodes_values[3];
			}else if(row==GRID_SIZE-1 && column==GRID_SIZE-1){
				newValue = G*neighbor_nodes_values[3];
			}
			setNewValue(newValue, row, column);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void shiftValues(int rank){
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	grid[row][column]->prev_prev_value = grid[row][column]->prev_value;
	grid[row][column]->prev_value = grid[row][column]->value;
	MPI_Barrier(MPI_COMM_WORLD);
}

void perform_iteration(int rank){
	shiftValues(rank);
	perform_iterationHelper(rank, UPDATE_INTERIOR);
	perform_iterationHelper(rank, UPDATE_BOUNDARY);
	perform_iterationHelper(rank, UPDATE_CORNER);
}

void printMiddleValue(int iteration, int rank){
	//Print value at [GRID_SIZE/2,GRID_SIZE/2]
	if(rank==getRankFromRowColumn(GRID_SIZE/2, GRID_SIZE/2)){
		printf("Iteration %d: %f\n", iteration, grid[GRID_SIZE/2][GRID_SIZE/2]->value);
	}
}

//If includeRank=1, also include rank values
//If includePrevValues=1, also include previous values
void printGrid(int rank, int includeRank, int includePrevValues){
	MPI_Barrier(MPI_COMM_WORLD);
	//Send grid Elements to master
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	char textOutputBuffer[256];
	
	if(includeRank==1 && includePrevValues==1){
		//Print (x,y) coordinates, rank, present value and previous values
		sprintf( textOutputBuffer, "(%d,%d): [%d,%f,%f,%f]", row, column, grid[row][column]->rank, grid[row][column]->value, grid[row][column]->prev_value, grid[row][column]->prev_prev_value);
	}else if(includeRank==1){
		//Print (x,y) coordinates, rank and present value
		sprintf( textOutputBuffer, "(%d,%d): [%d,%f]", row, column, grid[row][column]->rank, grid[row][column]->value);
	}else if(includePrevValues==1){
		//Print (x,y) coordinates, present value and previous values
		sprintf( textOutputBuffer, "(%d,%d): [%f,%f,%f]", row, column, grid[row][column]->value, grid[row][column]->prev_value, grid[row][column]->prev_prev_value);
	}else {
		//Print only (x,y) coordinates and present value
		sprintf( textOutputBuffer, "(%d,%d): %f", row, column, grid[row][column]->value);
	}
	MPI_Send(&textOutputBuffer, 256, MPI_CHAR, 0, ANSWER_TAG, MPI_COMM_WORLD);
	
	if(rank == 0){
		//Print the grid from the master by receiving results from all processes
		printf("\n\n");
		int i;
		for (i = 0; i <= numberOfSlaves; i++) {	
			//Receive one message from the master and one message from each slave
			char textOutputBuffer[256];
			MPI_Recv(&textOutputBuffer, 256, MPI_CHAR, i, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("%s   ", textOutputBuffer);
			if(getColumnFromRank(i) == GRID_SIZE-1){
				//New line
				printf("\n");
			}
		}
		printf("\n\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void printProcessInfo(int rank){
	MPI_Barrier(MPI_COMM_WORLD);
	//Send process info to master
	int row = getRowFromRank(rank);
	int column = getColumnFromRank(rank);
	char textOutputBuffer[256];
	sprintf( textOutputBuffer, "I am process rank %d with:  value = %f    prev_value = %f    prev_prev_value = %f\n", grid[row][column]->rank, grid[row][column]->value, grid[row][column]->prev_value, grid[row][column]->prev_prev_value);
	MPI_Send(&textOutputBuffer, 256, MPI_CHAR, 0, ANSWER_TAG, MPI_COMM_WORLD);
	
	if(rank == 0){
		//Print the process info from the master by receiving results from all processes
		printf("\n\n");
		int i;
		for (i = 0; i <= numberOfSlaves; i++) {	
			//Receive one message from the master and one message from each slave
			char textOutputBuffer[256];
			MPI_Recv(&textOutputBuffer, 256, MPI_CHAR, i, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("%s\n", textOutputBuffer);
		}  
		printf("\n\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
}



int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int numberOfProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

	//Test if at least one process provided
	if (numberOfProcesses < 1 )
	{
	  printf("ERROR! Need at least one process!\n");
	  MPI_Abort(MPI_COMM_WORLD, 1);
	  exit(1);
	}
	
	numberOfSlaves = numberOfProcesses - 1;

    // Get the rank of the process
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	setupGrid(rank);
	
	//If includeRank=1, also include rank values
	//If includePrevValues=1, also include previous values
	printGrid(rank,0,0);
	//printProcessInfo(rank);
	
	int iterations = atoi(argv[1]);
	int i;
	for(i=1;i<=iterations;i++){
		perform_iteration(rank);
		printGrid(rank,0,0);
		printMiddleValue(i, rank);
		//printProcessInfo(rank);
	}

    // Finalize the MPI environment.
    MPI_Finalize();
}