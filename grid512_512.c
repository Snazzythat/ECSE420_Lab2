#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <math.h>
#include "output.h"

#define SEND_TAG 0
#define ANSWER_TAG 1
#define COMMUNICATE_TAG 2
#define GRID_SIZE 512
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
float top_neighbor_interior_nodes_values[GRID_SIZE-2];
float bottom_neighbor_interior_nodes_values[GRID_SIZE-2];
float neighbor_nodes_values[4];
int numberOfSlaves;
int numberOfRowsToProcess;
int offsetRows;



void setupGrid(){
	int row;
	int column;
	for (row=0;row<GRID_SIZE;row++){
		for(column=0;column<GRID_SIZE;column++){
			// Allocate memory to each gridNode pointer that will contain the info relative to a node
			grid[row][column] = malloc(sizeof(gridNode));
			
			//Fill gridNodes
			if(row == GRID_SIZE/2 && column == GRID_SIZE/2){
				//Center node (Set value to 1.0 to simulate a hit on the drum, It will be shifted to prev_value before the iteration)
				grid[row][column]->value = 1.0;
				grid[row][column]->prev_value = 0.0;
				grid[row][column]->prev_prev_value = 0.0;
			}else{
				grid[row][column]->value = 0.0;
				grid[row][column]->prev_value = 0.0;
				grid[row][column]->prev_prev_value = 0.0;
			}
		}
	}
}

//Return 1 if coordinates point to interior node, 0 otherwise
int isInteriorNode(int row, int column){
	if((row>0 && row<GRID_SIZE-1) && (column>0 && column<GRID_SIZE-1)){
		return 1;
	}else{
		return 0;
	}
}

void splitRows(int rank, int numberOfProcesses){
	if(rank==0){
		//Split rows and send them to all workers
		int rowsForEachProcess = floor(GRID_SIZE/numberOfProcesses);
		int offset=0;
		int i;
		for(i=0;i<numberOfProcesses;i++){
				//Assign usual amount of rows to process rank i
				MPI_Send(&rowsForEachProcess, 1, MPI_INT, i, COMMUNICATE_TAG, MPI_COMM_WORLD);
				//Send offset
				offset = rowsForEachProcess*i;
				MPI_Send(&offset, 1, MPI_INT, i, COMMUNICATE_TAG, MPI_COMM_WORLD);
		}
	}
	
	//All ranks receive number of rows to process and offset from rank 0
	MPI_Recv(&numberOfRowsToProcess, 1, MPI_INT, 0, COMMUNICATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&offsetRows, 1, MPI_INT, 0, COMMUNICATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int shouldTopRow(int rank){
	if(rank==0){
		return -1;
	}else{
		return 1;
	}
}

int shouldBottomRow(int rank, int numberOfProcesses){
	if(numberOfProcesses<2){
		return -1;
	}else{
		if(rank==numberOfProcesses-1){
			return -1;
		}else{
			return 1;
		}
	}
}

int getTopRowIndex(){
	return offsetRows;
}

int getBottomRowIndex(){
	return offsetRows+numberOfRowsToProcess-1;
}

int getStartIndex(){
	return offsetRows*GRID_SIZE;
}

int getEndIndex(){
	int startIndex = getStartIndex();
	return startIndex + numberOfRowsToProcess*GRID_SIZE - 1;
}

int containsPoint(int row, int column){
	int startIndex = getStartIndex();
	int endIndex = getEndIndex();
	int searchedIndex = row*GRID_SIZE + column;
	if(startIndex<=searchedIndex && searchedIndex<=endIndex){
		return 1;
	}else {
		return -1;
	}
}

void performAllDataExchangesHelper(int rank, int numberOfProcesses, int operation){
	int row;
	int column;
	
	if(operation==SEND_TAG){
		if(shouldTopRow(rank)==1){
			//Send top row of interior nodes to rank-1
			int topRowIndex = getTopRowIndex();
			int neighborRankToSendTo = rank-1;
			int i;
			for(i=1;i<GRID_SIZE-1;i++){
				MPI_Send(&(grid[topRowIndex][i]->prev_value), 1, MPI_FLOAT, neighborRankToSendTo, COMMUNICATE_TAG, MPI_COMM_WORLD);
				//printf("I am process rank %d and sent value %f to neigbor rank %d for row %d and column %d.\n", rank, grid[topRowIndex][i]->prev_value, neighborRankToSendTo, topRowIndex, i);
			}
		}
		if(shouldBottomRow(rank, numberOfProcesses)==1){
			//Send bottom row of interior nodes to rank+1
			int bottomRowIndex = getBottomRowIndex();
			int neighborRankToSendTo = rank+1;
			int i;
			for(i=1;i<GRID_SIZE-1;i++){
				MPI_Send(&(grid[bottomRowIndex][i]->prev_value), 1, MPI_FLOAT, neighborRankToSendTo, COMMUNICATE_TAG, MPI_COMM_WORLD);
				//printf("I am process rank %d and sent value %f to neigbor rank %d for row %d and column %d.\n", rank, grid[bottomRowIndex][i]->prev_value, neighborRankToSendTo, bottomRowIndex, i);
			}
		}
	}else if(operation==ANSWER_TAG) {
		//top_neighbor_nodes_values
		if(shouldTopRow(rank)==1){
			//Receive top row of interior nodes from rank-1
			int i;
			int neighborRankToReceiveFrom = rank-1;
			for(i=0;i<GRID_SIZE-2;i++){
				MPI_Recv(&(top_neighbor_interior_nodes_values[i]), 1, MPI_FLOAT, neighborRankToReceiveFrom, COMMUNICATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("I am process rank %d and received value %f from neigbor rank %d for column %d.\n", rank, top_neighbor_interior_nodes_values[i], neighborRankToReceiveFrom, i+1);
			}
		}
		if(shouldBottomRow(rank, numberOfProcesses)==1){
			//Receive bottom row of interior nodes from rank+1
			int i;
			int neighborRankToReceiveFrom = rank+1;
			for(i=0;i<GRID_SIZE-2;i++){
				MPI_Recv(&(bottom_neighbor_interior_nodes_values[i]), 1, MPI_FLOAT, neighborRankToReceiveFrom, COMMUNICATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("I am process rank %d and received value %f from neigbor rank %d for column %d.\n", rank, bottom_neighbor_interior_nodes_values[i], neighborRankToReceiveFrom, i+1);
			}
		}
	}
}

void performAllDataExchanges(int rank, int numberOfProcesses){
	performAllDataExchangesHelper(rank, numberOfProcesses, SEND_TAG);
	MPI_Barrier(MPI_COMM_WORLD);
	performAllDataExchangesHelper(rank, numberOfProcesses, ANSWER_TAG);
}

void setNewValue(float newValue, int row, int column){
	grid[row][column]->prev_prev_value = grid[row][column]->prev_value;
	grid[row][column]->prev_value = grid[row][column]->value;
	grid[row][column]->value = newValue;
}

void setNeighborNodesValuesInterior(int row, int column){
	//[0]=top [1]=right [2]=bottom [3]=left
	int k;
	int topRowIndex = getTopRowIndex();
	int bottomRowIndex = getBottomRowIndex();
	if(row == topRowIndex){
		//Grab top neighbor from top_neighbor_interior_nodes_values. Grab bottom neighbor from grid
		neighbor_nodes_values[0] = top_neighbor_interior_nodes_values[column-1];
		neighbor_nodes_values[2] = grid[row+1][column]->prev_value;
	}else if(row == bottomRowIndex){
		//Grab bottom neighbor from bottom_neighbor_interior_nodes_values. Grab top neighbor from grid
		neighbor_nodes_values[2] = bottom_neighbor_interior_nodes_values[column-1];
		neighbor_nodes_values[0] = grid[row-1][column]->prev_value;
	}else{
		//Grab top and bottom neighbor from same process (grid)
		neighbor_nodes_values[0] = grid[row-1][column]->prev_value;
		neighbor_nodes_values[2] = grid[row+1][column]->prev_value;
	}
	//Left and right neighbor from same process (grid)
	neighbor_nodes_values[1] = grid[row][column+1]->prev_value;
	neighbor_nodes_values[3] = grid[row][column-1]->prev_value;
}

void perform_iterationHelper(int rank, int operation, int numberOfProcesses){
	//Update Interior Nodes
	if(operation == UPDATE_INTERIOR){
		performAllDataExchanges(rank,numberOfProcesses);
		int topRowIndex = getTopRowIndex();
		int bottomRowIndex = getBottomRowIndex();
		int row;
		int column;
		for(row=topRowIndex;row<=bottomRowIndex;row++){
			for(column=1;column<GRID_SIZE-1;column++){
				if(isInteriorNode(row,column)==1){
					setNeighborNodesValuesInterior(row, column);
					float newValue = (P*(neighbor_nodes_values[0]+neighbor_nodes_values[1]+neighbor_nodes_values[2]+neighbor_nodes_values[3]-4*grid[row][column]->prev_value)+2*grid[row][column]->prev_value-(1-N)*grid[row][column]->prev_prev_value)/(1+N);
					setNewValue(newValue, row, column);
				}
			}
		}
	}
	
	//Update Boundary Nodes
	if(operation == UPDATE_BOUNDARY){
		int topRowIndex = getTopRowIndex();
		int bottomRowIndex = getBottomRowIndex();
		int row;
		int column;

		for(row=topRowIndex;row<=bottomRowIndex;row++){
			if(row==0){
				//Top row of grid: All values inside the row are boundaries (except corners)
				for(column=1;column<GRID_SIZE-1;column++){
					float newValue = G*grid[row+1][column]->value;
					setNewValue(newValue, row, column);
				}
			}else if(row==GRID_SIZE-1){
				//Bottom row of grid: All values inside the row are boundaries (except corners)
				for(column=1;column<GRID_SIZE-1;column++){
					float newValue = G*grid[row-1][column]->value;
					setNewValue(newValue, row, column);
				}
			}else{
				//Not top row of grid (we got only two boundary values per row at column=0 and column=GRID_SIZE-1)
				float newValueLeft = G*grid[row][1]->value;
				setNewValue(newValueLeft, row, 0);
				float newValueRight = G*grid[row][GRID_SIZE-2]->value;
				setNewValue(newValueRight, row, GRID_SIZE-1);
			}
		}
	}
	
	//Update Corner Nodes
	if(operation == UPDATE_CORNER){
		int topRowIndex = getTopRowIndex();
		int bottomRowIndex = getBottomRowIndex();
		
		if(topRowIndex==0){
			//Top row of grid: update both corners
			float newValueLeft = G*grid[1][0]->value;
			setNewValue(newValueLeft, 0, 0);
			float newValueRight = G*grid[0][GRID_SIZE-2]->value;
			setNewValue(newValueRight, 0, GRID_SIZE-1);
		}
		if(bottomRowIndex==GRID_SIZE-1){
			//Bottom row of grid: update both corners
			float newValueLeft = G*grid[GRID_SIZE-2][0]->value;
			setNewValue(newValueLeft, GRID_SIZE-1, 0);
			float newValueRight = G*grid[GRID_SIZE-1][GRID_SIZE-2]->value;
			setNewValue(newValueRight, GRID_SIZE-1, GRID_SIZE-1);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}


void shiftValues(){
	int topRowIndex = getTopRowIndex();
	int bottomRowIndex = getBottomRowIndex();
	int row;
	int column;
	for(row=topRowIndex;row<=bottomRowIndex;row++){
		for(column=0;column<GRID_SIZE;column++){
			grid[row][column]->prev_prev_value = grid[row][column]->prev_value;
			grid[row][column]->prev_value = grid[row][column]->value;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void perform_iteration(int rank, int numberOfProcesses){
	shiftValues();
	perform_iterationHelper(rank, UPDATE_INTERIOR, numberOfProcesses);
	perform_iterationHelper(rank, UPDATE_BOUNDARY, numberOfProcesses);
	perform_iterationHelper(rank, UPDATE_CORNER, numberOfProcesses);
}

void printMiddleValue(int iteration, int rank){
	//Print value at [GRID_SIZE/2,GRID_SIZE/2]
	if(containsPoint(GRID_SIZE/2, GRID_SIZE/2)==1){
		float midValue = grid[GRID_SIZE/2][GRID_SIZE/2]->value;
		float error = (midValue - output[iteration])*(midValue - output[iteration]);
		if(error < 0.00001){
			//printf("%f\n", midValue);
		}else{
			//printf("0.000000\n");
		}
		printf("%f\n", midValue);
	}
}

void printProcessInfo(int rank){
	MPI_Barrier(MPI_COMM_WORLD);
	//Send process info to master
	char textOutputBuffer[256];
	int topRowIndex = getTopRowIndex();
	int bottomRowIndex = getBottomRowIndex();
	int startIndex = getStartIndex();
	int endIndex = getEndIndex();
	sprintf( textOutputBuffer, "I am process rank %d with:  Rows to process: %d   Offset: %d   Top Row Index: %d    Bottom Row Index: %d    Start Index: %d   End Index: %d \n", rank, numberOfRowsToProcess, offsetRows, topRowIndex, bottomRowIndex, startIndex, endIndex);
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
	
	setupGrid();
	
	//If includeRank=1, also include rank values
	//If includePrevValues=1, also include previous values
	//printGrid(rank,0,0);
	splitRows(rank,numberOfProcesses);
	//printProcessInfo(rank);
	
	int iterations = atoi(argv[1]);
	int i;
	for(i=0;i<iterations;i++){
		perform_iteration(rank, numberOfProcesses);
		//printGrid(rank,0,0);
		printMiddleValue(i, rank);
		//printProcessInfo(rank);
	}

    // Finalize the MPI environment.
    MPI_Finalize();
}