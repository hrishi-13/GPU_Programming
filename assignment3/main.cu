/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

 // all levels are calculated correctly

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;
#define BLOCK_SIZE 1024

ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

// initializing the vertexLevel array

__global__ void initializingVertexLevel(unsigned int* d_aid, int *d_apr, int *d_vertexLevel, unsigned int* d_activeVertex, int V, int L){

    __shared__ int s_vertexLevel[BLOCK_SIZE];

    int id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(id >= 0 and id < V) {

        if(d_apr[id] > 0){
            s_vertexLevel[threadIdx.x] = L;
        }

        else if (d_apr[id] == 0){
            s_vertexLevel[threadIdx.x] = 0;
        }

        __syncthreads();
        d_vertexLevel[id] = s_vertexLevel[threadIdx.x];
    }
}

// finding which vertices are active and calculating the levels of each vertex

__global__ void calculatingVertexLevels(unsigned int *d_aid, int *d_apr, int *d_csrList, int *d_offset, int *d_vertexLevel, int V, int currentLevel) {

    int id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(id >= 0 and id < V){

        if(d_vertexLevel[id] == currentLevel){
            int start_index = d_offset[id];
            int end_index = d_offset[id+1];

            int destinationLevel = currentLevel + 1;

            for(int i = start_index; i < end_index; i++) {
                int destination = d_csrList[i];
                d_vertexLevel[destination] = destinationLevel;
            } 

            // checking deactivation condition
            if(id >= 1 and id <= V-2){
                int prev = id - 1;
                int next = id + 1;
                if(d_aid[prev] < d_apr[prev] and d_aid[next] < d_apr[next] and d_vertexLevel[prev] == d_vertexLevel[next] ) {
                    d_aid[id] = 0;
                }
            }

            for(int i = start_index; i < end_index; i++) {
                // checking activation condition
                if(d_aid[id] >= d_apr[id]) {
                    int destination = d_csrList[i];
                    // increasing aid of vertex by 1
                    atomicAdd(&d_aid[destination], 1);
                }
            }
        }
    }
}

// counting no. of active vertices at each level

__global__ void countingActiveVertices(unsigned int *d_aid, int *d_apr, int *d_vertexLevel, unsigned int *d_activeVertex, int V, int noOfLevel) {

    unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(id >=0 and id < V){
        // active vertex condition
        if((d_aid[id] >= (unsigned) d_apr[id])){ 
            atomicInc(&d_activeVertex[d_vertexLevel[id]], V+1); 
        }
    }
}
    
/**************************************END*************************************************/

//Function to write result in output file
void printResult( int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    unsigned int *d_aid; // acive in-degree array

    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1) * sizeof(int));
    cudaMalloc(&d_csrList, E * sizeof(int)); 
    cudaMalloc(&d_apr, V * sizeof(int)); 
    cudaMalloc(&d_aid, V * sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V * sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	// cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

// h_activeVertex already initialized
unsigned int *d_activeVertex;
cudaMalloc(&d_activeVertex, L * sizeof(int));

// initializing active vertex array with 0
cudaMemset(d_activeVertex, 0, L * sizeof(int));

int *d_vertexLevel; // contains levels for each vertex
cudaMalloc(&d_vertexLevel, V * sizeof(int));

// initializing active in degree array with 0
cudaMemset(d_aid, 0, V * sizeof(int));  

dim3 dimGrid1((V / BLOCK_SIZE) + 1, 1, 1); 
dim3 dimBlock1(BLOCK_SIZE, 1, 1);

// launching the kerneL: initializing the vertex level array
initializingVertexLevel<<<dimGrid1, dimBlock1>>>(d_aid, d_apr, d_vertexLevel, d_activeVertex, V, L);
cudaDeviceSynchronize();

dim3 dimGrid2((V / BLOCK_SIZE ) + 1, 1, 1); 
dim3 dimBlock2(BLOCK_SIZE, 1, 1);

for(int currentLevel = 0; currentLevel < L; currentLevel++) {
    // launching the kerneL: calculating the vertex levels
    calculatingVertexLevels<<<dimGrid2, dimBlock2>>>(d_aid, d_apr, d_csrList, d_offset, d_vertexLevel, V, currentLevel);
}
cudaDeviceSynchronize();

// launching the kernel: counting the number of active vertices at each level
countingActiveVertices<<<dimGrid2, dimBlock2>>>(d_aid, d_apr, d_vertexLevel, d_activeVertex, V, L);
cudaDeviceSynchronize();

cudaMemcpy(h_activeVertex, d_activeVertex, L * sizeof(int), cudaMemcpyDeviceToHost);

/********************************END OF CODE AREA**********************************/

double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);

if(argc > 2){
    for(int i=0; i<L; i++){
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}