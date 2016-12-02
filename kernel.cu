/*
	Sample input file format:

	1.Line : 6 => Number of nodes(int)
	2.Line : 7 => Number of edges(int)
	3.Line : 1 2 5.0 ----------------
	4.Line : 2 3 1.5                |
	5.Line : 1 3 2.1				|
	6.Line : 1 4 1.2				|=> Edges
	7.Line : 1 5 15.5				|
	8.Line : 2 5 3.6				|
	9.Line : 3 6 1.2-----------------
	10.Line : 1 => Start node.
	///////////////////////////////////////////////////////

	Doesn't check any error condition.
*/

#include <iostream>
#include <fstream>
#include <limits>
#include <cstddef>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <device_atomic_functions.h>

using namespace std;

// Edge struct.
typedef struct {
	int* startPoints;
	int* endPoints;
	double* weights;
}Edge;

__global__ void updateQueueKernel(int *queueu,int *queueSize, const int *startPoints, const int *endPoints,const int*visitedArray,
	const int *currentVertex ) {

	int index = threadIdx.x;
	if (startPoints[index] == *currentVertex && visitedArray[endPoints[index]] == 0 )
	{
		int oldValue = atomicAdd(queueSize,1);
		queueu[oldValue] = index;
	}
}

// This kernel will call queue size thread.
__global__ void processQueueKernel(int *parentArray, double *resultWeightArray,
	const int* queue,const int *startPoints,const int *endPoints, const double *weightArray) {

	int threadIndex = threadIdx.x;
	int elementIndex = queue[threadIndex];
	int startNode = startPoints[elementIndex];
	int endNode = endPoints[elementIndex];
	double edgeWeight = weightArray[elementIndex];
	double nodeWeight = resultWeightArray[startNode];

	if (nodeWeight + edgeWeight < resultWeightArray[endNode])
	{
		resultWeightArray[endNode] = nodeWeight + edgeWeight;
		parentArray[endNode] = startNode;
	}
}

int main()
{
	// Input file.
	string input = "input.txt";
	fstream file(input);
	cerr << "# Input file      : " << input.c_str() << endl << endl;
	///// Read node number and edges number from file
	int numberOfNodes,numberOfEdges;
	file >> numberOfNodes >> numberOfEdges;
	++numberOfNodes;
	
	cerr << "# Number of node  : " << (numberOfNodes-1) << endl;
	cerr << "# Number of edges : " << numberOfEdges << endl << endl;

	// Allocate memory for edges.
	Edge edges;
	edges.startPoints = new int[numberOfEdges];
	edges.endPoints = new int[numberOfEdges];
	edges.weights = new double[numberOfEdges];

	// Read all edges from file.
	int tempStart, tempEnd;
	double tempWeight;
	for (int i = 0; i < numberOfEdges; i++)
	{
		file >> tempStart >> tempEnd >> tempWeight;
		edges.startPoints[i] = tempStart;
		edges.endPoints[i] = tempEnd;
		edges.weights[i] = tempWeight;
	}

	// Print all edges to screen
	cerr << endl << "### Edges ###" << endl;
	cerr << "-------------" << endl << endl;
	for (int i = 0; i < numberOfEdges; i++)
	{
		cerr <<  "# ["<< (i+1) << "]  #From : "<<edges.startPoints[i] << "  #To : " << edges.endPoints[i] << "  #Cost:  " << edges.weights[i] << endl;
	}
	cerr << endl << endl;

	// Create parent array to hold each parent.
	int *parentArray = new int[numberOfNodes];

	// Create visited array to hold "Did i visit this node ? "
	int *visitedArray = new int[numberOfNodes];

	// Crate costs of array.
	double *resultWeightArray = new double[numberOfNodes];

	// Queue to hold nearest nodes.
	int *queue = new int[numberOfNodes];

	// Queue size variable.
	int *queueSize = new int;
	*queueSize = 1;

	// initialize arrays.
	for (int i = 0; i < numberOfNodes; i++)
	{
		queue[i] = -1;
		parentArray[i] = -1;
		resultWeightArray[i] = numeric_limits<double>::max() ;
		visitedArray[i] = 0;
	}

	// Get start vertex from file.
	int startVertex;
	file >> startVertex;
	visitedArray[startVertex] = 1;
	resultWeightArray[startVertex] = 0;
	cerr << "# Start Node      : " << startVertex << endl << endl;
	cudaError_t cudaStatus;
	// Create device arrays.

	int *deviceQueue = 0;
	cudaStatus = cudaMalloc((void**)&deviceQueue, (numberOfNodes) * sizeof(int));
	cudaStatus = cudaMemcpy(deviceQueue, queue, sizeof(int) * numberOfNodes, cudaMemcpyHostToDevice);

	int *deviceQueueSize = 0;
	cudaStatus = cudaMalloc((void**)&deviceQueueSize, sizeof(int));
	cudaStatus = cudaMemcpy(deviceQueueSize, queueSize, sizeof(int), cudaMemcpyHostToDevice);

	int *deviceStartPoints = 0;
	cudaStatus = cudaMalloc((void**)&deviceStartPoints, sizeof(int) * numberOfEdges);
	cudaStatus = cudaMemcpy(deviceStartPoints,edges.startPoints,sizeof(int) * numberOfEdges,cudaMemcpyHostToDevice);

	int *deviceEndPoints = 0;
	cudaStatus = cudaMalloc((void**)&deviceEndPoints, sizeof(int) * numberOfEdges);
	cudaStatus = cudaMemcpy(deviceEndPoints, edges.endPoints, sizeof(int) * numberOfEdges, cudaMemcpyHostToDevice);

	double *deviceWeights = 0;
	cudaStatus = cudaMalloc((void**)&deviceWeights, sizeof(double) * numberOfEdges);
	cudaStatus = cudaMemcpy(deviceWeights, edges.weights, sizeof(double) * numberOfEdges, cudaMemcpyHostToDevice);

	int *deviceCurrentVertex = 0;
	cudaStatus = cudaMalloc((void**)&deviceCurrentVertex, sizeof(int));

	int *deviceParentArray = 0;
	cudaStatus = cudaMalloc((void**)&deviceParentArray, sizeof(int) * numberOfNodes);
	cudaStatus = cudaMemcpy(deviceParentArray, parentArray, sizeof(int) * numberOfNodes, cudaMemcpyHostToDevice);

	double *deviceResultWeights = 0;
	cudaStatus = cudaMalloc((void**)&deviceResultWeights, sizeof(double) * numberOfNodes);
	cudaStatus = cudaMemcpy(deviceResultWeights, resultWeightArray, sizeof(double) * numberOfNodes, cudaMemcpyHostToDevice);

	int *deviceVisitedArray = 0;
	cudaStatus = cudaMalloc((void**)&deviceVisitedArray, sizeof(int) * numberOfNodes);
	cudaStatus = cudaMemcpy(deviceVisitedArray,visitedArray, sizeof(int) * numberOfNodes, cudaMemcpyHostToDevice);

	int currentVertex = startVertex;
	// Start bfs
	while ((*queueSize) != 0 && currentVertex != -1) {
		
		--(*queueSize);
		cudaStatus = cudaMemcpy(deviceQueueSize, queueSize, sizeof(int), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(deviceCurrentVertex, &currentVertex, sizeof(int), cudaMemcpyHostToDevice);

		updateQueueKernel << <1, numberOfEdges >> > (deviceQueue, deviceQueueSize, deviceStartPoints, deviceEndPoints, deviceVisitedArray, deviceCurrentVertex);

		cudaStatus = cudaMemcpy(queueSize, deviceQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
		
		processQueueKernel << < 1, (*queueSize) >> > (deviceParentArray, deviceResultWeights, deviceQueue, deviceStartPoints, deviceEndPoints, deviceWeights);
		
		cudaStatus = cudaMemcpy(queue, deviceQueue, sizeof(int)*numberOfNodes, cudaMemcpyDeviceToHost);
		currentVertex = edges.endPoints[queue[0]];
		for (int i = 1; i < numberOfNodes; ++i)
			queue[i - 1] = queue[i];

		cudaStatus = cudaMemcpy(deviceQueueSize, queueSize, sizeof(int), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(deviceQueue, queue, sizeof(int) * numberOfNodes, cudaMemcpyHostToDevice);
		
	}

	cerr << endl << "#### Result ####" << endl;
	cerr << "----------------" << endl << endl;
	cudaStatus = cudaMemcpy(parentArray, deviceParentArray, numberOfNodes * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(resultWeightArray, deviceResultWeights,numberOfNodes * sizeof(double), cudaMemcpyDeviceToHost);
	
	for (int i = 1; i < numberOfNodes; i++)
	{
		cerr << "# Node : " <<  i << "     # Parent Node :     " << parentArray[i] << "   #Cost :  "<< resultWeightArray[i] << endl;
	}
	cudaFree(deviceVisitedArray);
	cudaFree(deviceParentArray);
	cudaFree(deviceWeights);
	cudaFree(deviceQueue);
	cudaFree(deviceQueueSize);
	cudaFree(deviceStartPoints);
	cudaFree(deviceEndPoints);
	cudaFree(deviceCurrentVertex);
	free(queueSize);
	free(queue);
	free(visitedArray);
	free(parentArray);
	getchar();
    return 0;
}
