#include "mpi.h" // MPI library for parallel processing
#include <stdio.h> // Standard I/O library
#include <stdlib.h>  // For memory allocation and conversions
#include <math.h> // For math functions (floor)
#include <stdbool.h> // For boolean type (boolean array allocation)

#define TAG_ARRAY 0 // Tag for sending/receiving array slices
#define TAG_LAST 1 // Tag for sending/receiving the last slice
#define SLICER 50000 // Tag for sending/receiving the last slice

int main(int argc, char *argv[])
{   
    int myRank; // Current process rank (ID)
    int numProc; // Total number of processes

    double startTime = 0.0; // Variable to store start time
    double endTime = 0.0; // Variable to store end time  
  
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank); // Get process rank 
    MPI_Comm_size(MPI_COMM_WORLD, &numProc); // Get total number of processes

    // Check if the user provided the required argument (size of matrix n)
    if (argc < 2) {
        if (myRank == 0) { // Only root process prints the error message
            printf("Invalid arguments, Usage: ./new_code n (number)\n");
        }
        MPI_Finalize(); // Clean up MPI
        return 0; // Exit program
    }
  
    startTime = MPI_Wtime(); // Starting the timer to benchmark execution time
  
    // converting the argument into long integer
    long n = atol(argv[1]); 
    
    //max product value can be the square of the matrix size
    unsigned long maxMatrix = n * n; 
    // Count of unique products in the upper triangular (including diagonal):
    // symmetric matrix of size n x n has (n * (n + 1)) / 2 unique products so we are only interested in the upper triangular part.
    unsigned long upperTriangularPart = ( maxMatrix + n ) / 2; 

    
    //an array to store the work assigned to every process
    unsigned long workPerProcess[numProc];
    
    // load balancing (static): distribute the work among processes
    for (int proc = 0; proc < numProc; proc++){
        workPerProcess[proc] = floor((float) upperTriangularPart / numProc);  // Divide work evenly


        if (proc < upperTriangularPart % numProc) { //distributing the remainder work 
            workPerProcess[proc] += 1;
        } 
    }

 
    MPI_Barrier(MPI_COMM_WORLD); // Synchronizing all processes before starting computation

    // Determining the starting offset for each process by summing work assigned to earlier processes.
    unsigned long startOffset = 0;  
    unsigned long localCount = 1;
    for (int proc = 0; proc < myRank; proc++){
        startOffset += workPerProcess[proc]; 
    }
  
    // Traversing the upper triangular part using currentRow and currentCol.
    unsigned long currentRow = 1; // these are for current row and column
    unsigned long currentCol = 1; 

    // Advancing to the starting point of this process's work.
    while (localCount <= startOffset) {
        localCount++;
        currentRow++;
        if (currentRow == (n + 1)) { // if reached the end of the row, move to next column
            currentCol++;
            currentRow = currentCol; // resetting column to currentRow
        }
    }
  
    // Number of products for this process.
    unsigned long section = workPerProcess[myRank];  
    // Allocate a boolean array to mark computed products.
    bool *productArray = (bool*) calloc(maxMatrix + 1, sizeof(bool));  
    if (productArray == NULL) {
        printf("Failed to allocate product array\n");
        MPI_Finalize();
        return 1;
    }

    // Each process computes its section of products.
    // Use localProductCount to count how many products have been computed.
    unsigned long localProductCount = 0;
    unsigned long product;
    while (localProductCount < section) {
        product = currentRow * currentCol;  // computing the current product
        if (product > maxMatrix) {
            product = maxMatrix;  // making sure not to get ahead of maxMatrix
        }
        productArray[product] = true;  // marking once the product is computed
        currentRow++;
        localProductCount++;
        if (currentRow == (n + 1)) {  //move to next column if reached the end row
            currentCol++;
            currentRow = currentCol;
        }
    }
	
    // Total number of elements in productArray.
    unsigned long totalElements = maxMatrix + 1;	
    bool *slice;
    // For slicing: determine full slices and the remaining slice size.
    unsigned long numFullSlices = totalElements / SLICER;
    unsigned long remainingSliceSize = totalElements % SLICER;
  
    if (myRank != 0) { // Non-root processes send their computed boolean array to root.
        unsigned long slicePos = 0;
        while (numFullSlices > 0) {
            slice = &productArray[slicePos];
            // Sending full slice using MPI_C_BOOL datatype.
            MPI_Send(slice, SLICER, MPI_C_BOOL, 0, TAG_ARRAY, MPI_COMM_WORLD);
            slicePos += SLICER;
            numFullSlices--;
        }
        slice = &productArray[slicePos]; // sending the remaining slice
        // Sending the last slice with remaining elements because the last slice may be smaller than SLICER.
        MPI_Send(slice, remainingSliceSize, MPI_C_BOOL, 0, TAG_LAST, MPI_COMM_WORLD);
        free(productArray); // freeing memory after sending
    }

    if (myRank == 0) { // Root process: merge arrays from other processes.
        bool *recvArray = (bool*) calloc(totalElements, sizeof(bool));
        if (recvArray == NULL) {
            printf("Failed to allocate recvArray\n");
            MPI_Finalize();
            return 1;
        }
        // For each non-root process, receive its slices and merge into productArray.
        for (int proc = 1; proc < numProc; proc++) {
            unsigned long slicePos = 0;
            unsigned long fullSlicesCopy = totalElements / SLICER;

            // Receiving full slices from other processes.
            while (fullSlicesCopy > 0) {
                MPI_Recv(recvArray, SLICER, MPI_C_BOOL, proc, TAG_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

                // Merging the received slice into productArray.
                for (unsigned long i = 0; i < SLICER; i++) {
                    if (recvArray[i]) { // If this product was computed by the sender, mark it.
                        productArray[slicePos + i] = true;
                    }
                }
                slicePos += SLICER;
                fullSlicesCopy--;
            }

            // Receiving the last slice with remaining elements.
            MPI_Recv(recvArray, remainingSliceSize, MPI_C_BOOL, proc, TAG_LAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (unsigned long i = 0; i < remainingSliceSize; i++) {
                if (recvArray[i]) {
                    productArray[slicePos + i] = true;
                }
            }
        }
        free(recvArray); // Freeing memory after receiving all slices.

        // Count total unique products marked true.
        unsigned long totalUniqueCount = 0;
        for (unsigned long idx = 1; idx <= maxMatrix; idx++) {
            if (productArray[idx]) {
                totalUniqueCount++;
            }
        }
        free(productArray); // Freeing memory after counting unique products.

        // Print the results.
        printf("-------------------------------------------------------------\n");
        printf("Function to determine the number of different elements\n");
        printf("-------------------------------------------------------------\n");
        printf("Dimension (N): %lu\n", n);
        printf("Total number of different elements: %lu\n", totalUniqueCount);
        endTime = MPI_Wtime();
        printf("Time elapsed: %lf seconds\n", endTime - startTime); // benchmarking execution time
        printf("-------------------------------------------------------------\n");
    }
    MPI_Finalize(); // Finalize MPI 
    return 0;
}
