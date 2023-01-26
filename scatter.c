#include "mpi.h"
#include <stdio.h>

#define SIZE 4

// Perform a scatter operation on the rows of an array
int main(int argc, char *argv[])
{
    // Declare and initialise stuff
    int numtasks, rank, sendcount, recvcount, source;

    // Receive buffer
    float recvbuf[SIZE];

    // Send buffer (SIZE x SIZE matrix)
    float sendbuf[SIZE][SIZE] = { {1.0,  2.0,  3.0,  4.0},
                                  {5.0,  6.0,  7.0,  8.0},
                                  {9.0,  10.0, 11.0, 12.0},
                                  {13.0, 14.0, 15.0, 16.0}  };

    // Initialise MPI environment
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // Check if the number of processes matches data shape
    if (numtasks == SIZE)
    {
        source = 1;       // Process which will perform the scatter (root)
        sendcount = SIZE; // Number of elements which root will send to each process
        recvcount = SIZE; // Number of elements which will be received by each process

        // Perform scatter
        MPI_Scatter(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount,
                    MPI_FLOAT, source, MPI_COMM_WORLD);

        // Check received data
        printf("rank= %d  Results: %f %f %f %f\n",rank, recvbuf[0],
               recvbuf[1],
               recvbuf[2],
               recvbuf[3]);
    }
    else
        // Incorrect configuration, do nothing
        printf("Must specify %d processors. Terminating.\n",SIZE);

    // Tear down MPI environment
    MPI_Finalize();
}
