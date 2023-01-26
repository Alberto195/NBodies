#include <stdio.h>
#include <mpi.h>
#include <omp.h>

// Main program 
int main(int argc, char **argv)
{
    int  numtasks, rank, len, rc;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    // Initialise MPI and check for errors
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS)
    {
        printf ("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Code to be executed on each instance 
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Get_processor_name(hostname, &len);

#pragma omp parallel
    {
        // Obtain thread number
        unsigned tid = omp_get_thread_num();

        printf ("Number of MPI tasks: %d, My rank: %d, Tid: %d, Running on %s\n", numtasks,rank,tid, hostname);
    }

    // Finalize MPI process
    MPI_Finalize();
}