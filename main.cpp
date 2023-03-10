//----------------------------------------------------------------------------------------------
//	Filename:	nbody.cpp
//	Author:		Keith Bugeja
//----------------------------------------------------------------------------------------------
//  CPS3236 assignment for academic year 2017/2018:
//	Sample naive [O(n^2)] implementation for the n-Body problem.
//----------------------------------------------------------------------------------------------

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <omp.h>
#include "vector2.h"
#include "mpi.h"

#define SIZE 4

/*
 * Constant definitions for field dimensions, and particle masses
 */
const int fieldWidth = 1000;
const int fieldHalfWidth = fieldWidth >> 1;
const int fieldHeight = 1000;
const int fieldHalfHeight = fieldHeight >> 1;

const float minBodyMass = 2.5f;
const float maxBodyMassVariance = 5.f;

/*
 * Particle structure
 */
struct Particle
{
    Vector2 Position;
    Vector2 Velocity;
    float	Mass;

    Particle()
            : Position( ((float)rand()) / RAND_MAX * fieldWidth - fieldHalfWidth,
                        ((float)rand()) / RAND_MAX * fieldHeight - fieldHalfHeight)
            , Velocity( 0.f, 0.f )
            , Mass ( ((float)rand()) / RAND_MAX * maxBodyMassVariance + minBodyMass )
    { }
};

/*
 * Compute forces of particles exerted on one another
 */
void ComputeForces(std::vector<Particle> &p_bodies, float p_gravitationalTerm, float p_deltaT, int num_threads, int argc, char **argv)
{

    #pragma omp parallel num_threads(num_threads) shared(p_gravitationalTerm, p_bodies, p_deltaT) default(none)
    {
        Vector2 direction,
                force, acceleration;

        float distance;

        #pragma omp for
        for (size_t j = 0; j < p_bodies.size(); ++j)
        {
            Particle &p1 = p_bodies[j];

            force = 0.f, acceleration = 0.f;

            for (size_t k = 0; k < p_bodies.size(); ++k)
            {
                if (k == j) continue;

                Particle &p2 = p_bodies[k];

                // Compute direction vector
                direction = p2.Position - p1.Position;

                // Limit distance term to avoid singularities
                distance = std::max<float>( 0.5f * (p2.Mass + p1.Mass), fabs(direction.Length()) );

                // Accumulate force
                force += direction / (distance * distance * distance) * p2.Mass;
            }

            // Compute acceleration for body
            acceleration = force * p_gravitationalTerm;

            // Integrate velocity (m/s)
            p1.Velocity += acceleration * p_deltaT;
        }
    }
}

/*
 * Update particle positions
 */
void MoveBodies(std::vector<Particle> &p_bodies, float p_deltaT, int num_threads)
{
    #pragma omp parallel num_threads(num_threads) shared(p_bodies, p_deltaT) default(none)
    #pragma omp for
    for (auto & p_bodie : p_bodies)
    {
        p_bodie.Position += p_bodie.Velocity * p_deltaT;
    }
}

/*
 * Commit particle masses and positions to file in CSV format
 */
void PersistPositions(const std::string &p_strFilename, std::vector<Particle> &p_bodies)
{
    std::cout << "\nWriting to file: " << p_strFilename << std::endl;

    std::ofstream output(p_strFilename.c_str());

    if (output.is_open())
    {
        for (auto & p_bodie : p_bodies)
        {
            output << 	p_bodie.Mass << ", " <<
                   p_bodie.Position.Element[0] << ", " <<
                   p_bodie.Position.Element[1] << std::endl;
        }

        output.close();
    }
    else
        std::cerr << "Unable to persist data to file:" << p_strFilename << std::endl;

}


int main(int argc, char **argv)
{
    int counter;
    bool enable_output = true;
    std::string input_file;

    int particleCount = 64;
    int maxIteration = 10;
    float deltaT = 0.005f;
    float gTerm = 2000.f;

    std::stringstream fileOutput;
    std::vector<Particle> bodies;

    for(counter=0;counter<argc;counter++) {
        if(std::strcmp(argv[counter], "-o") == 0) {
            if(std::strcmp(argv[counter+1], "false") == 0) {
                enable_output = false;
            }
            printf("\nenable_output: %d", enable_output);
            continue;
        } else if(std::strcmp(argv[counter], "-b") == 0) {
            particleCount = strtol(argv[counter+1],   nullptr, 10);
            printf("\nparticleCount: %d", particleCount);
            continue;
        } else if(std::strcmp(argv[counter], "-g") == 0) {
            gTerm = strtol(argv[counter+1],   nullptr, 10);
            printf("\ngTerm: %f", gTerm);
            continue;
        } else if(std::strcmp(argv[counter], "-i") == 0) {
            maxIteration = strtol(argv[counter+1],   nullptr, 10);
            printf("\nmaxIteration: %d", maxIteration);
            continue;
        } else if(std::strcmp(argv[counter], "-d") == 0) {
            deltaT = strtol(argv[counter+1],   nullptr, 10);
            printf("\ndeltaT: %f", deltaT);
            continue;
        } else if(std::strcmp(argv[counter], "-f") == 0) {
            input_file = argv[counter+1];
            continue;
        }
        printf("\nargv[%d]: %s", counter, argv[counter]);
    }

    if (input_file.empty()) {
        for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
            bodies.push_back(Particle());
    } else {
        std::ifstream input_stream(input_file, std::ios_base::in);

        if (!input_stream) std::cerr << "Can't open input file!";

        std::string line;
        while (getline(input_stream, line)) {

            Particle particle;
            int it = 0;
            float mass;
            float x;
            float y;

            size_t last = 0;
            size_t next = 0;
            while ((next = line.find(',', last)) != std::string::npos)
            {
                std::__sso_string var = line.substr(last, next-last);
                if(it == 0) {
                    mass = std::stof(var);
                } else {
                    x = std::stof(var);
                }
                it = it + 1;
                last = next + 1;
            }
            std::__sso_string var = line.substr(last, (line.length()-last));
            y = std::stof(var);

            particle.Mass = mass;
            particle.Position.Set(x, y);

            bodies.push_back(particle);
        }
    }

    // Declare and initialise stuff
    int numtasks, rank, sendcount, recvcount, source;

    // Initialise MPI environment
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    std::size_t const size = bodies.size() / numtasks;

    // printf ("Number of MPI tasks: %d, My rank: %d, Tid: %d\n", numtasks, rank, omp_get_thread_num());
    enable_output = false;
    double start_time = omp_get_wtime();
    int num_threads = 1;

    for (int iteration = 0; iteration < maxIteration; ++iteration)
    {
        if (enable_output and rank == 0) {
            fileOutput.str(std::string());
            fileOutput << "./NBodyOutput/nbody_-1_" << iteration << ".txt";
            PersistPositions(fileOutput.str(), bodies);
        }

        std::vector<Particle> bodies_part(bodies.begin() + size*rank, bodies.begin() + size*(rank+1));
        std::vector<Particle> new_bodies;
        std::vector<Particle> vec(size);
        std::vector<std::vector<Particle>> sample(size, std::vector<Particle>(size));
        // printf("\nBegin: %lu, end: %lu, tasks: %d, My rank: %d", size*rank, size*(rank+1), numtasks, rank);

        ComputeForces(bodies_part, gTerm, deltaT, num_threads, argc, argv);
        MoveBodies(bodies_part, deltaT, num_threads);

        MPI_Barrier(MPI_COMM_WORLD);

        for (int j = 0; j < numtasks; j++) {
            if (rank == j) {
                MPI_Send(&bodies_part.front(), bodies_part.size(), MPI_FLOAT, 0, 10 + j, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            for (int j = 0; j < numtasks; j++) {
                MPI_Status status;
                MPI_Recv(&vec.front(), bodies_part.size(), MPI_FLOAT, j, 10 + j, MPI_COMM_WORLD, &status);
                for(const Particle& value : vec)
                {
                    sample[j].push_back(value);
                }
            }

            for (auto &&v : sample) {
                new_bodies.insert(new_bodies.end(), v.begin(), v.end());
            }
            bodies.clear();
            bodies = new_bodies;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (enable_output and rank == 0) {
            fileOutput.str(std::string());
            fileOutput << "./NBodyOutput/nbody_from0_" << iteration << ".txt";
            PersistPositions(fileOutput.str(), bodies);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&bodies.front(), bodies.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        if (enable_output and rank == 1) {
            fileOutput.str(std::string());
            fileOutput << "./NBodyOutput/nbody_from1_" << iteration << ".txt";
            PersistPositions(fileOutput.str(), bodies);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (enable_output and rank == 0) {
            fileOutput.str(std::string());
            fileOutput << "./NBodyOutput/nbody_" << iteration << ".txt";
            PersistPositions(fileOutput.str(), bodies);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    printf("Execution time: %.2f ms\n", (omp_get_wtime() - start_time) * 1e3);
    return 0;
}
