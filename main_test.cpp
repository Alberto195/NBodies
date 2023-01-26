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
#include <cstddef>

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
void ComputeForces(std::vector<Particle> &p_bodies, float p_gravitationalTerm, float p_deltaT, int num_threads)
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
    double start_time = omp_get_wtime();
    int counter;
    bool enable_output = true;
    std::string input_file;

    int particleCount = 1024;
    int maxIteration = 1000;
    float deltaT = 0.005f;
    float gTerm = 1.f;
    int num_threads = 1;

    std::stringstream fileOutput;
    std::vector<Particle> bodies;

    // Declare and initialise stuff
    int numtasks, rank, rc;

    // Initialise MPI environment
    // Initialise MPI and check for errors
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS)
    {
        printf ("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

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
        } else if(std::strcmp(argv[counter], "-n") == 0) {
            num_threads = strtol(argv[counter+1],   nullptr, 10);
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
                // printf("%s", var.c_str());
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

    printf("\nBody Size: %zu", bodies.size());
    printf("\nExecution time middle: %.2f ms\n", (omp_get_wtime() - start_time) * 1e3);

    std::size_t const size = bodies.size() / numtasks;

    start_time = omp_get_wtime();

    std::vector<Particle> bodies_part(bodies.begin() + size*rank, bodies.begin() + size*(rank+1));
    printf("\nBodies_part Size: %zu", bodies_part.size());
    // printf ("\nNumber of MPI tasks: %d, My rank: %d, Tid: %d", numtasks, rank, omp_get_thread_num());

    for (int iteration = 0; iteration < maxIteration; ++iteration)
    {
        printf("\niteration: %d", iteration);
        ComputeForces(bodies_part, gTerm, deltaT, num_threads);
        MoveBodies(bodies_part, deltaT, num_threads);

        std::vector<float> floats;
        for(const Particle& value : bodies_part)
        {
            floats.push_back(value.Mass);
            floats.push_back(value.Position.X);
            floats.push_back(value.Position.Y);
            floats.push_back(value.Position.Element[0]);
            floats.push_back(value.Position.Element[1]);
            floats.push_back(value.Velocity.X);
            floats.push_back(value.Velocity.Y);
            floats.push_back(value.Velocity.Element[0]);
            floats.push_back(value.Velocity.Element[1]);
        }

        rc = MPI_Send(&floats.front(), 1, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
        if (rc != MPI_SUCCESS)
        {
            printf ("Error send MPI program. Terminating.\n");
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        rc = MPI_Barrier(MPI_COMM_WORLD);
        if (rc != MPI_SUCCESS)
        {
            printf ("Error in Barrier 2 MPI program. Terminating.\n");
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        if (rank == 0) {
            bodies.clear();
            for (int j = 0; j < numtasks; j++) {
                std::vector<float> buffer(bodies_part.size()*9);
                MPI_Status status;
                rc = MPI_Recv(&buffer.front(), 1, MPI_FLOAT, j,  j, MPI_COMM_WORLD, &status);
                if (rc != MPI_SUCCESS)
                {
                    printf ("Error receive MPI program. Terminating.\n");
                    MPI_Abort(MPI_COMM_WORLD, rc);
                }

                for (int i = 0; i < bodies_part.size(); i++) {
                    Particle particle;
                    particle.Mass = buffer[0 + i*9];
                    particle.Position.X = buffer[1+ i*9];
                    particle.Position.Y = buffer[2+ i*9];
                    particle.Position.Element[0] = buffer[3+ i*9];
                    particle.Position.Element[1] = buffer[4+ i*9];
                    particle.Velocity.X = buffer[5+ i*9];
                    particle.Velocity.Y = buffer[6+ i*9];
                    particle.Velocity.Element[0] = buffer[7+ i*9];
                    particle.Velocity.Element[1] = buffer[8+ i*9];
                    bodies.push_back(particle);
                }
            }
        }
        floats.clear();

        if (enable_output and rank == 0) {
            fileOutput.str(std::string());
            fileOutput << "./NBodyOutput/nbody_" << iteration << ".txt";
            PersistPositions(fileOutput.str(), bodies);
        }
    }

    rc = MPI_Barrier(MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
    {
        printf ("Error in last Barrier MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    rc = MPI_Finalize();
    if (rc != MPI_SUCCESS)
    {
        printf ("Error in Finalize MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    printf("\nExecution time: %.2f ms\n", (omp_get_wtime() - start_time) * 1e3);
    return 0;
}
