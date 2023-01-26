//----------------------------------------------------------------------------------------------
//	Filename:	nbody.cpp
//	Author:		Keith Bugeja
//----------------------------------------------------------------------------------------------
//  CPS3236 assignment for academic year 2017/2018:
//	Sample naive [O(n^2)] implementation for the n-Body problem.
//----------------------------------------------------------------------------------------------

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "vector2.h"

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
	
	Particle(void) 
		: Position( ((float)rand()) / RAND_MAX * fieldWidth - fieldHalfWidth,
					((float)rand()) / RAND_MAX * fieldHeight - fieldHalfHeight)
		, Velocity( 0.f, 0.f )
		, Mass ( ((float)rand()) / RAND_MAX * maxBodyMassVariance + minBodyMass )
	{ }
};

/*
 * Compute forces of particles exerted on one another
 */
void ComputeForces(std::vector<Particle> &p_bodies, float p_gravitationalTerm, float p_deltaT)
{
	Vector2 direction,
		force, acceleration;

	float distance;

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

/*
 * Update particle positions
 */
void MoveBodies(std::vector<Particle> &p_bodies, float p_deltaT)
{
	for (size_t j = 0; j < p_bodies.size(); ++j)
	{
		p_bodies[j].Position += p_bodies[j].Velocity * p_deltaT;
	}
}

/*
 * Commit particle masses and positions to file in CSV format
 */
void PersistPositions(const std::string &p_strFilename, std::vector<Particle> &p_bodies)
{
	std::cout << "Writing to file: " << p_strFilename << std::endl;
	
	std::ofstream output(p_strFilename.c_str());
	
	if (output.is_open())
	{	
		for (int j = 0; j < p_bodies.size(); j++)
		{
			output << 	p_bodies[j].Mass << ", " <<
				p_bodies[j].Position.Element[0] << ", " <<
				p_bodies[j].Position.Element[1] << std::endl;
		}
		
		output.close();
	}
	else
		std::cerr << "Unable to persist data to file:" << p_strFilename << std::endl;

}
