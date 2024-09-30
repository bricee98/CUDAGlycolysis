#pragma once

#include <cuda_runtime.h>

enum AtomType {
    HYDROGEN,
    CARBON,
    NITROGEN,
    OXYGEN,
    PHOSPHORUS,
    SULFUR
};

class Atom {
public:
    // Add default constructor
    Atom();
    
    // Existing constructor
    Atom(AtomType type, float x, float y, float z, float charge, float mass);

    // Getters
    __host__ __device__ AtomType getType() const { return type; }
    __host__ __device__ float getX() const { return x; }
    __host__ __device__ float getY() const { return y; }
    __host__ __device__ float getZ() const { return z; }
    __host__ __device__ float getCharge() const { return charge; }
    __host__ __device__ float getMass() const { return mass; }
    __host__ __device__ float getVanDerWaalsRadius() const { return vanDerWaalsRadius; }
    __host__ __device__ float getEpsilon() const { return epsilon; }
    __host__ __device__ float getBornRadius() const { return bornRadius; }
    __host__ __device__ float getInverseBornRadius() const { return inverseBornRadius; }

    // Setters
    __host__ __device__ void setPosition(float newX, float newY, float newZ);
    __host__ __device__ void setBornRadius(float newBornRadius);
    __host__ __device__ void setInverseBornRadius(float newInverseBornRadius);

private:
    AtomType type;
    float x, y, z;
    float charge;
    float mass;
    float vanDerWaalsRadius;
    float epsilon;  // Lennard-Jones well depth
    float bornRadius;
    float inverseBornRadius;

    void initializeAtomProperties();
};