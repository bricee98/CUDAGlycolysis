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
    // Member variables
    AtomType type;
    float x, y, z;
    float charge;
    float mass;
    float vanDerWaalsRadius;
    float epsilon;       // Lennard-Jones parameter
    float bornRadius;

    // Constructors
    __host__ Atom() : type(HYDROGEN), x(0), y(0), z(0), charge(0), mass(1.008f) {
        initializeProperties();
    }

    __host__ Atom(AtomType t, float xPos, float yPos, float zPos, float q, float m)
        : type(t), x(xPos), y(yPos), z(zPos), charge(q), mass(m) {
        initializeProperties();
    }

private:
    void initializeProperties() {
        switch (type) {
            case HYDROGEN:
                vanDerWaalsRadius = 1.2f;
                epsilon = 0.0157f;
                break;
            case CARBON:
                vanDerWaalsRadius = 1.7f;
                epsilon = 0.0860f;
                break;
            case NITROGEN:
                vanDerWaalsRadius = 1.55f;
                epsilon = 0.1700f;
                break;
            case OXYGEN:
                vanDerWaalsRadius = 1.52f;
                epsilon = 0.2100f;
                break;
            case PHOSPHORUS:
                vanDerWaalsRadius = 1.8f;
                epsilon = 0.2000f;
                break;
            case SULFUR:
                vanDerWaalsRadius = 1.8f;
                epsilon = 0.2500f;
                break;
            default:
                vanDerWaalsRadius = 1.5f;
                epsilon = 0.1000f;
                break;
        }
        bornRadius = vanDerWaalsRadius; // Initial estimate
    }
};
