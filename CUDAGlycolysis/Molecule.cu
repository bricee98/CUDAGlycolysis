#include "Molecule.cuh"
#include <cmath>
#include <cstdlib>

__host__ Molecule::Molecule()
    : type(WATER),
      markedForDeletion(false), creationFlag(WATER),
      centerOfMass(make_float3(0.0f, 0.0f, 0.0f)), radius(0.0f), mass(0.0f), totalCharge(0.0f) {}


// Helper function to create coarse-grained molecules
__host__ Molecule createCoarseGrainedMolecule(MoleculeType type, float radius, float mass, float totalCharge) {
    Molecule m;
    m.type = type;
    m.radius = radius;
    m.mass = mass;
    m.totalCharge = totalCharge;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    return m;
}

__host__ Molecule Molecule::createNone() {
    return createCoarseGrainedMolecule(NONE, 0.0f, 0.0f, 0.0f);
}

__host__ Molecule Molecule::createGlucose() {
    return createCoarseGrainedMolecule(GLUCOSE, 0.38f, 180.156f, 0.0f);
}

__host__ Molecule Molecule::createATP() {
    return createCoarseGrainedMolecule(ATP, 0.47f, 507.181f, -4.0f);
}

__host__ Molecule Molecule::createADP() {
    return createCoarseGrainedMolecule(ADP, 0.45f, 427.201f, -3.0f);
}

__host__ Molecule Molecule::createGlucose6Phosphate() {
    return createCoarseGrainedMolecule(GLUCOSE_6_PHOSPHATE, 0.42f, 260.136f, -2.0f);
}

__host__ Molecule Molecule::createFructose6Phosphate() {
    return createCoarseGrainedMolecule(FRUCTOSE_6_PHOSPHATE, 0.42f, 260.136f, -2.0f);
}

__host__ Molecule Molecule::createFructose16Bisphosphate() {
    return createCoarseGrainedMolecule(FRUCTOSE_1_6_BISPHOSPHATE, 0.46f, 340.116f, -4.0f);
}

__host__ Molecule Molecule::createDihydroxyacetonePhosphate() {
    return createCoarseGrainedMolecule(DIHYDROXYACETONE_PHOSPHATE, 0.36f, 170.058f, -2.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3Phosphate() {
    return createCoarseGrainedMolecule(GLYCERALDEHYDE_3_PHOSPHATE, 0.36f, 170.058f, -2.0f);
}

__host__ Molecule Molecule::create13Bisphosphoglycerate() {
    return createCoarseGrainedMolecule(_1_3_BISPHOSPHOGLYCERATE, 0.41f, 266.047f, -4.0f);
}

__host__ Molecule Molecule::create3Phosphoglycerate() {
    return createCoarseGrainedMolecule(_3_PHOSPHOGLYCERATE, 0.38f, 186.067f, -3.0f);
}

__host__ Molecule Molecule::create2Phosphoglycerate() {
    return createCoarseGrainedMolecule(_2_PHOSPHOGLYCERATE, 0.38f, 186.067f, -3.0f);
}

__host__ Molecule Molecule::createPhosphoenolpyruvate() {
    return createCoarseGrainedMolecule(PHOSPHOENOLPYRUVATE, 0.37f, 168.042f, -3.0f);
}

__host__ Molecule Molecule::createPyruvate() {
    return createCoarseGrainedMolecule(PYRUVATE, 0.33f, 88.062f, -1.0f);
}

__host__ Molecule Molecule::createNADPlus() {
    return createCoarseGrainedMolecule(NAD_PLUS, 0.54f, 663.425f, -1.0f);
}

__host__ Molecule Molecule::createNADH() {
    return createCoarseGrainedMolecule(NADH, 0.54f, 665.441f, -2.0f);
}

__host__ Molecule Molecule::createProton() {
    return createCoarseGrainedMolecule(PROTON, 0.01f, 1.008f, 1.0f);
}

__host__ Molecule Molecule::createInorganicPhosphate() {
    return createCoarseGrainedMolecule(INORGANIC_PHOSPHATE, 0.28f, 95.979f, -2.0f);
}

__host__ Molecule Molecule::createWater() {
    return createCoarseGrainedMolecule(WATER, 0.14f, 18.015f, 0.0f);
}

__host__ Molecule Molecule::createAMP() {
    return createCoarseGrainedMolecule(AMP, 0.43f, 347.221f, -2.0f);
}

__host__ Molecule Molecule::createCitrate() {
    return createCoarseGrainedMolecule(CITRATE, 0.40f, 192.124f, -3.0f);
}

__host__ Molecule Molecule::createFructose26Bisphosphate() {
    return createCoarseGrainedMolecule(FRUCTOSE_2_6_BISPHOSPHATE, 0.46f, 340.116f, -4.0f);
}

// Helper function to create coarse-grained enzymes
__host__ Molecule createCoarseGrainedEnzyme(MoleculeType type, float radius, float mass) {
    Molecule m;
    m.type = type;
    m.radius = radius;
    m.mass = mass;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    return m;
}

__host__ Molecule Molecule::createHexokinase() {
    return createCoarseGrainedEnzyme(HEXOKINASE, 5.0f, 100000.0f);
}

__host__ Molecule Molecule::createGlucose6PhosphateIsomerase() {
    return createCoarseGrainedEnzyme(PHOSPHOGLUCOSE_ISOMERASE, 4.8f, 95000.0f);
}

__host__ Molecule Molecule::createPhosphofructokinase1() {
    return createCoarseGrainedEnzyme(PHOSPHOFRUCTOKINASE_1, 5.2f, 105000.0f);
}

__host__ Molecule Molecule::createAldolase() {
    return createCoarseGrainedEnzyme(ALDOLASE, 5.1f, 102000.0f);
}

__host__ Molecule Molecule::createTriosephosphateIsomerase() {
    return createCoarseGrainedEnzyme(TRIOSEPHOSPHATE_ISOMERASE, 4.5f, 90000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateDehydrogenase() {
    return createCoarseGrainedEnzyme(GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE, 5.3f, 106000.0f);
}

__host__ Molecule Molecule::createPhosphoglycerateKinase() {
    return createCoarseGrainedEnzyme(PHOSPHOGLYCERATE_KINASE, 4.9f, 98000.0f);
}

__host__ Molecule Molecule::createPhosphoglycerateMutase() {
    return createCoarseGrainedEnzyme(PHOSPHOGLYCERATE_MUTASE, 4.7f, 94000.0f);
}

__host__ Molecule Molecule::createEnolase() {
    return createCoarseGrainedEnzyme(ENOLASE, 5.0f, 100000.0f);
}

__host__ Molecule Molecule::createPyruvateKinase() {
    return createCoarseGrainedEnzyme(PYRUVATE_KINASE, 5.4f, 108000.0f);
}

// Helper function to create coarse-grained enzyme complexes
__host__ Molecule createCoarseGrainedEnzymeComplex(MoleculeType type, float radius, float mass) {
    Molecule m;
    m.type = type;
    m.radius = radius;
    m.mass = mass;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    return m;
}

__host__ Molecule Molecule::createHexokinaseGlucoseComplex() {
    return createCoarseGrainedEnzymeComplex(HEXOKINASE_GLUCOSE_COMPLEX, 5.2f, 105000.0f);
}

__host__ Molecule Molecule::createHexokinaseGlucoseATPComplex() {
    return createCoarseGrainedEnzymeComplex(HEXOKINASE_GLUCOSE_ATP_COMPLEX, 5.4f, 110000.0f);
}

__host__ Molecule Molecule::createGlucose6PhosphateIsomeraseComplex() {
    return createCoarseGrainedEnzymeComplex(GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX, 5.0f, 100000.0f);
}

__host__ Molecule Molecule::createFructose6PhosphateIsomeraseComplex() {
    return createCoarseGrainedEnzymeComplex(FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX, 5.0f, 100000.0f);
}

__host__ Molecule Molecule::createPhosphofructokinase1Complex() {
    return createCoarseGrainedEnzymeComplex(PHOSPHOFRUCTOKINASE_1_COMPLEX, 5.3f, 106000.0f);
}

__host__ Molecule Molecule::createPhosphofructokinase1ATPComplex() {
    return createCoarseGrainedEnzymeComplex(PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX, 5.5f, 111000.0f);
}

__host__ Molecule Molecule::createFructose16BisphosphateAldolaseComplex() {
    return createCoarseGrainedEnzymeComplex(FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX, 5.2f, 104000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateAldolaseComplex() {
    return createCoarseGrainedEnzymeComplex(GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX, 5.2f, 104000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateAldolaseDHAPComplex() {
    return createCoarseGrainedEnzymeComplex(GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX, 5.3f, 106000.0f);
}

__host__ Molecule Molecule::createDHAPTriosephosphateIsomeraseComplex() {
    return createCoarseGrainedEnzymeComplex(DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX, 4.7f, 94000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateTriosephosphateIsomeraseComplex() {
    return createCoarseGrainedEnzymeComplex(GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX, 4.7f, 94000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateDehydrogenaseComplex() {
    return createCoarseGrainedEnzymeComplex(GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX, 5.4f, 108000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateDehydrogenaseNADPlusComplex() {
    return createCoarseGrainedEnzymeComplex(GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX, 5.5f, 110000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateDehydrogenaseNADPlusInorganicPhosphateComplex() {
    return createCoarseGrainedEnzymeComplex(GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX, 5.6f, 112000.0f);
}

__host__ Molecule Molecule::createPhosphoglycerateKinaseComplex() {
    return createCoarseGrainedEnzymeComplex(PHOSPHOGLYCERATE_KINASE_COMPLEX, 5.0f, 100000.0f);
}

__host__ Molecule Molecule::createPhosphoglycerateKinaseADPComplex() {
    return createCoarseGrainedEnzymeComplex(PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX, 5.2f, 104000.0f);
}

__host__ Molecule Molecule::createPhosphoglycerateMutaseComplex() {
    return createCoarseGrainedEnzymeComplex(PHOSPHOGLYCERATE_MUTASE_COMPLEX, 4.8f, 96000.0f);
}

__host__ Molecule Molecule::createEnolaseComplex() {
    return createCoarseGrainedEnzymeComplex(ENOLASE_COMPLEX, 5.1f, 102000.0f);
}

__host__ Molecule Molecule::createPyruvateKinaseComplex() {
    return createCoarseGrainedEnzymeComplex(PYRUVATE_KINASE_COMPLEX, 5.5f, 110000.0f);
}

__host__ Molecule Molecule::createPyruvateKinaseADPComplex() {
    return createCoarseGrainedEnzymeComplex(PYRUVATE_KINASE_ADP_COMPLEX, 5.6f, 112000.0f);
}