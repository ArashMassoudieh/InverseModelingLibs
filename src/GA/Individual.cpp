/*
 * OpenHydroQual - Environmental Modeling Platform
 * Copyright (C) 2025 Arash Massoudieh
 *
 * This file is part of OpenHydroQual.
 *
 * OpenHydroQual is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * If you use this file in a commercial product, you must purchase a
 * commercial license. Contact arash.massoudieh@enviroinformatics.co for details.
 */

#include "Individual.h"
#include "Binary.h"
#include <cstdlib>
#include <algorithm>
#include "QuickSort.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

// Default constructor - delegates to parameterized constructor
CIndividual::CIndividual()
    : CIndividual(1)  // Default: 1 parameter
{
}

// Primary constructor - all initialization happens here
CIndividual::CIndividual(int n)
    : nParams(n)
    , x(n, 0.0)
    , pert(n, 0.0)
    , dir(n, 0)
    , perteff(n, 0.0)
    , precision(n)
    , minrange(n, 0.0)
    , maxrange(n, 1.0)
    , fit_measures(3)  // Default to 3 measures
    , fitness(0.0)
    , actual_fitness(0.0)
    , actual_fitness2(0.0)
    , rank(0)
    , parent1(0)
    , parent2(0)
    , xsite(0)
{
}

// Copy constructor - use member initializer list
CIndividual::CIndividual(const CIndividual &C)
    : nParams(C.nParams)
    , x(C.x)
    , pert(C.pert)
    , dir(C.dir)
    , perteff(C.perteff)
    , precision(C.precision)
    , minrange(C.minrange)
    , maxrange(C.maxrange)
    , fit_measures(C.fit_measures)
    , parents(C.parents)
    , fitness(C.fitness)
    , actual_fitness(C.actual_fitness)
    , actual_fitness2(C.actual_fitness2)
    , rank(C.rank)
    , parent1(C.parent1)
    , parent2(C.parent2)
    , xsite(C.xsite)
{
}

// Destructor
CIndividual::~CIndividual()
{
}

// Assignment operator with self-assignment check
CIndividual& CIndividual::operator=(const CIndividual &C)
{
    // Check for self-assignment
    if (this == &C)
        return *this;

    // Copy all members
    nParams = C.nParams;
    x = C.x;
    pert = C.pert;
    dir = C.dir;
    perteff = C.perteff;
    precision = C.precision;
    minrange = C.minrange;
    maxrange = C.maxrange;
    parents = C.parents;
    fitness = C.fitness;
    fit_measures = C.fit_measures;
    actual_fitness = C.actual_fitness;
    actual_fitness2 = C.actual_fitness2;
    rank = C.rank;
    parent1 = C.parent1;
    parent2 = C.parent2;
    xsite = C.xsite;

    return *this;
}

//////////////////////////////////////////////////////////////////////
// Genetic Operations
//////////////////////////////////////////////////////////////////////

void CIndividual::initialize()
{
    for (int i = 0; i < nParams; i++)
    {
        x[i] = GetRndUnif(0, 1) * (maxrange[i] - minrange[i]) + minrange[i];
    }
}

void CIndividual::mutate(double mu)
{
    for (int i = 0; i < nParams; i++)
    {
        CBinary B = code(x[i], minrange[i], maxrange[i], precision[i]);
        B.mutate(mu);
        x[i] = B.decode(minrange[i]);

        // Clamp to valid range
        if (x[i] > maxrange[i])
            x[i] = maxrange[i];
    }
}

void CIndividual::shake(double shakescale)
{
    for (int i = 0; i < nParams; i++)
    {
        double dev = GetRndUnif(-shakescale, +shakescale);
        x[i] *= (1.0 + dev);

        // Clamp to valid range
        if (x[i] > maxrange[i])
            x[i] = maxrange[i];
        if (x[i] < minrange[i])
            x[i] = minrange[i];
    }
}

//////////////////////////////////////////////////////////////////////
// Parent Tracking
//////////////////////////////////////////////////////////////////////

void CIndividual::SetParents(int i)
{
    parents.clear();
    parents.push_back(i);
    parents.push_back(i);
}

void CIndividual::SetParents(int i, int j)
{
    parents.clear();
    parents.push_back(i);
    parents.push_back(j);
}

//////////////////////////////////////////////////////////////////////
// Free Functions
//////////////////////////////////////////////////////////////////////

double GetRndUnif(double xmin, double xmax)
{
    return double(rand()) / double(RAND_MAX) * (xmax - xmin) + xmin;
}

void cross(const CIndividual &I1, const CIndividual &I2,
           CIndividual &IR1, CIndividual &IR2)
{
    IR1 = I1;
    IR2 = I2;

    // Encode first parameter of both parents
    CBinary B1 = code(I1.x[0], I1.minrange[0], I1.maxrange[0], I1.precision[0]);
    CBinary B2 = code(I2.x[0], I2.minrange[0], I2.maxrange[0], I2.precision[0]);

    // Track end positions of each parameter in the concatenated binary
    std::vector<int> endpos(I1.nParams);
    endpos[0] = B1.getNDigits();

    // Concatenate all parameters into single binary strings
    for (int i = 1; i < I1.nParams; i++)
    {
        B1 = B1 + code(I1.x[i], I1.minrange[i], I1.maxrange[i], I1.precision[i]);
        B2 = B2 + code(I2.x[i], I2.minrange[i], I2.maxrange[i], I2.precision[i]);
        endpos[i] = B1.getNDigits();
    }

    // Generate random crossover points
    int nn = std::max(1, static_cast<int>(GetRndUnif(0, 1) * (I1.nParams / 4)));
    std::vector<int> rndint(nn);
    for (int i = 0; i < static_cast<int>(rndint.size()); i++)
        rndint[i] = static_cast<int>(GetRndUnif(0, 1) * B1.getNDigits());

    // Add boundary points
    rndint.push_back(0);
    rndint.push_back(B1.getNDigits() - 1);
    rndint = QSort(rndint);

    // Perform crossover on binary representations
    cross(B1, B2, rndint);

    // Decode first parameter
    CBinary BT1 = B1.extract(0, endpos[0] - 1);
    CBinary BT2 = B2.extract(0, endpos[0] - 1);
    BT1.setPrecision(I1.precision[0]);
    BT2.setPrecision(I2.precision[0]);

    IR1.x[0] = BT1.decode(I1.minrange[0]);
    IR2.x[0] = BT2.decode(I2.minrange[0]);

    // Clamp to range
    if (IR1.x[0] > IR1.maxrange[0])
        IR1.x[0] = IR1.maxrange[0];
    if (IR2.x[0] > IR2.maxrange[0])
        IR2.x[0] = IR2.maxrange[0];

    // Decode remaining parameters
    for (int i = 1; i < I1.nParams; i++)
    {
        CBinary BT1 = B1.extract(endpos[i - 1], endpos[i] - 1);
        CBinary BT2 = B2.extract(endpos[i - 1], endpos[i] - 1);
        BT1.setPrecision(I1.precision[i]);
        BT2.setPrecision(I2.precision[i]);

        IR1.x[i] = BT1.decode(I1.minrange[i]);
        IR2.x[i] = BT2.decode(I2.minrange[i]);

        // Clamp to range
        if (IR1.x[i] > IR1.maxrange[i])
            IR1.x[i] = IR1.maxrange[i];
        if (IR2.x[i] > IR2.maxrange[i])
            IR2.x[i] = IR2.maxrange[i];
    }
}

void cross2p(const CIndividual &I1, const CIndividual &I2,
             CIndividual &IR1, CIndividual &IR2)
{
    IR1 = I1;
    IR2 = I2;

    // Encode first parameter of both parents
    CBinary B1 = code(I1.x[0], I1.minrange[0], I1.maxrange[0], I1.precision[0]);
    CBinary B2 = code(I2.x[0], I2.minrange[0], I2.maxrange[0], I2.precision[0]);

    // Track end positions - use vector instead of fixed array
    std::vector<int> endpos(I1.nParams);
    endpos[0] = B1.getNDigits();

    // Concatenate all parameters into single binary strings
    for (int i = 1; i < I1.nParams; i++)
    {
        B1 = B1 + code(I1.x[i], I1.minrange[i], I1.maxrange[i], I1.precision[i]);
        B2 = B2 + code(I2.x[i], I2.minrange[i], I2.maxrange[i], I2.precision[i]);
        endpos[i] = B1.getNDigits();
    }

    // Generate two random crossover points
    int rndint1 = static_cast<int>(GetRndUnif(0, 1) * B1.getNDigits());
    int rndint2 = static_cast<int>(GetRndUnif(0, 1) * B1.getNDigits());

    // Ensure rndint1 <= rndint2
    if (rndint1 > rndint2)
    {
        std::swap(rndint1, rndint2);
    }

    // Perform two-point crossover
    cross2p(B1, B2, rndint1, rndint2);

    // Decode first parameter
    CBinary BT1 = B1.extract(0, endpos[0] - 1);
    CBinary BT2 = B2.extract(0, endpos[0] - 1);
    BT1.setPrecision(I1.precision[0]);
    BT2.setPrecision(I2.precision[0]);

    IR1.x[0] = BT1.decode(I1.minrange[0]);
    IR2.x[0] = BT2.decode(I2.minrange[0]);

    // Clamp to range
    if (IR1.x[0] > IR1.maxrange[0])
        IR1.x[0] = IR1.maxrange[0];
    if (IR2.x[0] > IR2.maxrange[0])
        IR2.x[0] = IR2.maxrange[0];

    // Decode remaining parameters
    for (int i = 1; i < I1.nParams; i++)
    {
        CBinary BT1 = B1.extract(endpos[i - 1], endpos[i] - 1);
        CBinary BT2 = B2.extract(endpos[i - 1], endpos[i] - 1);
        BT1.setPrecision(I1.precision[i]);
        BT2.setPrecision(I2.precision[i]);

        IR1.x[i] = BT1.decode(I1.minrange[i]);
        IR2.x[i] = BT2.decode(I2.minrange[i]);

        // Clamp to range
        if (IR1.x[i] > IR1.maxrange[i])
            IR1.x[i] = IR1.maxrange[i];
        if (IR2.x[i] > IR2.maxrange[i])
            IR2.x[i] = IR2.maxrange[i];
    }
}

void cross_RC_L(const CIndividual &I1, const CIndividual &I2,
                CIndividual &IR1, CIndividual &IR2)
{
    IR1 = I1;
    IR2 = I2;

    // Linear combination crossover (real-coded GA)
    for (int i = 0; i < IR1.nParams; i++)
    {
        double rnd = GetRndUnif(0, 1);
        IR1.x[i] = I1.x[i] * rnd + I2.x[i] * (1.0 - rnd);
        IR2.x[i] = I2.x[i] * rnd + I1.x[i] * (1.0 - rnd);
    }
}
