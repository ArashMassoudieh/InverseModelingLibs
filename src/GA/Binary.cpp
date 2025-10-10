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

#include "Binary.h"
#include "math.h"
#include <iostream>
#include <stdexcept>
#include "DistributionNUnif.h"
#ifdef QT_version
#include "qdebug.h"
#endif

using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

// Private helper method - centralizes initialization logic
void CBinary::initialize(int n, int preci)
{
    nDigits = n;
    precision = preci;
    Digit.resize(nDigits);
    sign = true;
}

// Default constructor - delegates to parameterized constructor
CBinary::CBinary()
    : CBinary(0, DEFAULT_PRECISION)
{
}

// Single parameter constructor - delegates to two-parameter constructor
CBinary::CBinary(int n)
    : CBinary(n, DEFAULT_PRECISION)
{
}

// Primary constructor - all initialization happens here
CBinary::CBinary(int n, int preci)
    : nDigits(n)
    , precision(preci)
    , Digit(n)
    , sign(true)
{
}

// Copy constructor - use member initializer list
CBinary::CBinary(const CBinary &B)
    : nDigits(B.nDigits)
    , precision(B.precision)
    , Digit(B.Digit)
    , sign(B.sign)
{
}

// Destructor - default is fine (vector cleans itself up)
CBinary::~CBinary()
{
}

// Assignment operator - return reference for chaining, check self-assignment
CBinary& CBinary::operator=(const CBinary &B)
{
    if (this != &B) // Check for self-assignment
    {
        nDigits = B.nDigits;
        precision = B.precision;
        Digit = B.Digit;
        sign = B.sign;
    }
    return *this;
}

// Decode method - now const-correct
double CBinary::decode(double minrange) const
{
    double sum = 0.0;
    for (int i = nDigits - 1; i >= 0; i--)
    {
        if (Digit[i] == true)
            sum += pow(2.0, nDigits - i - 1);
    }
    return sum / pow(10.0, precision) + minrange;
}

// Free function for encoding
CBinary code(double x, double minrange, double maxrange, int precision)
{
    int n = static_cast<int>(log((maxrange - minrange) * pow(10.0, precision)) / log(2.0) + 1);
    int xi = static_cast<int>((x - minrange) * pow(10.0, precision));
    CBinary B(n, precision);  // Use two-parameter constructor

    for (int i = 0; i < n; i++)
    {
        B[B.getNDigits() - i - 1] = (xi % 2 == 1);
        xi = static_cast<int>(xi / 2);
    }

    return B;
}

// Addition operator - concatenation, now const
CBinary CBinary::operator+(const CBinary &B1) const
{
    int n = nDigits + B1.nDigits;
    CBinary B(n);

    for (int i = 0; i < n; i++)
    {
        if (i < nDigits)
            B[i] = Digit[i];
        else
            B[i] = B1.Digit[i - nDigits];
    }

    return B;
}

// Extract substring
CBinary CBinary::extract(int spoint, int epoint) const
{
    if (spoint < 0 || epoint >= nDigits || spoint > epoint)
        throw std::out_of_range("Invalid extract range");

    int n = epoint - spoint + 1;
    CBinary B(n);

    for (int i = 0; i < n; i++)
    {
        B[i] = Digit[i + spoint];
    }

    return B;
}

// Subscript operator - safer version with bounds checking
int& CBinary::operator[](unsigned int i)
{
    if (i >= Digit.size())
        throw std::out_of_range("CBinary index out of range");
    return Digit[i];
}

// Const version of subscript operator
const int& CBinary::operator[](unsigned int i) const
{
    if (i >= Digit.size())
        throw std::out_of_range("CBinary index out of range");
    return Digit[i];
}

// Crossover - single point
void cross(CBinary &B1, CBinary &B2, int p)
{
    CBinary BT1 = B1;
    CBinary BT2 = B2;

    for (int i = 0; i < B1.getNDigits(); i++)
    {
        if (i < p)
        {
            B1[i] = BT1[i];
            B2[i] = BT2[i];
        }
        else
        {
            B1[i] = BT2[i];
            B2[i] = BT1[i];
        }
    }
}

// Crossover - multiple points
void cross(CBinary &B1, CBinary &B2, vector<int> p)
{
    CBinary BT1 = B1;
    CBinary BT2 = B2;

    for (int i = 0; i < B1.getNDigits(); i++)
    {
        for (unsigned int j = 1; j < p.size(); j++)
        {
            if (p[j - 1] < i && i < p[j])
            {
                if (i % 2 == 0)
                {
                    B1[i] = BT1[i];
                    B2[i] = BT2[i];
                }
                else
                {
                    B1[i] = BT2[i];
                    B2[i] = BT1[i];
                }
            }
        }
    }
}

// Crossover - two point
void cross2p(CBinary &B1, CBinary &B2, int p1, int p2)
{
    CBinary BT1 = B1;
    CBinary BT2 = B2;

    for (int i = 0; i < B1.getNDigits(); i++)
    {
        if (i < p1)
        {
            B1[i] = BT1[i];
            B2[i] = BT2[i];
        }
        else if (i >= p1 && i < p2)
        {
            B1[i] = BT2[i];
            B2[i] = BT1[i];
        }
        else // i >= p2
        {
            B1[i] = BT1[i];
            B2[i] = BT2[i];
        }
    }
}

// Show method - for debugging
void CBinary::show() const
{
    for (int i = 0; i < nDigits; i++)
        cout << Digit[i];
    cout << endl;
}

// Mutation
void CBinary::mutate(double mu)
{
    for (int i = 0; i < nDigits; i++)
    {
        if (GetRndUniF(0, 1) < mu)
            Digit[i] = !Digit[i];
    }
}
