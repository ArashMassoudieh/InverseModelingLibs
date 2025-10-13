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

#ifndef BINARY_H
#define BINARY_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <random>

using namespace std;

/**
 * @class CBinary
 * @brief Binary-encoded representation of real numbers for genetic algorithms
 *
 * CBinary represents a real number as a binary string (chromosome) for use in
 * genetic algorithms. It handles encoding/decoding between real numbers and
 * binary representations, and supports genetic operators like crossover and
 * mutation.
 *
 * Encoding scheme:
 * - A real number x in range [minrange, maxrange] is encoded as an integer
 * - The integer represents (x - minrange) * 10^precision
 * - This integer is then converted to binary representation
 * - Example: 3.567 with precision=3 and minrange=0 becomes 3567 in binary
 *
 * @note The binary representation uses vector<int> where each element is 0 or 1
 */
class CBinary
{
public:
    // ========================================================================
    // Constructors and Destructor
    // ========================================================================

    /**
     * @brief Default constructor - creates empty binary with default precision
     * Creates a binary with 0 digits and precision=3
     */
    CBinary();

    /**
     * @brief Construct binary with specified number of digits
     * @param n Number of binary digits to allocate
     * Uses default precision of 3
     */
    explicit CBinary(int n);

    /**
     * @brief Construct binary with specified digits and precision
     * @param n Number of binary digits to allocate
     * @param preci Decimal precision for encoding/decoding (number of decimal places)
     */
    CBinary(int n, int preci);

    /**
     * @brief Copy constructor - deep copy of another binary
     * @param B The binary to copy from
     */
    CBinary(const CBinary &B);

    /**
     * @brief Virtual destructor
     */
    virtual ~CBinary();

    // ========================================================================
    // Operators
    // ========================================================================

    /**
     * @brief Assignment operator with self-assignment safety
     * @param B The binary to assign from
     * @return Reference to this object for chaining (a = b = c)
     */
    CBinary& operator=(const CBinary &B);

    /**
     * @brief Concatenation operator - combines two binaries
     * @param B The binary to concatenate to this one
     * @return New binary containing this binary followed by B
     *
     * Example: If this=[1,0,1] and B=[1,1], result=[1,0,1,1,1]
     */
    CBinary operator+(const CBinary &B) const;

    /**
     * @brief Subscript operator for non-const access to bits
     * @param i Index of bit to access (0-based)
     * @return Reference to the bit value (0 or 1)
     * @throws std::out_of_range if index is out of bounds
     */
    int& operator[](unsigned int i);

    /**
     * @brief Subscript operator for const access to bits
     * @param i Index of bit to access (0-based)
     * @return Const reference to the bit value (0 or 1)
     * @throws std::out_of_range if index is out of bounds
     */
    const int& operator[](unsigned int i) const;

    // ========================================================================
    // Core Functionality
    // ========================================================================

    /**
     * @brief Decode binary representation back to real number
     * @param minrange Minimum value of the encoded range
     * @return The decoded real number
     *
     * Converts the binary digits to an integer, then applies the formula:
     * result = (binary_as_integer / 10^precision) + minrange
     *
     * Example: Binary [1,0,1,1] = 11 decimal
     *          With precision=2 and minrange=5.0
     *          Result = 11/100 + 5.0 = 5.11
     */
    double decode(double minrange) const;

    /**
     * @brief Extract a substring of bits
     * @param spoint Start position (inclusive, 0-based)
     * @param epoint End position (inclusive, 0-based)
     * @return New binary containing bits from [spoint, epoint]
     * @throws std::out_of_range if indices are invalid
     *
     * Example: If this=[1,0,1,1,0], extract(1,3) returns [0,1,1]
     */
    CBinary extract(int spoint, int epoint) const;

    /**
     * @brief Apply mutation operator - randomly flip bits
     * @param mu Mutation probability (0.0 to 1.0) for each bit
     *
     * Each bit has probability 'mu' of being flipped (0->1 or 1->0)
     * Typical GA mutation rates are 0.001 to 0.05
     */
    void mutate(double mu);

    /**
     * @brief Print binary representation to console (for debugging)
     * Outputs the sequence of 0s and 1s followed by newline
     */
    void show() const;

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * @brief Get number of binary digits
     * @return Number of bits in this binary
     */
    int getNDigits() const { return nDigits; }

    /**
     * @brief Get decimal precision
     * @return Precision used for encoding/decoding
     */
    int getPrecision() const { return precision; }

    /**
     * @brief Get sign flag (currently unused in implementation)
     * @return Sign flag value
     */
    bool getSign() const { return sign; }

    /**
     * @brief Set decimal precision
     * @param preci New precision value
     */
    void setPrecision(int preci) { precision = preci; }

private:
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<double> uniformDistribution;

    /// Default precision for encoding/decoding (3 decimal places)
    static constexpr int DEFAULT_PRECISION = 3;

    /// Number of binary digits in this representation
    int nDigits;

    /// Decimal precision: how many decimal places to preserve
    /// Example: precision=3 means values like 1.234
    int precision;

    /// The actual binary digits stored as vector of 0s and 1s
    vector<int> Digit;

    /// Sign flag (positive/negative) - currently unused
    bool sign;

    /**
     * @brief Helper method for initialization (used by constructors)
     * @param n Number of digits
     * @param preci Precision value
     * @deprecated This method is no longer used due to constructor delegation
     */
    void initialize(int n, int preci);
};

// ============================================================================
// Free Functions (Non-member functions)
// ============================================================================

/**
 * @brief Encode a real number as binary
 * @param x The real number to encode
 * @param minrange Minimum value of the range
 * @param maxrange Maximum value of the range
 * @param precision Decimal precision (number of decimal places)
 * @return Binary representation of x
 *
 * Encoding process:
 * 1. Calculate required bits: log2((maxrange-minrange)*10^precision) + 1
 * 2. Convert (x-minrange)*10^precision to integer
 * 3. Convert integer to binary representation
 *
 * Example: code(5.67, 0.0, 10.0, 2)
 *   -> (5.67-0.0)*100 = 567
 *   -> 567 in binary = [1,0,0,0,1,1,0,1,1,1]
 */
CBinary code(double x, double minrange, double maxrange, int precision);

/**
 * @brief Single-point crossover operator
 * @param B1 First parent (modified in-place to become first offspring)
 * @param B2 Second parent (modified in-place to become second offspring)
 * @param p Crossover point (position where chromosomes are split)
 *
 * Swaps bits after position p between B1 and B2
 * Example: B1=[1,1,1,1], B2=[0,0,0,0], p=2
 *   Result: B1=[1,1,0,0], B2=[0,0,1,1]
 */
void cross(CBinary &B1, CBinary &B2, int p);

/**
 * @brief Multi-point crossover operator
 * @param B1 First parent (modified in-place)
 * @param B2 Second parent (modified in-place)
 * @param p Vector of crossover points
 *
 * Alternates swapping between segments defined by the crossover points
 */
void cross(CBinary &B1, CBinary &B2, vector<int> p);

/**
 * @brief Two-point crossover operator
 * @param B1 First parent (modified in-place)
 * @param B2 Second parent (modified in-place)
 * @param p1 First crossover point
 * @param p2 Second crossover point
 *
 * Swaps the segment between p1 and p2 (exclusive p2)
 * Example: B1=[1,1,1,1,1,1], B2=[0,0,0,0,0,0], p1=2, p2=4
 *   Result: B1=[1,1,0,0,1,1], B2=[0,0,1,1,0,0]
 */
void cross2p(CBinary &B1, CBinary &B2, int p1, int p2);

#endif // BINARY_H

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
        std::cout << Digit[i];
    std::cout << endl;
}

// Mutation
void CBinary::mutate(double mu)
{
    for (int i = 0; i < nDigits; i++)
    {
        if (uniformDistribution(randomGenerator) < mu)
            Digit[i] = !Digit[i];
    }
}
