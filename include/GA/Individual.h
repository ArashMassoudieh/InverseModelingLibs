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

#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <vector>

/**
 * @class CIndividual
 * @brief Represents a single individual (candidate solution) in a genetic algorithm
 *
 * An individual contains:
 * - Parameter values (x): The actual solution being optimized
 * - Fitness values: Measures of solution quality
 * - Range constraints: Min/max bounds for each parameter
 * - Precision settings: For binary encoding
 * - Parent tracking: Genealogy information for analysis
 *
 * The individual supports genetic operations like crossover, mutation, and "shake"
 * (small random perturbations).
 */
class CIndividual
{
public:
    // ========================================================================
    // Constructors and Destructor
    // ========================================================================

    /**
     * @brief Default constructor - creates individual with 1 parameter
     */
    CIndividual();

    /**
     * @brief Construct individual with specified number of parameters
     * @param n Number of parameters (decision variables)
     */
    explicit CIndividual(int n);

    /**
     * @brief Copy constructor
     * @param C Individual to copy from
     */
    CIndividual(const CIndividual &C);

    /**
     * @brief Destructor
     */
    virtual ~CIndividual();

    /**
     * @brief Assignment operator with self-assignment safety
     * @param C Individual to assign from
     * @return Reference to this object for chaining
     */
    CIndividual& operator=(const CIndividual &C);

    // ========================================================================
    // Genetic Operations
    // ========================================================================

    /**
     * @brief Initialize parameter values randomly within their ranges
     *
     * For each parameter i: x[i] = random(minrange[i], maxrange[i])
     */
    void initialize();

    /**
     * @brief Apply mutation operator to all parameters
     * @param mu Mutation probability (0.0 to 1.0)
     *
     * Each parameter is encoded as binary, mutated at bit level,
     * then decoded back to real value
     */
    void mutate(double mu);

    /**
     * @brief Apply small random perturbations to parameters ("shake" operator)
     * @param shakescale Maximum relative perturbation (e.g., 0.05 = ±5%)
     *
     * For each parameter: x[i] *= (1 + random(-shakescale, +shakescale))
     * Values are clamped to [minrange, maxrange]
     */
    void shake(double shakescale);

    // ========================================================================
    // Parent Tracking
    // ========================================================================

    /**
     * @brief Set both parents to the same index (cloning)
     * @param i Parent index
     */
    void SetParents(int i);

    /**
     * @brief Set two different parents (crossover)
     * @param i First parent index
     * @param j Second parent index
     */
    void SetParents(int i, int j);

    /**
     * @brief Get parent indices
     * @return Vector containing parent indices
     */
    const std::vector<int>& GetParents() const { return parents; }

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * @brief Get number of parameters
     * @return Number of decision variables
     */
    int getNumParams() const { return nParams; }

    /**
     * @brief Get parameter values (non-const)
     * @return Reference to parameter vector
     */
    std::vector<double>& getX() { return x; }

    /**
     * @brief Get parameter values (const)
     * @return Const reference to parameter vector
     */
    const std::vector<double>& getX() const { return x; }

    /**
     * @brief Get fitness value
     * @return Fitness score (lower is better for minimization)
     */
    double getFitness() const { return fitness; }

    /**
     * @brief Get actual fitness value (objective function value)
     * @return Actual objective function evaluation
     */
    double getActualFitness() const { return actual_fitness; }

    /**
     * @brief Get rank in population
     * @return Rank (1 = best, higher = worse)
     */
    int getRank() const { return rank; }

    /**
     * @brief Get fit measures (e.g., MSE, R², NSE for each observation)
     * @return Vector of fitness measures
     */
    const std::vector<double>& getFitMeasures() const { return fit_measures; }

    // ========================================================================
    // Public Data Members (for direct access by GA - legacy design)
    // ========================================================================
    // Note: These are public for backward compatibility with existing GA code
    // In a future refactor, these should be made private with proper accessors

    /// Parameter values (decision variables)
    std::vector<double> x;

    /// Perturbation values (used in some GA variants)
    std::vector<double> pert;

    /// Direction values (used in some GA variants)
    std::vector<int> dir;

    /// Perturbation effectiveness (used in some GA variants)
    std::vector<double> perteff;

    /// Scaled fitness value (used for selection)
    double fitness;

    /// Actual objective function value
    double actual_fitness;

    /// Secondary fitness value (purpose unclear - possibly unused)
    double actual_fitness2;

    /// Legacy parent tracking (replaced by parents vector)
    int parent1, parent2, xsite;

    /// Rank in population (1 = best)
    int rank;

    /// Number of parameters
    int nParams;

    /// Precision for binary encoding of each parameter
    std::vector<int> precision;

    /// Minimum values for each parameter
    std::vector<double> minrange;

    /// Maximum values for each parameter
    std::vector<double> maxrange;

    /// Detailed fitness measures (MSE, R², NSE per observation)
    std::vector<double> fit_measures;

    /// Parent indices in population
    std::vector<int> parents;
};

// ============================================================================
// Free Functions (Non-member functions)
// ============================================================================

/**
 * @brief Generate random number uniformly distributed in [xmin, xmax]
 * @param xmin Minimum value (inclusive)
 * @param xmax Maximum value (inclusive)
 * @return Random number in specified range
 */
double GetRndUnif(double xmin, double xmax);

/**
 * @brief Multi-point crossover operator for individuals
 * @param I1 First parent (const)
 * @param I2 Second parent (const)
 * @param IR1 First offspring (output)
 * @param IR2 Second offspring (output)
 *
 * Process:
 * 1. Encode all parameters of both parents as concatenated binary strings
 * 2. Generate random crossover points
 * 3. Perform multi-point crossover on binary representations
 * 4. Decode back to parameter values
 * 5. Clamp values to valid ranges
 */
void cross(const CIndividual &I1, const CIndividual &I2,
           CIndividual &IR1, CIndividual &IR2);

/**
 * @brief Two-point crossover operator for individuals
 * @param I1 First parent (const)
 * @param I2 Second parent (const)
 * @param IR1 First offspring (output)
 * @param IR2 Second offspring (output)
 *
 * Similar to multi-point crossover but uses exactly 2 random crossover points
 */
void cross2p(const CIndividual &I1, const CIndividual &I2,
             CIndividual &IR1, CIndividual &IR2);

/**
 * @brief Real-coded linear crossover operator (RCGA)
 * @param I1 First parent (const)
 * @param I2 Second parent (const)
 * @param IR1 First offspring (output)
 * @param IR2 Second offspring (output)
 *
 * For each parameter i and random weight w ∈ [0,1]:
 * - IR1.x[i] = I1.x[i] * w + I2.x[i] * (1-w)
 * - IR2.x[i] = I2.x[i] * w + I1.x[i] * (1-w)
 *
 * This is a real-valued crossover that doesn't use binary encoding
 */
void cross_RC_L(const CIndividual &I1, const CIndividual &I2,
                CIndividual &IR1, CIndividual &IR2);

#endif // INDIVIDUAL_H
