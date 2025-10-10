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

#ifndef GA_H
#define GA_H

#include "Individual.h"
#include "Distribution.h"
#include <stdio.h>
#include "TimeSeries.h"
#include "DistributionNUnif.h"
#include <cmath>
#include <iostream>
#include "Matrix.h"
#ifndef mac_version
#include "omp.h"
#endif
#include <vector>
#include <string>

// Forward declaration for GUI support
class RunTimeWindow;

/**
 * @struct GAParameters
 * @brief Configuration parameters for the genetic algorithm
 *
 * This structure contains all tunable parameters that control
 * the behavior of the genetic algorithm.
 */
struct GAParameters
{
    // Population parameters
    int maxpop = 100;           ///< Population size
    int nParam = 0;             ///< Number of parameters to optimize
    int nGen = 100;             ///< Number of generations

    // Genetic operator probabilities
    double pcross = 1.0;        ///< Crossover probability (0.0 to 1.0)
    double pmute = 0.02;        ///< Mutation probability (0.0 to 1.0)

    // Crossover configuration
    int cross_over_type = 1;    ///< 1=multi-point, 2=two-point
    bool RCGA = false;          ///< Use real-coded GA (linear crossover)

    // Shake operator parameters
    double shakescale = 0.05;       ///< Initial shake scale (±5% perturbation)
    double shakescalered = 0.75;    ///< Shake scale reduction factor

    // Fitness scaling
    double N = 1.0;             ///< Rank-based fitness exponent

    // Enhancement/restart parameters
    int numenhancements = 0;    ///< Number of enhancements to perform
    int num_enh = 0;            ///< Enhancement counter

    // Unused/legacy parameters
    int totnumparams = 0;       ///< Total number of parameters (legacy)
    int no_bins = 0;            ///< Number of bins (purpose unclear)
    double exponentcoeff = 1.0; ///< Exponent coefficient (unused?)
    char fitnesstype = 'R';     ///< Fitness type ('R'=rank-based)
    bool sens_out = false;      ///< Output sensitivity (unused?)
    bool readfromgafile = false;///< Read from GA file (unused?)
};

/**
 * @struct GAFilenames
 * @brief File paths used by the genetic algorithm
 */
struct GAFilenames
{
    std::string initialpopfilename; ///< Initial population file
    std::string pathname;           ///< Output directory path
    std::string getfromfilename;    ///< Input results file
    std::string outputfilename;     ///< Main output file
};

/**
 * @class CGA
 * @brief Template-based Genetic Algorithm optimizer
 * @tparam T Model type that provides objective function evaluation
 *
 * CGA implements a genetic algorithm for parameter optimization. It supports:
 * - Binary-encoded and real-coded crossover
 * - Rank-based fitness selection
 * - Parallel fitness evaluation using OpenMP
 * - Adaptive shake operator
 * - Initial population loading
 * - Progress tracking and detailed logging
 *
 * The template parameter T must provide:
 * - Parameters() method returning parameter list
 * - SetParameterValue() method
 * - ApplyParameters() method
 * - Solve() method
 * - GetObjectiveFunctionValue() method
 */
template<class T>
class CGA
{
public:
    // ========================================================================
    // Constructors and Destructor
    // ========================================================================

    /**
     * @brief Default constructor - creates GA with default settings
     */
    CGA();

    /**
     * @brief Construct GA from configuration file and model
     * @param filename Path to GA configuration file
     * @param model Reference to model instance
     */
    CGA(const std::string& filename, const T& model);

    /**
     * @brief Construct GA from model pointer
     * @param model Pointer to model instance
     */
    explicit CGA(T* model);

    /**
     * @brief Copy constructor
     * @param C GA instance to copy from
     */
    CGA(const CGA<T>& C);

    /**
     * @brief Assignment operator
     * @param C GA instance to assign from
     * @return Reference to this instance
     */
    CGA<T>& operator=(const CGA<T>& C);

    /**
     * @brief Destructor
     */
    virtual ~CGA();

    // ========================================================================
    // Main API Methods
    // ========================================================================

    /**
     * @brief Initialize the population
     *
     * Randomly initializes all individuals within parameter bounds.
     * If an initial population file is specified, loads from that file.
     */
    void initialize();

    /**
     * @brief Run the genetic algorithm optimization
     * @return Index of best individual in final population
     *
     * Performs the complete GA optimization:
     * 1. Initialize population
     * 2. For each generation:
     *    - Evaluate fitness
     *    - Select parents
     *    - Apply crossover
     *    - Apply mutation
     *    - Apply shake
     * 3. Return best solution
     */
    int optimize();

    /**
     * @brief Set a GA parameter by name
     * @param varname Parameter name (case-insensitive)
     * @param value Parameter value as string
     * @return true if parameter was set, false if not found
     *
     * Supported parameters:
     * - maxpop: Population size
     * - ngen: Number of generations
     * - pcross: Crossover probability
     * - pmute: Mutation probability
     * - shakescale: Shake operator scale
     * - shakescalered: Shake scale reduction factor
     * - outputfile: Output filename
     * - initial_population: Initial population filename
     * - numthreads: Number of parallel threads
     */
    bool SetProperty(const std::string& varname, const std::string& value);

    /**
     * @brief Load final parameters from previous GA run
     * @param filename Path to GA output file
     * @return Fitness value of loaded parameters
     */
    double getfromoutput(const std::string& filename);

    /**
     * @brief Load initial population from file
     * @param filename Path to population file
     */
    void getinitialpop(const std::string& filename);

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * @brief Get the last error message
     * @return Error message string
     */
    std::string getLastError() const { return last_error; }

    /**
     * @brief Get final optimized parameters
     * @return Vector of parameter values
     */
    const std::vector<double>& getFinalParams() const { return final_params; }

    /**
     * @brief Get parameter names
     * @return Vector of parameter names
     */
    const std::vector<std::string>& getParamNames() const { return paramname; }

    /**
     * @brief Get best fitness found
     * @return Best fitness value (lower is better)
     */
    double getMaxFitness() const { return MaxFitness; }

    /**
     * @brief Get current population
     * @return Reference to population vector
     */
    const std::vector<CIndividual>& getPopulation() const { return Ind; }

    /**
     * @brief Get number of parameters being optimized
     * @return Parameter count
     */
    int getNumParams() const { return GA_params.nParam; }

    /**
     * @brief Get population size
     * @return Population size
     */
    int getPopulationSize() const { return GA_params.maxpop; }

#ifdef Q_GUI_SUPPORT
    /**
     * @brief Set runtime window for progress visualization
     * @param _rtw Pointer to runtime window
     */
    void SetRunTimeWindow(RunTimeWindow* _rtw) { rtw = _rtw; }
#endif

private:
    // ========================================================================
    // Private Members - Configuration
    // ========================================================================

    GAParameters GA_params;     ///< GA configuration parameters
    GAFilenames filenames;      ///< File paths
    int numberOfThreads;        ///< Number of OpenMP threads
    int current_generation;     ///< Current generation number

    // ========================================================================
    // Private Members - Population and Evolution
    // ========================================================================

    std::vector<CIndividual> Ind;       ///< Current population
    std::vector<CIndividual> Ind_old;   ///< Previous generation (for elitism)
    CDistribution fitdist;              ///< Fitness distribution for selection

    // ========================================================================
    // Private Members - Model and Parameters
    // ========================================================================

    T* Model;                           ///< Pointer to base model
    std::vector<T> Models;              ///< Model copies for parallel evaluation
    T Model_out;                        ///< Best model output

    std::vector<int> params;            ///< Parameter indices
    std::vector<int> loged;             ///< Log-scale flags (1=log, 0=linear)
    std::vector<double> minval;         ///< Minimum parameter values
    std::vector<double> maxval;         ///< Maximum parameter values
    std::vector<std::string> paramname; ///< Parameter names

    // ========================================================================
    // Private Members - Results and State
    // ========================================================================

    std::vector<double> final_params;   ///< Final optimized parameters
    double MaxFitness;                  ///< Best fitness found
    double sumfitness;                  ///< Sum of all fitnesses (unused?)
    std::string last_error;             ///< Last error message

    // Legacy/unused members (kept for backward compatibility)
    std::vector<std::vector<double>> initial_pop;
    std::vector<double> calc_output_percentiles;
    std::vector<int> to_ts;
    std::vector<double> fixedinputvale;
    std::vector<bool> apply_to_all;
    std::vector<std::vector<int>> outcompare;

    /**
     * @brief Initialize GA from model (eliminates constructor duplication)
     * @param model Pointer to model instance
     *
     * Extracts parameters from model, sets up parameter ranges (handling log-scale
     * parameters), creates and initializes population, and sets up fitness distribution.
     * This method is called by both model-based constructors to avoid code duplication.
     *
     * Steps performed:
     * 1. Extract all parameters from model
     * 2. Determine if each parameter uses log or linear scale
     * 3. Set min/max ranges (converting to log10 if needed)
     * 4. Create population with correct size
     * 5. Allocate fit_measures vectors based on observation count
     * 6. Set parameter ranges for all individuals
     * 7. Initialize fitness distribution
     */
        void initFromModel(T* model);

#ifdef Q_GUI_SUPPORT
    RunTimeWindow* rtw;                 ///< Runtime window for progress display
#endif

    // ========================================================================
    // Private Methods - Initialization
    // ========================================================================

    /**
     * @brief Set min/max ranges for a parameter across all individuals
     * @param paramIndex Parameter index
     * @param minrange Minimum value
     * @param maxrange Maximum value
     * @param prec Precision for binary encoding
     */
    void Setminmax(int paramIndex, double minrange, double maxrange, int prec);

    /**
     * @brief Load initial population from output file
     * @param filename Path to output file containing "Final Enhancements"
     */
    void getinifromoutput(const std::string& filename);

    // ========================================================================
    // Private Methods - Fitness Evaluation
    // ========================================================================

    /**
     * @brief Evaluate fitness for entire population (parallelized)
     *
     * Creates model copies, sets parameters, runs simulations in parallel,
     * and assigns fitness values to all individuals.
     */
    void assignfitnesses();

    /**
     * @brief Evaluate fitness for specific parameter set
     * @param inp Parameter values
     * @return Fitness value
     */
    double assignfitnesses(const std::vector<double>& inp);

    /**
     * @brief Assign rank to each individual based on fitness
     *
     * Rank 1 = best fitness, rank 2 = second best, etc.
     */
    void assignrank();

    /**
     * @brief Convert ranks to scaled fitness values
     * @param N Exponent for rank-based scaling
     *
     * fitness[i] = (1 / rank[i])^N
     */
    void assignfitness_rank(double N);

    /**
     * @brief Find index of individual with best fitness
     * @return Index of best individual
     */
    int maxfitness() const;

    // ========================================================================
    // Private Methods - Genetic Operators
    // ========================================================================

    /**
     * @brief Perform crossover operation (binary-encoded)
     *
     * - Preserves best individual (elitism)
     * - Selects parents using fitness-proportionate selection
     * - Creates offspring via crossover
     */
    void crossover();

    /**
     * @brief Perform real-coded crossover (RCGA variant)
     *
     * Uses linear combination instead of binary encoding
     */
    void crossoverRC();

    /**
     * @brief Apply mutation operator to population
     * @param mu Mutation probability per bit
     */
    void mutate(double mu);

    /**
     * @brief Apply shake operator (small random perturbations)
     *
     * Multiplies each parameter by (1 + random(-shakescale, +shakescale))
     */
    void shake();

    // ========================================================================
    // Private Methods - Selection
    // ========================================================================

    /**
     * @brief Initialize fitness distribution for roulette wheel selection
     *
     * Creates cumulative probability distribution based on fitness values
     */
    void fillfitdist();

    // ========================================================================
    // Private Methods - Statistics
    // ========================================================================

    /**
     * @brief Calculate average scaled fitness
     * @return Average fitness
     */
    double avgfitness() const;

    /**
     * @brief Calculate average actual fitness
     * @return Average objective function value
     */
    double avg_actual_fitness() const;

    /**
     * @brief Calculate average inverse actual fitness
     * @return Average of 1/fitness
     */
    double avg_inv_actual_fitness() const;

    /**
     * @brief Calculate variance of scaled fitness
     * @return Fitness variance
     */
    double variancefitness() const;

    /**
     * @brief Calculate standard deviation of fitness
     * @return Fitness standard deviation
     */
    double stdfitness() const;

    // ========================================================================
    // Private Methods - I/O and Logging
    // ========================================================================

    /**
     * @brief Write message to detailed GA log file
     * @param s Message string
     */
    void write_to_detailed_GA(const std::string& message) const;

    // ========================================================================
    // Private Methods - Population Management
    // ========================================================================

    /**
     * @brief Resize population to new size
     * @param n New population size
     */
    void setnumpop(int n);

    /**
     * @brief Set number of parameters
     * @param n_params Number of parameters
     */
    void setnparams(int n_params);

    // ========================================================================
    // Unused/Legacy Methods (kept for compatibility)
    // ========================================================================

    void fitnessdistini();
    void assignfixedvalues();
    void getfromfile(char filename[]);
    int getparamno(int i, int ts);
    int get_act_paramno(int i);
    int get_time_series(int i);
    double evaluateforward();
    double evaluateforward_mixed(std::vector<double> v);
    int optimize(int nGens, char DefOutPutFileName[]);
};

#include "GA.hpp"

#endif // GA_H
