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




// GA.cpp: implementation of the CGA class.
////////////////////////////////////////////////////////////////////////
#include "GA.h"
#include <stdlib.h>
#ifndef mac_version
#include <omp.h>
#endif
#ifdef Q_JSON_SUPPORT
#include "QDebug"
#include <QString>
#endif

#include <iostream>
#include "Utilities.h"
#include <random>
#include "Matrix.h"
#include "Vector.h"
#include "parameter_set.h"
#include "observation.h"



#ifdef Q_GUI_SUPPORT
    #include "runtimewindow.h"
#endif

/**
 * @brief Default constructor
 *
 * Creates a GA with default settings but no model attached.
 * User must provide a model before calling optimize().
 */

template<class T>
CGA<T>::CGA()
    : Model(nullptr)
    , GA_params()
    , filenames()
    , numberOfThreads(16)
    , current_generation(0)
    , MaxFitness(0.0)
    , sumfitness(0.0)
    , randomGenerator(std::random_device{}())  // Seed with random device
    , uniformDistribution(0.0, 1.0)            // Uniform distribution [0, 1]
#ifdef Q_GUI_SUPPORT
    , rtw(nullptr)
#endif
{
    GA_params.maxpop = 100;
    GA_params.N = 1.0;
    GA_params.pcross = 1.0;
    GA_params.cross_over_type = 1;

    Ind.resize(GA_params.maxpop);
    Ind_old.resize(GA_params.maxpop);
    fitdist = CDistribution(GA_params.maxpop);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);
}


/**
 * @brief Private helper method to initialize GA from a model
 * @param model Pointer to model instance
 *
 * Extracts parameters from model, sets up ranges, creates population.
 * This eliminates duplication between the two model-based constructors.
 */
template<class T>
void CGA<T>::initFromModel(T* model)
{
    Model = model;

    // Get output path from model
    filenames.pathname = Model->OutputPath();

    // Initialize parameter count
    GA_params.nParam = 0;

    // Extract parameters from model
    Parameter_Set& parameters = Model->Parameters();
    const int modelParamCount = static_cast<int>(parameters.size());

    for (int i = 0; i < modelParamCount; i++)
    {
        GA_params.nParam++;
        params.push_back(i);

        // Check if parameter uses log-normal distribution
        const std::string priorDist = Model->Parameters()[i]->GetPriorDistribution();
        const bool isLogNormal = (priorDist == "lognormal");

        Parameter::Range range = Model->Parameters()[i]->GetRange();
        if (isLogNormal)
        {
            // Store log10 of bounds for log-normal parameters
            minval.push_back(log10(range.low));
            maxval.push_back(log10(range.high));
        }
        else
        {
            // Store bounds directly for linear parameters
            minval.push_back(range.low);
            maxval.push_back(range.high);
            loged.push_back(0);
        }

        apply_to_all.push_back(false);
        paramname.push_back(Model->Parameters()[i]->GetName());
    }

    // Ensure population size is at least 1
    GA_params.maxpop = std::max(1, GA_params.maxpop);

    // Create and initialize population
    Ind.resize(GA_params.maxpop);
    Ind_old.resize(GA_params.maxpop);

    const int observationCount = static_cast<int>(model->ObservationsCount());
    const int fitMeasuresSize = observationCount * 3; // MSE, R², NSE per observation

    for (int i = 0; i < GA_params.maxpop; i++)
    {
        Ind[i] = CIndividual(GA_params.nParam);
        Ind[i].fit_measures.resize(fitMeasuresSize);

        Ind_old[i] = CIndividual(GA_params.nParam);
        Ind_old[i].fit_measures.resize(fitMeasuresSize);
    }

    // Set parameter ranges for all individuals
    for (int j = 0; j < GA_params.nParam; j++)
    {
        Setminmax(j, minval[j], maxval[j], 4); // Default precision = 4
    }

    // Initialize fitness distribution
    fitdist = CDistribution(GA_params.maxpop);
    GA_params.cross_over_type = 1;
    MaxFitness = 0.0;
}

/**
 * @brief Construct GA from configuration file and model
 * @param filename Path to GA configuration file
 * @param model Reference to model instance (will be copied to internal pointer)
 *
 * Configuration file format (key-value pairs):
 * maxpop <value>
 * ngen <value>
 * pcross <value>
 * pmute <value>
 * shakescale <value>
 * shakescalered <value>
 * outputfile <filename>
 * getfromfilename <filename>
 * initial_population <filename>
 * numthreads <value>
 */
template<class T>
CGA<T>::CGA(const std::string& filename, const T& model)
    : Model(nullptr)
    , Model_out(nullptr)
    , GA_params()
    , filenames()
    , numberOfThreads(20)
    , current_generation(0)
    , MaxFitness(0.0)
    , sumfitness(0.0)
#ifdef Q_GUI_SUPPORT
    , rtw(nullptr)
#endif
{
    // Set default GA parameters
    GA_params.pcross = 1.0;
    GA_params.N = 1.0;
    GA_params.RCGA = false;

    // Read configuration from file
    std::ifstream file(filename);
    if (file.is_open())
    {
        std::vector<std::string> s;
        while (!file.eof())
        {
            s = aquiutils::getline(file);
            if (s.size() > 0)
            {
                const std::string& key = s[0];

                if (key == "maxpop" && s.size() > 1)
                    GA_params.maxpop = aquiutils::atoi(s[1]);
                else if (key == "ngen" && s.size() > 1)
                    GA_params.nGen = aquiutils::atoi(s[1]);
                else if (key == "pcross" && s.size() > 1)
                    GA_params.pcross = aquiutils::atof(s[1]);
                else if (key == "pmute" && s.size() > 1)
                    GA_params.pmute = aquiutils::atof(s[1]);
                else if (key == "shakescale" && s.size() > 1)
                    GA_params.shakescale = aquiutils::atof(s[1]);
                else if (key == "shakescalered" && s.size() > 1)
                    GA_params.shakescalered = aquiutils::atof(s[1]);
                else if (key == "outputfile" && s.size() > 1)
                    filenames.outputfilename = s[1];
                else if (key == "getfromfilename" && s.size() > 1)
                    filenames.getfromfilename = s[1];
                else if (key == "initial_population" && s.size() > 1)
                    filenames.initialpopfilename = s[1];
                else if (key == "numthreads" && s.size() > 1)
                    numberOfThreads = aquiutils::atoi(s[1]);
            }
        }
        file.close();
    }
    else
    {
        // File couldn't be opened - log warning but continue with defaults
        last_error = "Warning: Could not open configuration file '" + filename +
                     "'. Using default parameters.";
    }

    // Initialize from model (common code)
    T* modelPtr = const_cast<T*>(&model); // Need non-const pointer
    initFromModel(modelPtr);
}


/**
 * @brief Construct GA from model pointer
 * @param model Pointer to model instance
 *
 * Uses default GA parameters. User can modify them using SetProperty().
 */
template<class T>
CGA<T>::CGA(T* model)
    : Model(nullptr)
    , Model_out(nullptr)
    , GA_params()
    , filenames()
    , numberOfThreads(20)
    , current_generation(0)
    , MaxFitness(0.0)
    , sumfitness(0.0)
    , randomGenerator(std::random_device{}())  // Seed with random device
    , uniformDistribution(0.0, 1.0)            // Uniform distribution [0, 1]
#ifdef Q_GUI_SUPPORT
    , rtw(nullptr)
#endif
{
    // Set default GA parameters
    GA_params.pcross = 1.0;
    GA_params.N = 1.0;
    GA_params.RCGA = false;

    // Initialize from model (common code)
    initFromModel(model);
}

//////////////////////////////////////////////////////////////////////
// Helper Methods for Population Management
//////////////////////////////////////////////////////////////////////

/**
 * @brief Set number of parameters
 * @param n_params Number of parameters
 *
 * Resizes individuals in population. Should be called before optimization.
 */
template<class T>
void CGA<T>::setnparams(int n_params)
{
    Ind.resize(GA_params.maxpop);
    Ind_old.resize(GA_params.maxpop);

    const int fitMeasuresSize = static_cast<int>(Model->ObservationsCount() * 3);

    for (int i = 0; i < GA_params.maxpop; i++)
    {
        Ind[i] = CIndividual(n_params);
        Ind[i].fit_measures.resize(fitMeasuresSize);

        Ind_old[i] = CIndividual(n_params);
        Ind_old[i].fit_measures.resize(fitMeasuresSize);
    }
}

/**
 * @brief Resize population to new size
 * @param n New population size
 *
 * Preserves parameter ranges from first individual.
 * Creates new individuals with same configuration.
 */
template<class T>
void CGA<T>::setnumpop(int n)
{
    GA_params.maxpop = n;

    // Save configuration from first individual
    CIndividual TempInd = Ind[0];
    const int nParam = Ind[0].getNumParams();

    // Resize population vectors
    Ind.resize(GA_params.maxpop);
    Ind_old.resize(GA_params.maxpop);

    const int fitMeasuresSize = static_cast<int>(Model->ObservationsCount() * 3);

    // Create new individuals with saved configuration
    for (int i = 0; i < n; i++)
    {
        Ind[i] = CIndividual(GA_params.nParam);
        Ind_old[i] = CIndividual(GA_params.nParam);

        // Copy parameter ranges from template individual
        for (int j = 0; j < nParam; j++)
        {
            Ind[i].minrange[j] = TempInd.minrange[j];
            Ind[i].maxrange[j] = TempInd.maxrange[j];
            Ind[i].precision[j] = TempInd.precision[j];

            Ind_old[i].minrange[j] = TempInd.minrange[j];
            Ind_old[i].maxrange[j] = TempInd.maxrange[j];
            Ind_old[i].precision[j] = TempInd.precision[j];
        }

        Ind[i].fit_measures.resize(fitMeasuresSize);
        Ind_old[i].fit_measures.resize(fitMeasuresSize);
    }

    // Resize fitness distribution
    fitdist = CDistribution(GA_params.maxpop);
}
/**
 * @brief Copy constructor
 * @param C GA instance to copy from
 *
 * Note: Does not copy Model pointer, Models vector, or GUI window pointer
 */
template<class T>
CGA<T>::CGA(const CGA<T>& C)
    : Model(C.Model)
    , Model_out(nullptr)
    , GA_params(C.GA_params)
    , filenames(C.filenames)
    , numberOfThreads(C.numberOfThreads)
    , current_generation(C.current_generation)
    , Ind(C.Ind)
    , Ind_old(C.Ind_old)
    , fitdist(C.fitdist)
    , params(C.params)
    , loged(C.loged)
    , minval(C.minval)
    , maxval(C.maxval)
    , paramname(C.paramname)
    , final_params(C.final_params)
    , MaxFitness(C.MaxFitness)
    , sumfitness(C.sumfitness)
    , last_error(C.last_error)
    , initial_pop(C.initial_pop)
    , calc_output_percentiles(C.calc_output_percentiles)
    , to_ts(C.to_ts)
    , fixedinputvale(C.fixedinputvale)
    , apply_to_all(C.apply_to_all)
    , outcompare(C.outcompare)
#ifdef Q_GUI_SUPPORT
    , rtw(nullptr) // Don't copy GUI pointer
#endif
{
    // Models vector is not copied (would be expensive and usually not needed)
    // Model_out is not copied (will be regenerated during optimization)
}


/**
 * @brief Assignment operator
 * @param C GA instance to assign from
 * @return Reference to this instance
 *
 * Note: Does not copy Model pointer, Models vector, or GUI window pointer
 */
template<class T>
CGA<T>& CGA<T>::operator=(const CGA<T>& C)
{
    // Check for self-assignment
    if (this == &C)
        return *this;

    // Copy configuration and parameters
    GA_params = C.GA_params;
    filenames = C.filenames;
    numberOfThreads = C.numberOfThreads;
    current_generation = C.current_generation;

    // Copy population
    Ind = C.Ind;
    Ind_old = C.Ind_old;
    fitdist = C.fitdist;

    // Copy parameter information
    params = C.params;
    loged = C.loged;
    minval = C.minval;
    maxval = C.maxval;
    paramname = C.paramname;

    // Copy results
    final_params = C.final_params;
    MaxFitness = C.MaxFitness;
    sumfitness = C.sumfitness;
    last_error = C.last_error;

    // Copy legacy members
    initial_pop = C.initial_pop;
    calc_output_percentiles = C.calc_output_percentiles;
    to_ts = C.to_ts;
    fixedinputvale = C.fixedinputvale;
    apply_to_all = C.apply_to_all;
    outcompare = C.outcompare;

    // Model pointer is copied but Models vector is not
    Model = C.Model;

    // GUI pointer is not copied

    return *this;
}

/**
 * @brief Destructor
 *
 * Cleanup is mostly automatic due to RAII with std::vector.
 * Models vector will be automatically cleared.
 */
template<class T>
CGA<T>::~CGA()
{
    // Nothing to explicitly delete - vectors handle their own cleanup
}

/**
 * @brief Initialize the population
 *
 * Randomly initializes all individuals within their parameter bounds.
 * If an initial population file is specified, loads those values instead.
 *
 * Process:
 * 1. Call initialize() on each individual (random values within ranges)
 * 2. If initial population file exists, load and override random values
 * 3. Handle log-scale parameters appropriately
 *
 * @throws std::runtime_error if initial population file format is invalid
 */
template<class T>
void CGA<T>::initialize()
{
    // Step 1: Initialize all individuals with random values
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        Ind[i].initialize();
    }

    // Step 2: Load initial population from file if specified
    if (!filenames.initialpopfilename.empty())
    {
        const std::string fullPath = filenames.pathname + filenames.initialpopfilename;
        getinifromoutput(fullPath);

        // Override random values with loaded initial population
        const size_t popSize = initial_pop.size();
        const size_t numParams = static_cast<size_t>(GA_params.nParam);

        for (size_t i = 0; i < popSize && i < static_cast<size_t>(GA_params.maxpop); i++)
        {
            // Ensure the loaded individual has enough parameters
            const size_t availableParams = initial_pop[i].size();
            const size_t paramsToLoad = std::min(availableParams, numParams);

            for (size_t j = 0; j < paramsToLoad; j++)
            {
                if (loged[j] == 1)
                {
                    // Parameter uses log scale - convert to log10
                    Ind[i].x[j] = log10(initial_pop[i][j]);
                }
                else
                {
                    // Parameter uses linear scale - use directly
                    Ind[i].x[j] = initial_pop[i][j];
                }
            }
        }

        // Log how many individuals were initialized from file
        if (popSize < static_cast<size_t>(GA_params.maxpop))
        {
            write_to_detailed_GA("Loaded " + std::to_string(popSize) +
                                 " individuals from initial population file. " +
                                 "Remaining " + std::to_string(GA_params.maxpop - popSize) +
                                 " individuals are randomly initialized.");
        }
        else
        {
            write_to_detailed_GA("Loaded initial population from file: " + fullPath);
        }
    }
    else
    {
        write_to_detailed_GA("Population randomly initialized.");
    }
}

template<class T>
void CGA<T>::Setminmax(int paramIndex, double minrange, double maxrange, int prec)
{
    // Validate parameter index
    if (paramIndex < 0 || paramIndex >= GA_params.nParam)
    {
        last_error = "Invalid parameter index " + std::to_string(paramIndex) +
                     " in Setminmax(). Valid range is [0, " +
                     std::to_string(GA_params.nParam - 1) + "]";
        throw std::out_of_range(last_error);
    }

    // Validate range
    if (minrange >= maxrange)
    {
        last_error = "Invalid parameter range in Setminmax(): minrange (" +
                     std::to_string(minrange) + ") must be less than maxrange (" +
                     std::to_string(maxrange) + ")";
        throw std::invalid_argument(last_error);
    }

    // Validate precision
    if (prec < 0 || prec > 10)
    {
        last_error = "Warning: Unusual precision value " + std::to_string(prec) +
                     " in Setminmax(). Typical range is [0, 10]";
        // Don't throw - just warn, as this might be intentional
        write_to_detailed_GA(last_error);
    }

    // Set parameter configuration for all individuals in population
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        Ind[i].maxrange[paramIndex] = maxrange;
        Ind[i].minrange[paramIndex] = minrange;
        Ind[i].precision[paramIndex] = prec;
    }
}

template<class T>
void CGA<T>::assignfitnesses()
{
    sumfitness = 0.0;

    // ========================================================================
    // Step 1: Prepare parameter sets for all individuals
    // ========================================================================

    std::vector<std::vector<double>> parameterSets(GA_params.maxpop);
    for (int k = 0; k < GA_params.maxpop; k++)
    {
        parameterSets[k].resize(GA_params.nParam);
    }

    // Transform parameters (handle log-scale)
    for (int k = 0; k < GA_params.maxpop; k++)
    {
        for (int i = 0; i < GA_params.nParam; i++)
        {
            if (loged[i] == 1)
            {
                // Parameter uses log scale - transform from log10
                parameterSets[k][i] = pow(10.0, Ind[k].x[i]);
            }
            else
            {
                // Parameter uses linear scale - use directly
                parameterSets[k][i] = Ind[k].x[i];
            }
        }

        // Initialize fitness to zero
        Ind[k].actual_fitness = 0.0;
    }

    // ========================================================================
    // Step 2: Create model copies and apply parameters
    // ========================================================================

    write_to_detailed_GA("Creating model copies and applying parameters...");

    Models.clear();
    Models.reserve(GA_params.maxpop);  // Reserve space but don't construct

    for (int k = 0; k < GA_params.maxpop; k++)
    {
        // Create model copy by copy construction
        Models.push_back(*Model);

        // Configure the model
        //Models[k].SetSilent(true);
        //Models[k].SetRecordResults(false);
        //Models[k].SetNumThreads(1);

        // Apply parameters to model
        for (int i = 0; i < GA_params.nParam; i++)
        {
            Models[k].SetParameterValue(i, parameterSets[k][i]);
        }

    }

    // ========================================================================
    // Step 3: Parallel fitness evaluation
    // ========================================================================

    write_to_detailed_GA("Evaluating fitness in parallel...");

    std::vector<double> evaluationTimes(GA_params.maxpop, 0.0);
    std::vector<int> epochs(GA_params.maxpop, 0);
    int completedEvaluations = 0;

#ifndef NO_OPENMP
    omp_set_num_threads(numberOfThreads);
#endif

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < GA_params.maxpop; k++)
    {
        // Log parameters being evaluated (thread-safe)
#ifndef NO_OPENMP
#pragma omp critical
#endif
        {
            FILE* FileOut = fopen((filenames.pathname + "detail_GA.txt").c_str(), "a");
            if (FileOut)
            {
                fprintf(FileOut, "%i, ", k);
                for (int l = 0; l < Ind[0].getNumParams(); l++)
                {
                    if (loged[l] == 1)
                        fprintf(FileOut, "%le, ", pow(10.0, Ind[k].x[l]));
                    else
                        fprintf(FileOut, "%le, ", Ind[k].x[l]);
                }
                fprintf(FileOut, "\n");
                fclose(FileOut);
            }
        }

        // Time the evaluation
        time_t startTime = time(nullptr);

#ifdef Debug_GA
        // Save model configuration for debugging
        Models[k].SavetoScriptFile(
            filenames.pathname + "/temp/model_" +
                aquiutils::numbertostring(k) + "_" +
                aquiutils::numbertostring(current_generation) + ".ohq",
            std::string(""),
            std::vector<std::string>()
            );
#endif

        // Get fitness value
        Ind[k].actual_fitness = -Models[k].GetObjectiveFunctionValue();

        // Handle failed simulations
        if (Models[k].GetSolutionFailed())
        {
            // Use average of parent fitness as penalty
            if (Ind[k].parents.size() == 2 &&
                Ind[k].parents[0] < static_cast<int>(Ind_old.size()) &&
                Ind[k].parents[1] < static_cast<int>(Ind_old.size()))
            {
                const double parent1Fitness = Ind_old[Ind[k].parents[0]].actual_fitness;
                const double parent2Fitness = Ind_old[Ind[k].parents[1]].actual_fitness;
                Ind[k].actual_fitness = (parent1Fitness + parent2Fitness) / 2.0;

#ifndef NO_OPENMP
#pragma omp critical
#endif
                {
                    FILE* FileOut = fopen((filenames.pathname + "detail_GA.txt").c_str(), "a");
                    if (FileOut)
                    {
                        fprintf(FileOut,
                                "Simulation failed: %i, parent1=%i, parent2=%i, "
                                "parent1_fitness=%e, parent2_fitness=%e, fitness=%e\n",
                                k, Ind[k].parents[0], Ind[k].parents[1],
                                parent1Fitness, parent2Fitness, Ind[k].actual_fitness);
                        fclose(FileOut);
                    }
                }
            }
            else
            {
#ifndef NO_OPENMP
#pragma omp critical
#endif
                {
                    FILE* FileOut = fopen((filenames.pathname + "detail_GA.txt").c_str(), "a");
                    if (FileOut)
                    {
                        fprintf(FileOut,
                                "Simulation failed: %i, invalid parent info (size=%zu)\n",
                                k, Ind[k].parents.size());
                        fclose(FileOut);
                    }
                }
            }
        }

        // Warn about zero fitness (potential issue)
        if (Ind[k].actual_fitness == 0.0)
        {
#ifndef NO_OPENMP
#pragma omp critical
#endif
            {
                std::cout << "Warning: Individual " << k << " has zero fitness!" << std::endl;
            }
        }

        // Copy detailed fitness measures
        if (Ind[k].fit_measures.size() >= 3)  // Ensure space for MSE, R2, NSE
        {
            // Calculate fit measures for each observation
            for (size_t obsIdx = 0; obsIdx < Models[k].ObservationsCount(); ++obsIdx)
            {
                Observation* obs = Models[k].observation(static_cast<int>(obsIdx));
                if (obs)
                {
                    size_t baseIdx = obsIdx * 3;
                    if (baseIdx + 2 < Ind[k].fit_measures.size())
                    {
                        Ind[k].fit_measures[baseIdx]     = obs->CalculateRMSE();  // MSE
                        Ind[k].fit_measures[baseIdx + 1] = obs->CalculateR2();   // R²
                        Ind[k].fit_measures[baseIdx + 2] = obs->CalculateNSE();  // NSE
                    }
                }
            }
        }

#ifdef Debug_GA
        // Save outputs for debugging
        Models[k].GetModeledObjectiveFunctions().writetofile(
            filenames.pathname + "/temp/observedoutputs_" +
            aquiutils::numbertostring(k) + "_" +
            aquiutils::numbertostring(current_generation) + ".txt"
            );
        Models[k].GetOutputs().writetofile(
            filenames.pathname + "/temp/outputs_" +
            aquiutils::numbertostring(k) + "_" +
            aquiutils::numbertostring(current_generation) + ".txt"
            );
#endif

        evaluationTimes[k] = static_cast<double>(time(nullptr) - startTime);

        // Update progress (thread-safe)
#ifndef NO_OPENMP
#pragma omp critical
#endif
        {
            completedEvaluations++;

#ifdef Q_GUI_SUPPORT
            // Update GUI progress bar
            if (rtw != nullptr)
            {
#ifndef NO_OPENMP
                if (omp_get_thread_num() == 0)
#endif
                {
                    rtw->SetProgress2(static_cast<double>(completedEvaluations) /
                                      static_cast<double>(GA_params.maxpop));
                    QCoreApplication::processEvents();
                }
            }
#endif

            // Log completion
            FILE* FileOut = fopen((filenames.pathname + "detail_GA.txt").c_str(), "a");
            if (FileOut)
            {
                fprintf(FileOut,
                        "%i, fitness=%e, time=%e, internal_time=%e, failed=%i\n",
                        k, Ind[k].actual_fitness, evaluationTimes[k],
                        static_cast<double>(Models[k].GetSimulationDuration()),
                        Models[k].GetSolutionFailed());
                fclose(FileOut);
            }
        }
    }

    // ========================================================================
    // Step 4: Post-processing
    // ========================================================================

    // Store best model for output
    const int bestIndex = maxfitness();
    if (Model_out != nullptr)
    {
        delete Model_out;
    }
    Model_out = new T(Models[bestIndex]);

#ifdef Q_GUI_SUPPORT
    if (rtw != nullptr)
    {
        rtw->SetProgress2(1.0);
        QCoreApplication::processEvents();
    }
#endif

    // Clear parameter sets (no longer needed)
    parameterSets.clear();

    // Assign rank-based fitness values
    assignfitness_rank(GA_params.N);

    write_to_detailed_GA("Fitness evaluation complete.");
}

/**
 * @brief Perform crossover operation (binary-encoded)
 *
 * Creates next generation through crossover of selected parents.
 * Uses fitness-proportionate selection (roulette wheel) to choose parents.
 *
 * Process:
 * 1. Preserve best individual (elitism)
 * 2. For remaining population:
 *    - Select two parents based on fitness
 *    - Apply crossover with probability pcross
 *    - Create two offspring
 * 3. Track parent indices for analysis
 *
 * Crossover types:
 * - Type 1: Multi-point crossover (random number of points)
 * - Type 2: Two-point crossover (exactly 2 points)
 *
 * @note Uses binary encoding for parameters
 * @see crossoverRC() for real-coded alternative
 */
template<class T>
void CGA<T>::crossover()
{
    // ========================================================================
    // Step 1: Save current population and implement elitism
    // ========================================================================

    Ind_old = Ind;
    const int bestIndex = maxfitness();

    // Elitism: Copy best individual to first two positions
    Ind[0] = Ind_old[bestIndex];
    Ind[1] = Ind_old[bestIndex];
    Ind[0].SetParents(bestIndex);
    Ind[1].SetParents(bestIndex);

    // ========================================================================
    // Step 2: Generate offspring through crossover
    // ========================================================================

    for (int i = 2; i < GA_params.maxpop; i += 2)
    {
        // Select two parents using fitness-proportionate selection
        const int parent1Index = fitdist.GetRand();
        const int parent2Index = fitdist.GetRand();

        // Determine offspring index (handle odd population size)
        const int offspring1Index = i;
        const int offspring2Index = std::min(i + 1, GA_params.maxpop - 1);

        // Apply crossover with probability pcross
        const double randomValue = uniformDistribution(randomGenerator);

        if (randomValue < GA_params.pcross)
        {
            // Perform crossover
            if (GA_params.cross_over_type == 1)
            {
                // Multi-point crossover
                cross(Ind_old[parent1Index], Ind_old[parent2Index],
                      Ind[offspring1Index], Ind[offspring2Index]);
            }
            else
            {
                // Two-point crossover
                cross2p(Ind_old[parent1Index], Ind_old[parent2Index],
                        Ind[offspring1Index], Ind[offspring2Index]);
            }
        }
        else
        {
            // No crossover - copy parents directly
            Ind[offspring1Index] = Ind_old[parent1Index];
            Ind[offspring2Index] = Ind_old[parent2Index];
        }

        // Track parent indices for genealogy
        Ind[offspring1Index].SetParents(parent1Index, parent2Index);
        Ind[offspring2Index].SetParents(parent1Index, parent2Index);
    }
}

/**
 * @brief Perform real-coded crossover (RCGA variant)
 *
 * Uses linear combination for crossover instead of binary encoding.
 * This is typically more efficient for continuous optimization problems.
 *
 * Process:
 * 1. Preserve best individual (elitism)
 * 2. For remaining population:
 *    - Select two parents based on fitness
 *    - Apply linear crossover with probability pcross
 *    - Create two offspring:
 *      child1 = p1 × w + p2 × (1-w)
 *      child2 = p2 × w + p1 × (1-w)
 *    where w is random weight ∈ [0,1]
 *
 * Advantages over binary crossover:
 * - No encoding/decoding overhead
 * - Smoother parameter space exploration
 * - Better for real-valued parameters
 *
 * @note Parameters stay in continuous space (no binary conversion)
 * @see crossover() for binary-encoded alternative
 */
template<class T>
void CGA<T>::crossoverRC()
{
    // ========================================================================
    // Step 1: Save current population and implement elitism
    // ========================================================================

    Ind_old = Ind;
    const int bestIndex = maxfitness();

    // Elitism: Copy best individual to first two positions
    Ind[0] = Ind_old[bestIndex];
    Ind[1] = Ind_old[bestIndex];
    Ind[0].SetParents(bestIndex);
    Ind[1].SetParents(bestIndex);

    // ========================================================================
    // Step 2: Generate offspring through real-coded crossover
    // ========================================================================

    for (int i = 2; i < GA_params.maxpop; i += 2)
    {
        // Select two parents using fitness-proportionate selection
        const int parent1Index = fitdist.GetRand();
        const int parent2Index = fitdist.GetRand();

        // Determine offspring indices (handle odd population size)
        const int offspring1Index = i;
        const int offspring2Index = std::min(i + 1, GA_params.maxpop - 1);

        // Track parent indices for genealogy
        Ind[offspring1Index].SetParents(parent1Index, parent2Index);
        Ind[offspring2Index].SetParents(parent1Index, parent2Index);

        // Apply crossover with probability pcross
        const double randomValue = uniformDistribution(randomGenerator);

        if (randomValue < GA_params.pcross)
        {
            // Perform linear crossover
            cross_RC_L(Ind_old[parent1Index], Ind_old[parent2Index],
                       Ind[offspring1Index], Ind[offspring2Index]);
        }
        else
        {
            // No crossover - copy parents directly
            Ind[offspring1Index] = Ind_old[parent1Index];
            Ind[offspring2Index] = Ind_old[parent2Index];
        }
    }
}
/**
 * @brief Set a GA configuration parameter by name
 * @param varname Parameter name (case-insensitive)
 * @param value Parameter value as string
 * @return true if parameter was found and set, false otherwise
 *
 * Supported parameters:
 * - maxpop: Population size (positive integer)
 * - ngen: Number of generations (positive integer)
 * - pcross: Crossover probability [0.0, 1.0]
 * - pmute: Mutation probability [0.0, 1.0]
 * - shakescale: Shake operator scale (positive double)
 * - shakescalered: Shake scale reduction factor (0.0, 1.0)
 * - outputfile: Output filename (string)
 * - getfromfilename: Input results filename (string)
 * - initial_population: Initial population filename (string)
 * - numthreads: Number of parallel threads (positive integer)
 *
 * If parameter is not found, sets last_error message.
 *
 * @note Parameter names are case-insensitive
 * @note For maxpop, automatically resizes population
 *
 * Example:
 * @code
 * if (!ga.SetProperty("maxpop", "100"))
 *     std::cerr << ga.getLastError() << std::endl;
 * @endcode
 */
template<class T>
bool CGA<T>::SetProperty(const std::string& varname, const std::string& value)
{
    // Convert parameter name to lowercase for case-insensitive comparison
    const std::string lowerVarname = aquiutils::tolower(varname);

    // ========================================================================
    // Population Parameters
    // ========================================================================

    if (lowerVarname == "maxpop")
    {
        const int newPopSize = aquiutils::atoi(value);

        // Validate population size
        if (newPopSize <= 0)
        {
            last_error = "Invalid maxpop value '" + value +
                         "'. Population size must be positive.";
            return false;
        }

        GA_params.maxpop = newPopSize;
        setnumpop(GA_params.maxpop);
        return true;
    }

    if (lowerVarname == "ngen")
    {
        const int numGenerations = aquiutils::atoi(value);

        // Validate number of generations
        if (numGenerations <= 0)
        {
            last_error = "Invalid ngen value '" + value +
                         "'. Number of generations must be positive.";
            return false;
        }

        GA_params.nGen = numGenerations;
        return true;
    }

    // ========================================================================
    // Genetic Operator Probabilities
    // ========================================================================

    if (lowerVarname == "pcross")
    {
        const double crossoverProb = aquiutils::atof(value);

        // Validate crossover probability
        if (crossoverProb < 0.0 || crossoverProb > 1.0)
        {
            last_error = "Invalid pcross value '" + value +
                         "'. Crossover probability must be in [0.0, 1.0].";
            return false;
        }

        GA_params.pcross = crossoverProb;
        return true;
    }

    if (lowerVarname == "pmute")
    {
        const double mutationProb = aquiutils::atof(value);

        // Validate mutation probability
        if (mutationProb < 0.0 || mutationProb > 1.0)
        {
            last_error = "Invalid pmute value '" + value +
                         "'. Mutation probability must be in [0.0, 1.0].";
            return false;
        }

        GA_params.pmute = mutationProb;
        return true;
    }

    // ========================================================================
    // Shake Operator Parameters
    // ========================================================================

    if (lowerVarname == "shakescale")
    {
        const double shakeScale = aquiutils::atof(value);

        // Validate shake scale
        if (shakeScale < 0.0)
        {
            last_error = "Invalid shakescale value '" + value +
                         "'. Shake scale must be non-negative.";
            return false;
        }

        // Warn about unusual values
        if (shakeScale > 1.0)
        {
            last_error = "Warning: Large shakescale value '" + value +
                         "'. Typical range is [0.0, 0.2]. Value accepted but may cause instability.";
            write_to_detailed_GA(last_error);
            // Don't return false - just warn
        }

        GA_params.shakescale = shakeScale;
        return true;
    }

    if (lowerVarname == "shakescalered")
    {
        const double shakeReduction = aquiutils::atof(value);

        // Validate shake scale reduction
        if (shakeReduction <= 0.0 || shakeReduction >= 1.0)
        {
            last_error = "Invalid shakescalered value '" + value +
                         "'. Reduction factor must be in (0.0, 1.0).";
            return false;
        }

        GA_params.shakescalered = shakeReduction;
        return true;
    }

    // ========================================================================
    // File Path Parameters
    // ========================================================================

    if (lowerVarname == "outputfile")
    {
        // Validate that value is not empty
        if (value.empty())
        {
            last_error = "Invalid outputfile value. Filename cannot be empty.";
            return false;
        }

        filenames.outputfilename = value;
        return true;
    }

    if (lowerVarname == "getfromfilename")
    {
        // Note: Empty value is valid (disables loading from file)
        filenames.getfromfilename = value;
        return true;
    }

    if (lowerVarname == "initial_population")
    {
        // Note: Empty value is valid (disables initial population loading)
        filenames.initialpopfilename = value;
        return true;
    }

    // ========================================================================
    // Parallel Execution Parameters
    // ========================================================================

    if (lowerVarname == "numthreads")
    {
        const int numThreads = aquiutils::atoi(value);

        // Validate number of threads
        if (numThreads <= 0)
        {
            last_error = "Invalid numthreads value '" + value +
                         "'. Number of threads must be positive.";
            return false;
        }

        // Warn about very large thread counts
        if (numThreads > 64)
        {
            last_error = "Warning: Large numthreads value '" + value +
                         "'. Using more than 64 threads may reduce performance due to overhead.";
            write_to_detailed_GA(last_error);
            // Don't return false - just warn
        }

        numberOfThreads = numThreads;
        return true;
    }

    // ========================================================================
    // Parameter Not Found
    // ========================================================================

    last_error = "Unknown parameter '" + varname + "'. Valid parameters are: " +
                 "maxpop, ngen, pcross, pmute, shakescale, shakescalered, " +
                 "outputfile, getfromfilename, initial_population, numthreads";
    return false;
}

template<class T>
double CGA<T>::avgfitness() const
{
    if (GA_params.maxpop == 0)
    {
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        sum += Ind[i].fitness;
    }

    return sum / static_cast<double>(GA_params.maxpop);
}

/**
 * @brief Calculate average actual fitness (objective function values)
 * @return Average actual fitness across population
 *
 * Computes the mean of the actual objective function values.
 * For minimization problems, lower values are better.
 *
 * Formula: avg = (Σ actual_fitness[i]) / maxpop
 *
 * This is useful for tracking convergence and population quality.
 *
 * @note Returns actual objective function values, not scaled fitness
 * @see avgfitness() for scaled fitness average
 */
template<class T>
double CGA<T>::avg_actual_fitness() const
{
    if (GA_params.maxpop == 0)
    {
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        sum += Ind[i].actual_fitness;
    }

    return sum / static_cast<double>(GA_params.maxpop);
}

/**
 * @brief Calculate average inverse actual fitness
 * @return Average of (1 / actual_fitness) across population
 *
 * Computes the mean of inverse fitness values. This is useful for
 * analyzing fitness distribution and can be more stable than direct
 * fitness values when some individuals have very small fitness.
 *
 * Formula: avg_inv = (Σ 1/actual_fitness[i]) / maxpop
 *
 * Warning: Division by zero if any individual has zero fitness
 *
 * @see stdfitness() which uses this value
 */
template<class T>
double CGA<T>::avg_inv_actual_fitness() const
{
    if (GA_params.maxpop == 0)
    {
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        // Avoid division by zero
        if (Ind[i].actual_fitness != 0.0)
        {
            sum += 1.0 / Ind[i].actual_fitness;
        }
        else
        {
            // Handle zero fitness - use large penalty value
            sum += 1e10;  // Effectively infinite inverse fitness
        }
    }

    return sum / static_cast<double>(GA_params.maxpop);
}

/**
 * @brief Calculate variance of scaled fitness
 * @return Fitness variance
 *
 * Computes the variance of scaled fitness values. Low variance indicates
 * population convergence; high variance indicates diversity.
 *
 * Formula: var = Σ(avg - fitness[i])²
 *
 * Note: Returns sum of squared deviations, not divided by population size.
 * This is used internally and may not be the standard statistical variance.
 *
 * @see stdfitness() for standard deviation
 */
template<class T>
double CGA<T>::variancefitness() const
{
    if (GA_params.maxpop == 0)
    {
        return 0.0;
    }

    const double avg = avgfitness();
    double sum = 0.0;

    for (int i = 0; i < GA_params.maxpop; i++)
    {
        const double deviation = avg - Ind[i].fitness;
        sum += deviation * deviation;
    }

    return sum;
}

/**
 * @brief Calculate normalized standard deviation of fitness
 * @return Normalized standard deviation (coefficient of variation)
 *
 * Computes a normalized measure of fitness dispersion based on inverse
 * actual fitness values. This provides a scale-independent measure of
 * population diversity.
 *
 * Formula: normalized_std = sqrt(Σ(avg_inv - 1/fitness[i])²) / maxpop / avg_inv
 *
 * Values close to 0 indicate convergence; higher values indicate diversity.
 *
 * @note Uses inverse fitness for numerical stability
 * @see avg_inv_actual_fitness()
 */
template<class T>
double CGA<T>::stdfitness() const
{
    if (GA_params.maxpop == 0)
    {
        return 0.0;
    }

    const double avg_inv = avg_inv_actual_fitness();

    // Avoid division by zero
    if (avg_inv == 0.0)
    {
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        double inv_fitness;
        if (Ind[i].actual_fitness != 0.0)
        {
            inv_fitness = 1.0 / Ind[i].actual_fitness;
        }
        else
        {
            inv_fitness = 1e10;  // Large value for zero fitness
        }

        const double deviation = avg_inv - inv_fitness;
        sum += deviation * deviation;
    }

    // Normalize by population size and average
    return std::sqrt(sum) / static_cast<double>(GA_params.maxpop) / avg_inv;
}

/**
 * @brief Find index of individual with best (minimum) fitness
 * @return Index of best individual in population (0-based)
 *
 * Searches for the individual with the lowest actual_fitness value.
 * For minimization problems, lower fitness is better.
 *
 * If multiple individuals have the same best fitness, returns the first one.
 *
 * @note Returns index, not fitness value
 * @note For minimization problems (lower is better)
 */
template<class T>
int CGA<T>::maxfitness() const
{
    if (GA_params.maxpop == 0)
    {
        return -1;  // Invalid index for empty population
    }

    double bestFitness = 1e308;  // Very large initial value
    int bestIndex = 0;

    for (int i = 0; i < GA_params.maxpop; i++)
    {
        if (Ind[i].actual_fitness < bestFitness)
        {
            bestFitness = Ind[i].actual_fitness;
            bestIndex = i;
        }
    }

    return bestIndex;
}


template<class T>
void CGA<T>::write_to_detailed_GA(const std::string& message) const
{
    // Construct full file path
    const std::string logFilePath = filenames.pathname + "detail_GA.txt";

    // Try to open file in append mode
    FILE* fileOut = fopen(logFilePath.c_str(), "a");

    if (fileOut == nullptr)
    {
        // File could not be opened - print warning
        std::cerr << "Warning: Could not open log file '" << logFilePath
                  << "' for writing." << std::endl;
        return;
    }

    // Write message with newline
    fprintf(fileOut, "%s\n", message.c_str());

    // Close file
    fclose(fileOut);
}


/**
 * @brief Run the genetic algorithm optimization
 * @return Index of best individual in final population
 *
 * This is the main optimization loop that runs the complete GA process:
 *
 * Algorithm:
 * 1. Initialize population
 * 2. For each generation:
 *    a. Evaluate fitness for all individuals
 *    b. Log progress and write results
 *    c. Adapt parameters (shake scale, mutation rate)
 *    d. Select parents based on fitness
 *    e. Apply crossover operator
 *    f. Apply mutation operator
 *    g. Apply shake operator (perturbation)
 * 3. Final fitness evaluation
 * 4. Return best solution
 *
 * The method implements several adaptive mechanisms:
 * - Shake scale reduction when stagnant
 * - Shake scale increase when improving
 * - Enhancement counter for diversity maintenance
 *
 * Progress is logged to:
 * - Console (printf)
 * - Main output file (specified in filenames.outputfilename)
 * - Detailed log (detail_GA.txt)
 * - GUI window (if Q_GUI_SUPPORT enabled)
 *
 * @throws std::runtime_error if output file cannot be opened
 * @see initialize() for population initialization
 * @see assignfitnesses() for fitness evaluation
 */
template<class T>
int CGA<T>::optimize()
{
#ifdef Q_GUI_SUPPORT
    QCoreApplication::processEvents();
#endif

    // ========================================================================
    // Step 1: Setup and File Initialization
    // ========================================================================

    // Determine output file path
    std::string runFileName;
    if (aquiutils::contains(filenames.outputfilename, "/"))
    {
        // Absolute path or contains directory separator
        runFileName = filenames.outputfilename;
    }
    else
    {
        // Relative path - prepend pathname
        runFileName = filenames.pathname + filenames.outputfilename;
    }

    // Open and initialize main output file
    FILE* fileOut = fopen(runFileName.c_str(), "w");
    if (!fileOut)
    {
        last_error = "Unable to open output file '" + runFileName + "'";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::runtime_error(last_error);
    }
    fclose(fileOut);

    // Initialize detailed log file
    const std::string detailLogPath = filenames.pathname + "detail_GA.txt";
    FILE* detailFile = fopen(detailLogPath.c_str(), "w");
    if (detailFile)
    {
        fclose(detailFile);
    }
    else
    {
        std::cerr << "Warning: Could not initialize detail log file" << std::endl;
    }

    // ========================================================================
    // Step 2: Initialize Population
    // ========================================================================

    write_to_detailed_GA("Initializing population...");
#ifndef Q_GUI_SUPPORT
    std::cout << "Initializing population..." << std::endl;
#endif
    initialize();
    write_to_detailed_GA("Population initialized.");
#ifndef Q_GUI_SUPPORT
    std::cout << "Population initialized." << std::endl;
    std::cout << "Starting optimization for " << GA_params.nGen << " generations..." << std::endl;
    std::cout << "Population size: " << GA_params.maxpop << std::endl;
    std::cout << std::string(60, '-') << std::endl;
#endif

    // ========================================================================
    // Step 3: Setup Adaptive Parameters
    // ========================================================================

    const double initialShakeScale = GA_params.shakescale;
    const double initialEnhancements = static_cast<double>(GA_params.numenhancements);
    GA_params.numenhancements = 0;

    // Matrix to track fitness history for adaptation
    CMatrix fitnessHistory(GA_params.nGen, 3);

    // ========================================================================
    // Step 4: Main Optimization Loop
    // ========================================================================

    for (current_generation = 0; current_generation < GA_params.nGen; current_generation++)
    {
        // --------------------------------------------------------------------
        // Fitness Evaluation
        // --------------------------------------------------------------------

        write_to_detailed_GA("Assigning fitnesses ...");
        Models.clear();
        assignfitnesses();

        write_to_detailed_GA("Assigning fitnesses done!");

        // --------------------------------------------------------------------
        // Write Generation Results
        // --------------------------------------------------------------------

        fileOut = fopen(runFileName.c_str(), "a");
        if (!fileOut)
        {
            std::cerr << "Warning: Could not append to output file" << std::endl;
        }
        else
        {
            // Write generation header
            printf("Generation: %i\n", current_generation);
            fprintf(fileOut, "Generation: %i\n", current_generation);

            // Write column headers
            fprintf(fileOut, "ID, ");
            for (int k = 0; k < Ind[0].getNumParams(); k++)
            {
                fprintf(fileOut, "%s, ", paramname[k].c_str());
            }
            fprintf(fileOut, "%s, %s, %s, ", "likelihood", "Fitness", "Rank");

            // Add observation-specific fitness measures
            for (unsigned int i = 0; i < Model->ObservationsCount(); i++)
            {
                const std::string obsName = Model->observation(i)->GetName();
                fprintf(fileOut, "%s, %s, %s, ",
                        (obsName + "_MSE").c_str(),
                        (obsName + "_R2").c_str(),
                        (obsName + "_NSE").c_str());
            }
            fprintf(fileOut, "\n");

            // Write each individual's data
            write_to_detailed_GA("Generation: " + aquiutils::numbertostring(current_generation));
            for (int j = 0; j < GA_params.maxpop; j++)
            {
                fprintf(fileOut, "%i, ", j);

                // Write parameter values (transform log-scale back)
                for (int k = 0; k < Ind[0].getNumParams(); k++)
                {
                    if (loged[k] == 1)
                    {
                        fprintf(fileOut, "%le, ", pow(10.0, Ind[j].x[k]));
                    }
                    else
                    {
                        fprintf(fileOut, "%le, ", Ind[j].x[k]);
                    }
                }

                // Write fitness values
                fprintf(fileOut, "%le, %le, %i, ",
                        Ind[j].actual_fitness,
                        Ind[j].fitness,
                        Ind[j].rank);

                // Write detailed fitness measures
                for (unsigned int i = 0; i < Model->ObservationsCount(); i++)
                {
                    fprintf(fileOut, "%le, %le, %le, ",
                            Ind[j].fit_measures[i * 3],
                            Ind[j].fit_measures[i * 3 + 1],
                            Ind[j].fit_measures[i * 3 + 2]);
                }
                fprintf(fileOut, "\n");
            }

            fclose(fileOut);
        }

        // --------------------------------------------------------------------
        // Update GUI Progress
        // --------------------------------------------------------------------

        const int bestIndex = maxfitness();
        fitnessHistory[current_generation][0] = Ind[bestIndex].actual_fitness;

#ifdef Q_GUI_SUPPORT
        if (rtw)
        {
            if (current_generation == 0)
            {
                rtw->SetYRange(0, Ind[bestIndex].actual_fitness * 1.1);
            }
            rtw->SetProgress(static_cast<double>(current_generation) /
                             static_cast<double>(GA_params.nGen));
            rtw->AddDataPoint(current_generation + 1, Ind[bestIndex].actual_fitness);
            rtw->Replot();
            QCoreApplication::processEvents();
        }
#else
        // Console progress output
        const double progress = 100.0 * static_cast<double>(current_generation + 1) /
                                static_cast<double>(GA_params.nGen);
        std::cout << "Generation " << std::setw(4) << current_generation
                  << " | Progress: " << std::fixed << std::setprecision(1) << std::setw(5) << progress << "% "
                  << "| Best Fitness: " << std::scientific << std::setprecision(4)
                  << Ind[bestIndex].actual_fitness
                  << " | Shake Scale: " << std::scientific << std::setprecision(2)
                  << GA_params.shakescale << std::endl;
#endif

        // --------------------------------------------------------------------
        // Adaptive Parameter Adjustment
        // --------------------------------------------------------------------

        // Reduce shake scale if stagnant (after generation 10)
        if (current_generation > 10)
        {
            const bool isStagnant =
                (fitnessHistory[current_generation][0] ==
                 fitnessHistory[current_generation - 3][0]);

            const bool shakeScaleAboveMinimum =
                GA_params.shakescale > pow(10.0, -Ind[0].precision[0]);

            if (isStagnant && shakeScaleAboveMinimum)
            {
                GA_params.shakescale *= GA_params.shakescalered;
#ifndef Q_GUI_SUPPORT
                std::cout << "  -> Fitness stagnant, reducing shake scale to "
                          << std::scientific << std::setprecision(2)
                          << GA_params.shakescale << std::endl;
#endif
            }

            // Increase shake scale if improving
            const bool isImproving =
                (fitnessHistory[current_generation][0] >
                 fitnessHistory[current_generation - 1][0]);

            const bool shakeScaleBelowInitial =
                (GA_params.shakescale < initialShakeScale);

            if (isImproving && shakeScaleBelowInitial)
            {
                GA_params.shakescale /= GA_params.shakescalered;
#ifndef Q_GUI_SUPPORT
                std::cout << "  -> Fitness improving, increasing shake scale to "
                          << std::scientific << std::setprecision(2)
                          << GA_params.shakescale << std::endl;
#endif
            }

            GA_params.numenhancements = 0;
        }

        // Adjust enhancements based on longer-term stagnation (after generation 50)
        if (current_generation > 50)
        {
            if (fitnessHistory[current_generation][0] ==
                fitnessHistory[current_generation - 20][0])
            {
                GA_params.numenhancements *= 1.05;
                if (GA_params.numenhancements == 0)
                {
                    GA_params.numenhancements = static_cast<int>(initialEnhancements);
                }
#ifndef Q_GUI_SUPPORT
                std::cout << "  -> Long-term stagnation detected, adjusting enhancements to "
                          << GA_params.numenhancements << std::endl;
#endif
            }

            if (fitnessHistory[current_generation][0] ==
                fitnessHistory[current_generation - 50][0])
            {
                GA_params.numenhancements = static_cast<int>(initialEnhancements * 10);
#ifndef Q_GUI_SUPPORT
                std::cout << "  -> Extended stagnation detected, increasing enhancements to "
                          << GA_params.numenhancements << std::endl;
#endif
            }
        }

        // Store shake scale and mutation rate in history
        fitnessHistory[current_generation][1] = GA_params.shakescale;
        fitnessHistory[current_generation][2] = GA_params.pmute;

        // Reset shake scale if stuck (after generation 20)
        if (current_generation > 20)
        {
            if (GA_params.shakescale == fitnessHistory[current_generation - 20][1])
            {
                GA_params.shakescale = initialShakeScale;
#ifndef Q_GUI_SUPPORT
                std::cout << "  -> Resetting shake scale to initial value: "
                          << std::scientific << std::setprecision(2)
                          << initialShakeScale << std::endl;
#endif
            }
        }

        // Update best fitness tracker
        MaxFitness = Ind[bestIndex].actual_fitness;
        fitnessHistory[current_generation][0] = Ind[bestIndex].actual_fitness;

        // --------------------------------------------------------------------
        // Genetic Operators
        // --------------------------------------------------------------------

        // Fill fitness distribution for parent selection
        fillfitdist();

        // Crossover
        write_to_detailed_GA("Cross-over ...");
        if (GA_params.RCGA)
        {
            crossoverRC();
        }
        else
        {
            crossover();
        }
        write_to_detailed_GA("Cross-over done!");

        // Mutation
        write_to_detailed_GA("Mutation ...");
        mutate(GA_params.pmute);
        write_to_detailed_GA("Mutation done!");

        // Shake (perturbation)
        write_to_detailed_GA("Shake...!");
        shake();
        write_to_detailed_GA("Shake done!");
    }

    // ========================================================================
    // Step 5: Final Evaluation and Results
    // ========================================================================

#ifndef Q_GUI_SUPPORT
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Optimization complete!" << std::endl;
    std::cout << "Performing final fitness evaluation..." << std::endl;
#endif

    write_to_detailed_GA("Performing final fitness evaluation...");
    write_to_detailed_GA("Performing final fitness evaluation...");
    Models.clear();
    assignfitnesses();

    fileOut = fopen(runFileName.c_str(), "a");
    if (fileOut)
    {
        fprintf(fileOut, "Final Enhancements\n");

        const int bestIndex = maxfitness();
        MaxFitness = Ind[bestIndex].actual_fitness;
        final_params.resize(GA_params.nParam);

        // Write final parameters
        for (int k = 0; k < Ind[0].getNumParams(); k++)
        {
            if (loged[k] == 1)
            {
                final_params[k] = pow(10.0, Ind[bestIndex].x[k]);
            }
            else
            {
                final_params[k] = Ind[bestIndex].x[k];
            }

            fprintf(fileOut, "%s, ", paramname[k].c_str());
            fprintf(fileOut, "%le, ", final_params[k]);
            fprintf(fileOut, "%le, %le\n",
                    Ind[bestIndex].actual_fitness,
                    Ind[bestIndex].fitness);
        }

        // Write final fitness measures
        for (unsigned int i = 0; i < Model->ObservationsCount(); i++)
        {
            fprintf(fileOut, "%le, %le, %le\n",
                    Ind[bestIndex].fit_measures[i * 3],
                    Ind[bestIndex].fit_measures[i * 3 + 1],
                    Ind[bestIndex].fit_measures[i * 3 + 2]);
        }

        fclose(fileOut);
    }

    // Evaluate final parameters
    assignfitnesses(final_params);

#ifdef Q_GUI_SUPPORT
    if (rtw)
    {
        rtw->SetProgress(1.0);
        QCoreApplication::processEvents();
    }
#else
    std::cout << "Final best fitness: " << std::scientific << std::setprecision(6)
              << MaxFitness << std::endl;
    std::cout << "Results written to: " << runFileName << std::endl;
#endif

    // Clean up
    Models.clear();

    write_to_detailed_GA("Optimization complete!");

    return maxfitness();
}



template<class T>
double CGA<T>::assignfitnesses(const std::vector<double>& parameters)
{
    // Validate input size
    if (static_cast<int>(parameters.size()) != GA_params.nParam)
    {
        last_error = "Parameter vector size (" +
                     std::to_string(parameters.size()) +
                     ") does not match expected (" +
                     std::to_string(GA_params.nParam) + ")";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::invalid_argument(last_error);
    }

    // Create a copy of the base model
    T modelInstance = *Model;

    // Set all parameters
    for (int i = 0; i < GA_params.nParam; i++)
    {
        modelInstance.SetParameterValue(i, parameters[i]);
    }

    // Get objective function value
    const double objectiveFunctionValue = modelInstance.GetObjectiveFunctionValue();

    // Store results for later retrieval
    if (Model_out != nullptr)
    {
        delete Model_out;
    }
    Model_out = new T(modelInstance);

    // Return negative objective function (legacy convention)
    // Note: GA minimizes, so negative converts maximization to minimization
    return -objectiveFunctionValue;
}

template<class T>
double CGA<T>::getfromoutput(const std::string& filename)
{
    // ========================================================================
    // Step 1: Open and validate file
    // ========================================================================

    std::ifstream file(filename);
    if (!file.is_open())
    {
        last_error = "Cannot open GA output file '" + filename + "'";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::runtime_error(last_error);
    }

    // ========================================================================
    // Step 2: Search for "Final Enhancements" section
    // ========================================================================

    bool foundFinalSection = false;
    std::vector<std::string> line;

    while (!file.eof())
    {
        line = aquiutils::getline(file);

        if (line.size() > 0 && line[0] == "Final Enhancements")
        {
            foundFinalSection = true;
            break;
        }
    }

    if (!foundFinalSection)
    {
        file.close();
        last_error = "Could not find 'Final Enhancements' section in file '" +
                     filename + "'";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::runtime_error(last_error);
    }

    // ========================================================================
    // Step 3: Read parameter values
    // ========================================================================

    final_params.clear();
    final_params.resize(GA_params.nParam);

    for (int i = 0; i < GA_params.nParam; i++)
    {
        line = aquiutils::getline(file);

        // Validate line was read and has data
        if (line.size() == 0)
        {
            file.close();
            last_error = "Unexpected end of file while reading parameter " +
                         std::to_string(i) + ". Expected " +
                         std::to_string(GA_params.nParam) + " parameters but file has fewer.";
            std::cerr << "Error: " << last_error << std::endl;
            throw std::runtime_error(last_error);
        }

        // Validate line has enough fields (format: name, value, fitness, scaled_fitness)
        if (line.size() < 2)
        {
            file.close();
            last_error = "Invalid format in output file at parameter " +
                         std::to_string(i) + ". Expected at least 2 fields (name, value).";
            std::cerr << "Error: " << last_error << std::endl;
            throw std::runtime_error(last_error);
        }

        // Parse parameter value (second field)
        try
        {
            final_params[i] = aquiutils::atof(line[1]);
        }
        catch (...)
        {
            file.close();
            last_error = "Cannot parse parameter value '" + line[1] +
                         "' for parameter " + std::to_string(i);
            std::cerr << "Error: " << last_error << std::endl;
            throw std::runtime_error(last_error);
        }
    }

    file.close();

    // ========================================================================
    // Step 4: Log success
    // ========================================================================

    write_to_detailed_GA("Successfully loaded " + std::to_string(GA_params.nParam) +
                         " parameters from file: " + filename);

    // Log parameter values for verification
    std::string paramLog = "Loaded parameters: ";
    for (int i = 0; i < std::min(5, GA_params.nParam); i++)
    {
        paramLog += std::to_string(final_params[i]);
        if (i < std::min(5, GA_params.nParam) - 1)
            paramLog += ", ";
    }
    if (GA_params.nParam > 5)
        paramLog += ", ...";

    write_to_detailed_GA(paramLog);

    // ========================================================================
    // Step 5: Evaluate loaded parameters
    // ========================================================================

    write_to_detailed_GA("Evaluating loaded parameters...");

    double fitness = 0.0;
    try
    {
        fitness = assignfitnesses(final_params);
    }
    catch (const std::exception& e)
    {
        last_error = "Failed to evaluate loaded parameters: " + std::string(e.what());
        std::cerr << "Error: " << last_error << std::endl;
        throw;
    }

    write_to_detailed_GA("Evaluation complete. Fitness: " + std::to_string(fitness));

    return fitness;
}


template<class T>
int CGA<T>::getparamno(int i, int ts)
{
    int l = 0;
    for (int j = 0; j<i; j++)
        if (apply_to_all[j]) l++; else l += 1;

    if (apply_to_all[i])
        return l;
    else
        return l + ts;

}


template<class T>
void CGA<T>::shake()
{
    // Skip index 0 (elite individual preserved by elitism)
    for (int i = 1; i < GA_params.maxpop; i++)
    {
        Ind[i].shake(GA_params.shakescale);
    }
}

template<class T>
void CGA<T>::mutate(double mutationRate)
{
    // Validate mutation rate
    if (mutationRate < 0.0 || mutationRate > 1.0)
    {
        last_error = "Invalid mutation rate " + std::to_string(mutationRate) +
                     ". Must be in range [0.0, 1.0].";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::invalid_argument(last_error);
    }

    // Skip indices 0 and 1 (elite individuals preserved by elitism)
    // Elite individuals are the best from previous generation
    for (int i = 2; i < GA_params.maxpop; i++)
    {
        Ind[i].mutate(mutationRate);
    }
}


/**
 * @brief Assign rank to each individual based on actual fitness
 *
 * Ranks individuals in the population based on their actual fitness values.
 * For minimization problems (lower fitness is better):
 * - Rank 1 = best individual (lowest actual_fitness)
 * - Rank 2 = second best
 * - Rank N = worst individual
 *
 * If multiple individuals have identical fitness, they may receive different
 * ranks (order depends on iteration order).
 *
 * Algorithm: For each individual i, count how many individuals j have
 * better fitness (actual_fitness[j] < actual_fitness[i]). Rank = count + 1.
 *
 * Complexity: O(n²) where n = population size
 *
 * @note This method is called by assignfitness_rank()
 * @note For minimization: lower actual_fitness → lower rank (better)
 *
 * Example:
 * actual_fitness = [0.5, 0.2, 0.8, 0.2]
 * ranks =          [  2,   1,   4,   1]  (ties get same rank in this impl)
 */
template<class T>
void CGA<T>::assignrank()
{
    // Handle empty population
    if (GA_params.maxpop == 0)
    {
        return;
    }

    // Assign rank to each individual
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        int rank = 1;  // Start with best rank

        // Count how many individuals are better than this one
        for (int j = 0; j < GA_params.maxpop; j++)
        {
            // For minimization: lower fitness is better
            if (Ind[i].actual_fitness > Ind[j].actual_fitness)
            {
                rank++;
            }
        }

        Ind[i].rank = rank;
    }
}

/**
 * @brief Assign scaled fitness based on rank
 * @param exponent Exponent for rank-based scaling (typically 1.0)
 *
 * Converts ranks to scaled fitness values used for selection.
 * This implements rank-based selection where fitness depends only on
 * rank, not on the actual objective function values.
 *
 * Formula: fitness[i] = (1 / rank[i])^exponent
 *
 * Process:
 * 1. Call assignrank() to compute ranks
 * 2. Convert each rank to scaled fitness
 *
 * Effect of exponent:
 * - exponent = 1.0: Linear rank-based selection (standard)
 *   - Rank 1 → fitness = 1.0
 *   - Rank 2 → fitness = 0.5
 *   - Rank 10 → fitness = 0.1
 *
 * - exponent > 1.0: Stronger selection pressure
 *   - exponent = 2.0:
 *     - Rank 1 → fitness = 1.0
 *     - Rank 2 → fitness = 0.25
 *     - Rank 10 → fitness = 0.01
 *
 * - exponent < 1.0: Weaker selection pressure
 *   - exponent = 0.5:
 *     - Rank 1 → fitness = 1.0
 *     - Rank 2 → fitness = 0.707
 *     - Rank 10 → fitness = 0.316
 *
 * Benefits of rank-based selection:
 * - Robust to fitness scaling issues
 * - Prevents premature convergence with super-fit individuals
 * - Maintains selection pressure even with similar fitness values
 * - Independent of objective function magnitude
 *
 * @note Uses GA_params.N (should use parameter, but kept for compatibility)
 * @note Lower rank (better individual) → higher scaled fitness
 * @note Scaled fitness is used by fillfitdist() for parent selection
 *
 * @see assignrank() for rank computation
 * @see fillfitdist() for selection probability computation
 */
template<class T>
void CGA<T>::assignfitness_rank(double exponent)
{
    // First assign ranks based on actual fitness
    assignrank();

    // Convert ranks to scaled fitness values
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        // Avoid division by zero (shouldn't happen with valid ranks)
        if (Ind[i].rank <= 0)
        {
            std::cerr << "Warning: Invalid rank " << Ind[i].rank
                      << " for individual " << i << ". Setting rank to 1." << std::endl;
            Ind[i].rank = 1;
        }

        // fitness = (1/rank)^exponent
        // Note: Uses GA_params.N instead of parameter (legacy behavior)
        const double inverseRank = 1.0 / static_cast<double>(Ind[i].rank);
        Ind[i].fitness = std::pow(inverseRank, GA_params.N);
    }
}

/**
 * @brief Fill fitness distribution for roulette wheel selection
 *
 * Creates a cumulative probability distribution based on scaled fitness
 * values. This enables fitness-proportionate selection (roulette wheel).
 *
 * For each individual i, stores:
 * - fitdist.s[i]: Start of probability interval
 * - fitdist.e[i]: End of probability interval
 *
 * The distribution spans [0, 1] where each individual's interval size
 * is proportional to their scaled fitness.
 *
 * Example:
 * Fitness values: [1.0, 0.5, 0.25, 0.25]
 * Sum = 2.0
 * Probabilities: [0.5, 0.25, 0.125, 0.125]
 *
 * Distribution intervals:
 * - Individual 0: [0.0,   0.5  )
 * - Individual 1: [0.5,   0.75 )
 * - Individual 2: [0.75,  0.875)
 * - Individual 3: [0.875, 1.0  ]
 *
 * Selection: Generate random number r ∈ [0,1], select individual
 * whose interval contains r. Individuals with higher fitness have
 * larger intervals, thus higher selection probability.
 *
 * @note Requires assignfitness_rank() to be called first
 * @note Last interval always ends at exactly 1.0
 * @note Uses fitdist member (CDistribution object)
 *
 * @see fitdist.GetRand() to select using this distribution
 */
template<class T>
void CGA<T>::fillfitdist()
{
    // Handle empty or single-individual population
    if (GA_params.maxpop <= 0)
    {
        return;
    }

    if (GA_params.maxpop == 1)
    {
        fitdist.s[0] = 0.0;
        fitdist.e[0] = 1.0;
        return;
    }

    // Calculate sum of all fitness values
    double sumFitness = 0.0;
    for (int i = 0; i < GA_params.maxpop; i++)
    {
        sumFitness += Ind[i].fitness;
    }

    // Handle case where all fitness values are zero
    if (sumFitness == 0.0)
    {
        // Equal probability for all individuals
        for (int i = 0; i < GA_params.maxpop; i++)
        {
            fitdist.s[i] = static_cast<double>(i) / static_cast<double>(GA_params.maxpop);
            fitdist.e[i] = static_cast<double>(i + 1) / static_cast<double>(GA_params.maxpop);
        }
        return;
    }

    // Build cumulative distribution

    // First individual
    fitdist.s[0] = 0.0;
    fitdist.e[0] = Ind[0].fitness / sumFitness;

    // Middle individuals (if any)
    for (int i = 1; i < GA_params.maxpop - 1; i++)
    {
        fitdist.s[i] = fitdist.e[i - 1];
        fitdist.e[i] = fitdist.e[i - 1] + Ind[i].fitness / sumFitness;
    }

    // Last individual (ensure end is exactly 1.0 to avoid rounding errors)
    fitdist.s[GA_params.maxpop - 1] = fitdist.e[GA_params.maxpop - 2];
    fitdist.e[GA_params.maxpop - 1] = 1.0;
}


/**
 * @brief Evaluate forward model with default parameters
 * @return Likelihood value written to file
 *
 * This method appears to be a legacy testing/debugging function that:
 * 1. Temporarily modifies GA configuration
 * 2. Evaluates model with parameter value = 1.0
 * 3. Writes result to likelihood.txt
 * 4. Restores original configuration
 *
 * Purpose unclear - possibly for:
 * - Model validation
 * - Forward model evaluation
 * - Testing infrastructure
 *
 * @note This is a legacy method with unclear purpose
 * @note Temporarily modifies GA_params and params members
 * @note Creates file "likelihood.txt" in pathname directory
 * @deprecated Consider removing or documenting actual use case
 *
 * @throws May throw if file cannot be written
 */
template<class T>
double CGA<T>::evaluateforward()
{
    write_to_detailed_GA("Warning: evaluateforward() is a legacy method with unclear purpose");

    // Save current configuration
    const int savedNumParam = GA_params.nParam;
    const std::vector<int> savedParams = params;

    // Temporarily modify configuration
    GA_params.nParam = 1;
    params.resize(1);
    params[0] = 100;  // Unclear why 100

    // Evaluate with single parameter = 1.0
    std::vector<double> parameterValue(1, 1.0);
    CVector output(1);

    try
    {
        output[0] = assignfitnesses(parameterValue);
    }
    catch (const std::exception& e)
    {
        // Restore configuration before re-throwing
        params = savedParams;
        GA_params.nParam = savedNumParam;
        throw;
    }

    // Write to file
    const std::string outputPath = filenames.pathname + "likelihood.txt";
    try
    {
        output.writetofile(outputPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Warning: Could not write to " << outputPath
                  << ": " << e.what() << std::endl;
    }

    // Restore original configuration
    params = savedParams;
    GA_params.nParam = savedNumParam;

    return output[0];
}

/**
 * @brief Load initial population from previous GA output file
 * @param filename Path to GA output file
 *
 * Reads the "Final Enhancements" section from a previous GA run and
 * uses those parameter values as the initial population (single individual).
 *
 * This is used by initialize() if an initial population file is specified.
 *
 * File format expected:
 * ...
 * Final Enhancements
 * param1, value, fitness, scaled_fitness
 * param2, value, fitness, scaled_fitness
 * ...
 *
 * @note Populates initial_pop member (1 individual with nParam parameters)
 * @note Does NOT transform log-scale parameters (stores raw values)
 * @note Called by initialize() if filenames.initialpopfilename is set
 * @note Duplicate of getfromoutput() logic - consider refactoring
 *
 * @throws std::runtime_error if file cannot be opened
 * @throws std::runtime_error if file format is invalid
 *
 * @see initialize() which uses this data
 * @see getfromoutput() which has similar functionality
 */
template<class T>
void CGA<T>::getinifromoutput(const std::string& filename)
{
    // Open file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        last_error = "Cannot open initial population file '" + filename + "'";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::runtime_error(last_error);
    }

    // Initialize storage for one individual
    initial_pop.clear();
    initial_pop.resize(1);
    initial_pop[0].resize(GA_params.nParam);

    // Search for "Final Enhancements" section
    bool foundSection = false;
    std::vector<std::string> line;

    while (!file.eof())
    {
        line = aquiutils::getline(file);

        if (line.size() > 0 && line[0] == "Final Enhancements")
        {
            foundSection = true;
            break;
        }
    }

    if (!foundSection)
    {
        file.close();
        last_error = "Could not find 'Final Enhancements' section in file '" +
                     filename + "'";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::runtime_error(last_error);
    }

    // Read parameter values
    for (int i = 0; i < GA_params.nParam; i++)
    {
        line = aquiutils::getline(file);

        // Validate line
        if (line.size() == 0)
        {
            file.close();
            last_error = "Unexpected end of file at parameter " +
                         std::to_string(i) + " in '" + filename + "'";
            std::cerr << "Error: " << last_error << std::endl;
            throw std::runtime_error(last_error);
        }

        if (line.size() < 2)
        {
            file.close();
            last_error = "Invalid format at parameter " + std::to_string(i) +
                         " in '" + filename + "'";
            std::cerr << "Error: " << last_error << std::endl;
            throw std::runtime_error(last_error);
        }

        // Parse value (second field)
        try
        {
            // Note: Original code has duplicate logic (loged check does nothing)
            // Storing raw value regardless of log scale
            initial_pop[0][i] = aquiutils::atof(line[1]);
        }
        catch (...)
        {
            file.close();
            last_error = "Cannot parse parameter value '" + line[1] +
                         "' at parameter " + std::to_string(i);
            std::cerr << "Error: " << last_error << std::endl;
            throw std::runtime_error(last_error);
        }
    }

    file.close();

    write_to_detailed_GA("Loaded initial population (1 individual) from: " + filename);
}

/**
 * @brief Load initial population from CSV-style file
 * @param filename Path to population file
 *
 * Reads initial population from a CSV file where each line represents
 * one individual's parameter values.
 *
 * File format:
 * param1_ind1, param2_ind1, param3_ind1, ...
 * param1_ind2, param2_ind2, param3_ind2, ...
 * ...
 *
 * @note Different from getinifromoutput() - this reads raw CSV data
 * @note Each line becomes one individual in initial_pop
 * @note No header line expected
 * @note Empty lines are ignored
 * @note Number of values per line should match GA_params.nParam
 *
 * @throws std::runtime_error if file cannot be opened
 *
 * @see initialize() which uses this data if specified
 *
 * Example file:
 * 1.5, 2.3, 0.8
 * 2.1, 1.7, 1.2
 * 0.9, 3.1, 0.5
 */
template<class T>
void CGA<T>::getinitialpop(const std::string& filename)
{
    // Open file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        last_error = "Cannot open initial population file '" + filename + "'";
        std::cerr << "Error: " << last_error << std::endl;
        throw std::runtime_error(last_error);
    }

    // Clear any existing data
    initial_pop.clear();

    // Read all lines
    std::vector<std::string> line;
    int lineNumber = 0;

    while (!file.eof())
    {
        line = aquiutils::getline(file);
        lineNumber++;

        // Skip empty lines
        if (line.size() == 0)
        {
            continue;
        }

        // Validate number of parameters
        if (static_cast<int>(line.size()) != GA_params.nParam)
        {
            std::cerr << "Warning: Line " << lineNumber << " has " << line.size()
            << " values but expected " << GA_params.nParam
            << ". Skipping line." << std::endl;
            continue;
        }

        // Parse parameter values for this individual
        std::vector<double> individual(GA_params.nParam);
        bool parseSuccess = true;

        for (int j = 0; j < GA_params.nParam; j++)
        {
            try
            {
                individual[j] = aquiutils::ATOF(line);
            }
            catch (...)
            {
                std::cerr << "Warning: Cannot parse value '" << line[j]
                          << "' at line " << lineNumber << ", column " << j
                          << ". Skipping line." << std::endl;
                parseSuccess = false;
                break;
            }
        }

        // Add individual to population if parsing succeeded
        if (parseSuccess)
        {
            initial_pop.push_back(individual);
        }
    }

    file.close();

    // Log result
    if (initial_pop.size() > 0)
    {
        write_to_detailed_GA("Loaded " + std::to_string(initial_pop.size()) +
                             " individuals from: " + filename);
    }
    else
    {
        std::cerr << "Warning: No valid individuals loaded from " << filename << std::endl;
    }
}
