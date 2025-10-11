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

#ifndef PARAMETER_H
#define PARAMETER_H

#include <string>
#include "TimeSeries.h"
#include "TimeSeriesSet.h"

/**
 * @brief Represents a single parameter with prior distribution for calibration
 *
 * This class stores a parameter's value, name, valid range, and prior distribution
 * information. It supports uniform, normal, and log-normal priors and can calculate
 * log prior probabilities for Bayesian inference (MCMC).
 *
 * @example
 * @code
 * Parameter conductivity("k", 0.001, 100.0, "log-normal", 1.0);
 * double logProb = conductivity.CalcLogPriorProbability(0.5);
 * @endcode
 */
class Parameter
{
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * @brief Default constructor
     */
    Parameter();

    /**
     * @brief Constructor with full parameter specification
     * @param name Parameter name
     * @param low Lower bound of valid range
     * @param high Upper bound of valid range
     * @param dist Prior distribution type ("uniform", "normal", "log-normal")
     * @param val Initial value (if 0.0, uses midpoint of range)
     */
    Parameter(const std::string& name, double low, double high,
              const std::string& dist = "uniform", double val = 0.0);

    // ========================================================================
    // Basic Getters
    // ========================================================================

    /**
     * @brief Get parameter name
     * @return Parameter name
     */
    std::string GetName() const;

    /**
     * @brief Get prior distribution type
     * @return Distribution name ("uniform", "normal", "log-normal")
     */
    std::string GetPriorDistribution() const;

    /**
     * @brief Get current parameter value
     * @return Current value
     */
    double GetValue() const;

    /**
     * @brief Parameter range structure
     */
    struct Range {
        double low;   ///< Lower bound
        double high;  ///< Upper bound
    };

    /**
     * @brief Get valid parameter range
     * @return Range structure with low and high bounds
     */
    Range GetRange() const;

    /**
     * @brief Get mean of prior distribution
     * @return Prior mean (for normal/log-normal distributions)
     */
    double mean() const;

    /**
     * @brief Get standard deviation of prior distribution
     * @return Prior standard deviation (for normal/log-normal distributions)
     */
    double std() const;

    // ========================================================================
    // Setters
    // ========================================================================

    /**
     * @brief Set current parameter value
     * @param val New value
     *
     * @note Does not perform bounds checking. Use with caution or validate externally.
     */
    void SetValue(double val);

    // ========================================================================
    // Prior Probability Calculation
    // ========================================================================

    /**
     * @brief Calculate log prior probability at given value
     * @param val Value at which to evaluate prior
     * @return Log probability density
     *
     * For uniform distribution:
     *   - Returns -log(range) if val is within [low, high]
     *   - Returns -1e10 (very low probability) if outside range
     *
     * For normal distribution:
     *   - Returns log of Gaussian PDF: -0.5*(z^2) - log(σ*sqrt(2π))
     *   - where z = (val - μ) / σ
     *
     * For log-normal distribution:
     *   - Returns log of log-normal PDF
     *   - Returns -1e10 if val <= 0
     *
     * @example
     * @code
     * Parameter p("k", 1.0, 10.0, "uniform");
     * double logP = p.CalcLogPriorProbability(5.0);  // Within range
     * double logP_out = p.CalcLogPriorProbability(15.0);  // Outside range
     * @endcode
     */
    double CalcLogPriorProbability(double val) const;

    // ========================================================================
    // MCMC Post-Processing Storage
    // ========================================================================

    /**
     * @brief Store MCMC samples for this parameter
     * @param samples TimeSeriesSet containing samples from all chains
     *
     * Used to store the chain values after MCMC sampling is complete.
     */
    void SetMCMCSamples(const TimeSeriesSet<double>& samples);

    /**
     * @brief Store posterior distribution
     * @param dist TimeSeries representing the posterior density
     *
     * Typically computed as a histogram/kernel density estimate from MCMC samples.
     */
    void SetPosteriorDistribution(const TimeSeries<double>& dist);

    /**
     * @brief Store 95% credible interval
     * @param pct TimeSeriesSet containing percentile information
     *
     * Typically contains [2.5%, 50%, 97.5%] percentiles.
     */
    void SetPercentile95(const TimeSeriesSet<double>& pct);

private:
    // ========================================================================
    // Private Data Members
    // ========================================================================

    std::string paramName;              ///< Parameter name
    std::string priorDistribution;      ///< Prior distribution type
    double currentValue;                ///< Current parameter value
    Range range;                        ///< Valid parameter range
    double priorMean;                   ///< Mean for normal/log-normal prior
    double priorStd;                    ///< Std dev for normal/log-normal prior

    // MCMC results storage
    TimeSeriesSet<double> mcmcSamples;  ///< MCMC chain samples
    TimeSeries<double> posteriorDist;   ///< Posterior distribution
    TimeSeriesSet<double> percentile95; ///< Credible intervals
};

#endif // PARAMETER_H
