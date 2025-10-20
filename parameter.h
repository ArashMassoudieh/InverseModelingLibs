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
     * @brief Set parameter name
     * @param name New parameter name
     */
    void SetName(const std::string& name);

    /**
     * @brief Set parameter range
     * @param low Lower bound
     * @param high Upper bound
     */
    void SetRange(double low, double high);

    /**
     * @brief Set parameter range using Range struct
     * @param r Range structure
     */
    void SetRange(const Range& r);

    void SetLow(const double& r);
    void SetHigh(const double& r);

    /**
     * @brief Get locations this parameter applies to
     * @return Vector of location names (wells or tracers)
     */
    const std::vector<std::string>& GetLocations() const;

    /**
     * @brief Get quantities this parameter represents at each location
     * @return Vector of quantity names
     */
    const std::vector<std::string>& GetQuantities() const;

    /**
     * @brief Get location types
     * @return Vector of location type identifiers
     */
    const std::vector<std::string>& GetLocationTypes() const;

    /**
     * @brief Set locations this parameter applies to
     * @param locs Vector of location names
     */
    void SetLocations(const std::vector<std::string>& locs);

    /**
     * @brief Set quantities this parameter represents
     * @param quants Vector of quantity names
     */
    void SetQuantities(const std::vector<std::string>& quants);

    /**
     * @brief Set location types
     * @param types Vector of location types
     */
    void SetLocationTypes(const std::vector<std::string>& types);

    /**
     * @brief Add a single location-quantity-type mapping
     * @param location Location name
     * @param quantity Quantity name
     * @param type Location type
     */
    void AddLocation(const std::string& location, const std::string& quantity,
                     const std::string& type);

    /**
     * @brief Set prior distribution type
     * @param dist Distribution name ("uniform", "normal", "log-normal")
     */
    void SetPriorDistribution(const std::string& dist);

    /**
     * @brief Set prior mean and standard deviation
     * @param mean Prior mean
     * @param std Prior standard deviation
     */
    void SetPriorParameters(double mean, double std);

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

    /**
 * @brief Remove a specific location/quantity/type combination
 * @param location Location name to remove
 * @param quantity Quantity associated with this location
 * @param locationType Type of location (e.g., "well", "tracer", "0", "1")
 * @return true if found and removed, false otherwise
 */
    bool RemoveLocation(const std::string& location,
                        const std::string& quantity,
                        const std::string& locationType);

    /**
 * @brief Remove all occurrences of a location with specific type
 * @param location Location name to remove
 * @param locationType Type of location to match
 * @return Number of entries removed
 */
    int RemoveAllLocations(const std::string& location,
                           const std::string& locationType);
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

    std::vector<std::string> locations_;      ///< Locations this parameter applies to
    std::vector<std::string> quantities_;     ///< Quantities at each location
    std::vector<std::string> locationTypes_;  ///< Location types (e.g., "well", "tracer")
};

#endif // PARAMETER_H
