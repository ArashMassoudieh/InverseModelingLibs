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

#ifndef OBSERVATION_H
#define OBSERVATION_H

#include <string>
#include "TimeSeries.h"
#include "TimeSeriesSet.h"

/**
 * @brief Observation data and model predictions for calibration
 *
 * Stores both observed field/lab data and corresponding model predictions
 * for a single observable quantity. Used in model calibration to compute
 * objective functions and in MCMC for likelihood calculations.
 *
 * Can also store post-processing results such as prediction intervals
 * and ensemble realizations from uncertainty analysis.
 *
 * @example
 * @code
 * Observation temp_obs("Temperature");
 *
 * // Set observed data
 * TimeSeries<double> observed;
 * observed.append(0.0, 20.5);
 * observed.append(1.0, 21.2);
 * temp_obs.SetObservedTimeSeries(observed);
 *
 * // After running model
 * TimeSeries<double> modeled = model.GetTemperaturePrediction();
 * temp_obs.SetModeledTimeSeries(modeled);
 *
 * // Calculate fit
 * double sse = temp_obs.CalculateSSE();
 * @endcode
 */
class Observation
{
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * @brief Default constructor
     */
    Observation();

    /**
     * @brief Constructor with name
     * @param name Observation name/identifier
     *
     * Creates an observation with the given name and empty time series.
     */
    explicit Observation(const std::string& name);

    // ========================================================================
    // Basic Accessors
    // ========================================================================

    /**
     * @brief Get observation name
     * @return Observation name
     */
    std::string GetName() const;

    /**
     * @brief Set observation name
     * @param name New observation name
     */
    void SetName(const std::string& name);

    // ========================================================================
    // Data Management
    // ========================================================================

    /**
     * @brief Set observed (measured) data
     * @param ts TimeSeries containing observed values
     *
     * This is typically field or laboratory measurement data that
     * the model is trying to reproduce.
     */
    void SetObservedTimeSeries(const TimeSeries<double>& ts);

    /**
     * @brief Set model-predicted data
     * @param ts TimeSeries containing model predictions
     *
     * This is the model output corresponding to the observed data.
     * Should have values at the same times as observed data for
     * proper comparison.
     */
    void SetModeledTimeSeries(const TimeSeries<double>& ts);

    /**
     * @brief Get pointer to modeled time series (mutable)
     * @return Pointer to modeled TimeSeries
     *
     * Allows direct modification of the modeled data.
     */
    TimeSeries<double>* GetModeledTimeSeries();

    /**
     * @brief Get pointer to modeled time series (const)
     * @return Const pointer to modeled TimeSeries
     */
    const TimeSeries<double>* GetModeledTimeSeries() const;

    /**
     * @brief Get observed time series (const)
     * @return Const reference to observed TimeSeries
     */
    const TimeSeries<double>& GetObservedData() const;

    // ========================================================================
    // Objective Function Calculations
    // ========================================================================

    /**
     * @brief Calculate sum of squared errors
     * @return SSE = Σ(observed - modeled)²
     *
     * Computes the sum of squared differences between observed and
     * modeled values at all time points. Common objective function
     * for least-squares calibration.
     *
     * Only considers points where both observed and modeled data exist.
     */
    double CalculateSSE() const;

    /**
     * @brief Calculate root mean squared error
     * @return RMSE = sqrt(SSE / n)
     *
     * Square root of mean squared error, in same units as observations.
     */
    double CalculateRMSE() const;

    /**
     * @brief Calculate mean absolute error
     * @return MAE = Σ|observed - modeled| / n
     */
    double CalculateMAE() const;

    /**
     * @brief Calculate Nash-Sutcliffe efficiency
     * @return NSE ∈ (-∞, 1], where 1 is perfect fit
     *
     * NSE = 1 - SSE / Σ(observed - mean(observed))²
     */
    double CalculateNSE() const;

    // ========================================================================
    // MCMC Post-Processing Storage
    // ========================================================================

    /**
     * @brief Store ensemble realizations from uncertainty analysis
     * @param real TimeSeriesSet with multiple model realizations
     *
     * Each time series in the set represents one realization using
     * parameters sampled from the posterior distribution.
     */
    void SetRealizations(const TimeSeriesSet<double>& real);

    /**
     * @brief Store prediction interval percentiles
     * @param pct TimeSeriesSet with percentile time series
     *
     * Typically contains 2.5%, 50% (median), and 97.5% percentiles
     * to represent 95% prediction intervals.
     */
    void SetPercentile95(const TimeSeriesSet<double>& pct);

    /**
     * @brief Get stored realizations
     * @return TimeSeriesSet with all realizations
     */
    const TimeSeriesSet<double>& GetRealizations() const;

    /**
     * @brief Get stored percentiles
     * @return TimeSeriesSet with percentile time series
     */
    const TimeSeriesSet<double>& GetPercentile95() const;

private:
    // ========================================================================
    // Private Data Members
    // ========================================================================

    std::string obsName;                    ///< Observation identifier
    TimeSeries<double> observedData;        ///< Measured/observed data
    TimeSeries<double> modeledData;         ///< Model predictions
    TimeSeriesSet<double> realizations;     ///< Ensemble realizations
    TimeSeriesSet<double> percentile95;     ///< Prediction intervals
};

#endif // OBSERVATION_H
