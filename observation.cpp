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

#include "observation.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Constructors
// ============================================================================

Observation::Observation()
    : obsName("")
{
}

Observation::Observation(const std::string& name)
    : obsName(name)
{
}

// ============================================================================
// Basic Accessors
// ============================================================================

std::string Observation::GetName() const {
    return obsName;
}

void Observation::SetName(const std::string& name) {
    obsName = name;
}

// ============================================================================
// Data Management
// ============================================================================

void Observation::SetObservedTimeSeries(const TimeSeries<double>& ts) {
    observedData = ts;
}

void Observation::SetModeledTimeSeries(const TimeSeries<double>& ts) {
    modeledData = ts;
}

TimeSeries<double>* Observation::GetModeledTimeSeries() {
    return &modeledData;
}

const TimeSeries<double>* Observation::GetModeledTimeSeries() const {
    return &modeledData;
}

const TimeSeries<double>& Observation::GetObservedData() const {
    return observedData;
}

// ============================================================================
// Objective Function Calculations
// ============================================================================

double Observation::CalculateSSE() const {
    // Sum of squared errors
    double sse = 0.0;

    // Use minimum size to avoid out-of-bounds access
    int n = std::min(observedData.size(), modeledData.size());

    for (int i = 0; i < n; ++i) {
        double diff = observedData.getValue(i) - modeledData.getValue(i);
        sse += diff * diff;
    }

    return sse;
}

double Observation::CalculateRMSE() const {
    int n = std::min(observedData.size(), modeledData.size());

    if (n == 0) {
        return 0.0;
    }

    double sse = CalculateSSE();
    return std::sqrt(sse / static_cast<double>(n));
}

double Observation::CalculateMAE() const {
    double mae = 0.0;

    int n = std::min(observedData.size(), modeledData.size());

    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < n; ++i) {
        double diff = observedData.getValue(i) - modeledData.getValue(i);
        mae += std::abs(diff);
    }

    return mae / static_cast<double>(n);
}

double Observation::CalculateNSE() const {
    int n = std::min(observedData.size(), modeledData.size());

    if (n == 0) {
        return 0.0;
    }

    // Calculate mean of observed values
    double obs_mean = 0.0;
    for (int i = 0; i < n; ++i) {
        obs_mean += observedData.getValue(i);
    }
    obs_mean /= static_cast<double>(n);

    // Calculate SSE and total sum of squares
    double sse = 0.0;
    double sst = 0.0;

    for (int i = 0; i < n; ++i) {
        double obs = observedData.getValue(i);
        double mod = modeledData.getValue(i);

        double diff = obs - mod;
        sse += diff * diff;

        double dev = obs - obs_mean;
        sst += dev * dev;
    }

    // Avoid division by zero
    if (sst == 0.0) {
        return 0.0;
    }

    // NSE = 1 - SSE/SST
    return 1.0 - (sse / sst);
}

// ============================================================================
// MCMC Post-Processing Storage
// ============================================================================

void Observation::SetRealizations(const TimeSeriesSet<double>& real) {
    realizations = real;
}

void Observation::SetPercentile95(const TimeSeriesSet<double>& pct) {
    percentile95 = pct;
}

const TimeSeriesSet<double>& Observation::GetRealizations() const {
    return realizations;
}

const TimeSeriesSet<double>& Observation::GetPercentile95() const {
    return percentile95;
}
