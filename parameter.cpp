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

#include "parameter.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Constructors
// ============================================================================

Parameter::Parameter()
    : paramName("")
    , priorDistribution("uniform")
    , currentValue(0.0)
    , priorMean(0.0)
    , priorStd(1.0)
{
    range.low = 0.0;
    range.high = 1.0;
}

Parameter::Parameter(const std::string& name, double low, double high,
                     const std::string& dist, double val)
    : paramName(name)
    , priorDistribution(dist)
    , currentValue(val)
{
    // Set range
    range.low = low;
    range.high = high;

    // Set default value if not provided
    if (val == 0.0) {
        currentValue = (low + high) / 2.0;
    }

    // For normal/log-normal, set mean and std based on range
    // Assumes 95% of distribution is within [low, high]
    if (dist == "normal" || dist == "log-normal") {
        priorMean = (low + high) / 2.0;
        priorStd = (high - low) / 4.0;  // ±2σ covers ~95% of normal distribution
    }
    else {
        priorMean = 0.0;
        priorStd = 1.0;
    }
}

// ============================================================================
// Getters
// ============================================================================

std::string Parameter::GetName() const {
    return paramName;
}

std::string Parameter::GetPriorDistribution() const {
    return priorDistribution;
}

double Parameter::GetValue() const {
    return currentValue;
}

Parameter::Range Parameter::GetRange() const {
    return range;
}

double Parameter::mean() const {
    return priorMean;
}

double Parameter::std() const {
    return priorStd;
}

// ============================================================================
// Setters
// ============================================================================

void Parameter::SetValue(double val) {
    currentValue = val;
}

// ============================================================================
// Prior Probability Calculation
// ============================================================================

double Parameter::CalcLogPriorProbability(double val) const {
    if (priorDistribution == "uniform") {
        // Uniform distribution: constant density within range
        if (val >= range.low && val <= range.high) {
            return -std::log(range.high - range.low);
        }
        return -1e10;  // Very low probability outside range
    }
    else if (priorDistribution == "normal") {
        // Normal distribution: log of Gaussian PDF
        // log(PDF) = -0.5 * z^2 - log(σ * sqrt(2π))
        double z = (val - priorMean) / priorStd;
        return -0.5 * z * z - std::log(priorStd * std::sqrt(2.0 * M_PI));
    }
    else if (priorDistribution == "log-normal") {
        // Log-normal distribution
        // log(PDF) = -0.5 * [(ln(x) - μ) / σ]^2 - ln(x) - ln(σ) - ln(sqrt(2π))
        if (val <= 0) {
            return -1e10;  // Log-normal only defined for positive values
        }

        double z = (std::log(val) - std::log(priorMean)) / priorStd;
        return -0.5 * z * z - std::log(val * priorStd * std::sqrt(2.0 * M_PI));
    }

    // Unknown distribution type - return zero (uniform on log scale)
    return 0.0;
}

// ============================================================================
// MCMC Post-Processing Storage
// ============================================================================

void Parameter::SetMCMCSamples(const TimeSeriesSet<double>& samples) {
    mcmcSamples = samples;
}

void Parameter::SetPosteriorDistribution(const TimeSeries<double>& dist) {
    posteriorDist = dist;
}

void Parameter::SetPercentile95(const TimeSeriesSet<double>& pct) {
    percentile95 = pct;
}

void Parameter::SetRange(double low, double high) {
    range.low = low;
    range.high = high;
}

void Parameter::SetName(const std::string& name) {
    paramName = name;
}

void Parameter::SetRange(const Range& r) {
    range = r;
}

void Parameter::SetLow(const double& r) {
    range.low = r;
}

void Parameter::SetHigh(const double& r) {
    range.high = r;
}

void Parameter::SetPriorDistribution(const std::string& dist) {
    priorDistribution = dist;
}

void Parameter::SetPriorParameters(double mean, double std) {
    priorMean = mean;
    priorStd = std;
}

// ============================================================================
// Location/Quantity Getters
// ============================================================================

const std::vector<std::string>& Parameter::GetLocations() const {
    return locations_;
}

const std::vector<std::string>& Parameter::GetQuantities() const {
    return quantities_;
}

const std::vector<std::string>& Parameter::GetLocationTypes() const {
    return locationTypes_;
}

// ============================================================================
// Location/Quantity Setters
// ============================================================================

void Parameter::SetLocations(const std::vector<std::string>& locs) {
    locations_ = locs;
}

void Parameter::SetQuantities(const std::vector<std::string>& quants) {
    quantities_ = quants;
}

void Parameter::SetLocationTypes(const std::vector<std::string>& types) {
    locationTypes_ = types;
}

void Parameter::AddLocation(const std::string& location, const std::string& quantity,
                            const std::string& type) {
    locations_.push_back(location);
    quantities_.push_back(quantity);
    locationTypes_.push_back(type);
}

bool Parameter::RemoveLocation(const std::string& location,
                               const std::string& quantity,
                               const std::string& locationType)
{
    // Find matching entry
    for (size_t i = 0; i < locations_.size(); ++i) {
        bool locationMatch = (locations_[i] == location);
        bool quantityMatch = (i < quantities_.size() && quantities_[i] == quantity);
        bool typeMatch = (i < locationTypes_.size() && locationTypes_[i] == locationType);

        if (locationMatch && quantityMatch && typeMatch) {
            // Remove this entry
            locations_.erase(locations_.begin() + i);
            if (i < quantities_.size()) {
                quantities_.erase(quantities_.begin() + i);
            }
            if (i < locationTypes_.size()) {
                locationTypes_.erase(locationTypes_.begin() + i);
            }
            return true;
        }
    }

    return false;  // Not found
}

int Parameter::RemoveAllLocations(const std::string& location,
                                  const std::string& locationType)
{
    int removedCount = 0;

    // Iterate backwards to safely remove elements
    for (int i = locations_.size() - 1; i >= 0; --i) {
        bool locationMatch = (locations_[i] == location);
        bool typeMatch = (static_cast<size_t>(i) < locationTypes_.size() &&
                          locationTypes_[i] == locationType);

        if (locationMatch && typeMatch) {
            locations_.erase(locations_.begin() + i);
            if (static_cast<size_t>(i) < quantities_.size()) {
                quantities_.erase(quantities_.begin() + i);
            }
            if (static_cast<size_t>(i) < locationTypes_.size()) {
                locationTypes_.erase(locationTypes_.begin() + i);
            }
            removedCount++;
        }
    }

    return removedCount;
}
