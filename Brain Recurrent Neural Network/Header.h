#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <math.h>
#include <fstream>
#include <string>
#include "Randoms.h"

using namespace std;

const uint64_t InputNodes = 2;				// Number of input nodes - must stay the same when importing the network
const uint64_t NetworkNodes = 2;			// Number of nodes in the network - must stay the same when importing the network
const uint64_t OutputNodes = 2;				// Number of output nodes - must stay the same when importing the network

uint64_t RequiredConsecutiveOks = 1;	// Number of consecutive oks required to apply the averaged gradient - doesn't need to stay the same when importing the network
float GradientPrecision = 1;			// The precision of the gradient for it to be considered ok - doesn't need to stay the same when importing the network
float LearningRate = 0.01;				// The learning rate / gradient step - doesn't need to stay the same when importing the network
float PhantomCoefficient = 0.1;			// The percent of the gradient applied when the node is stale - doesn't need to stay the same when importing the network

// Data gathering Parameters
const uint64_t NumRuns = 12;					// Number of runs per parameter combination
const uint64_t NumTrainingPerRuns = 100;		// Number of error saples per run
const uint64_t NumTrainingBeforeSample = 1000;	// Number of training runs between each sample

const uint64_t NumRequiredConsecutiveOks = 4;	// Number of consecutive oks parameters we will be testing
const uint64_t NumGradientPrecision = 4;		// Number of precision parameters we will be testing
const uint64_t NumLearningRate = 4;				// Number of learning rate parameters we will be testing
const uint64_t NumPhantomCoefficient = 4;		// Number of phantom coefficient parameters we will be testing

const uint64_t RequiredConsecutiveOksArr[NumRequiredConsecutiveOks] = { 10, 6, 3, 1 };	// All the required consecutive ok parameters we will be testing
const float GradientPrecisionArr[NumGradientPrecision] = { 1, 0.5, 0.1, 0.05 };			// All the precision parameters we will be testing
const float LearningRateArr[NumLearningRate] = { 0.05, 0.01, 0.005, 0.001 };			// All the learning rate parameters we will be testing
const float PhantomCoefficientArr[NumPhantomCoefficient] = { 0, 0.1, 0.4, 1.0 };		// All the phantom coefficient parameters we will be testing

const uint64_t NumItems = NumRequiredConsecutiveOks * NumGradientPrecision * NumLearningRate * NumPhantomCoefficient;	// The total number of combinations of parameters we will be testing
