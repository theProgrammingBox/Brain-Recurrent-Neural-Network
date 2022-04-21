#pragma once
#include <iostream>
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
const uint64_t RequiredConsecutiveOks = 10; // Number of consecutive oks required to apply the averaged gradient - doesn't need to stay the same when importing the network
const float GradientPrecision = 0.1;		// The precision of the gradient for it to be considered ok - doesn't need to stay the same when importing the network
const float LearningRate = 0.01;			// The learning rate / gradient step - doesn't need to stay the same when importing the network