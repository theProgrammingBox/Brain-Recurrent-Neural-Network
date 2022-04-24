#pragma once
#include "NetworkParameters.h"

// Used just for efficiently running the network, no backpropagation done here
class Network
{
private:
	NetworkParameters* parameters;
	float* state = new float[NetworkNodes];				// State of the network, stored as an address to optimize speed and memory usage when setting it to its next state

	// Restricts x between -1 and 1
	float Binary(float x)
	{
		return x < -1 ? -1 : x > 1 ? 1 : x;
	}

public:
	Network(NetworkParameters* Parameters)
	{
		parameters = Parameters;
		Reset();
	}

	// Resets the network state to its initial state
	void Reset()
	{
		uint64_t node;
		for (node = 0; node < NetworkNodes; node++)
			state[node] = parameters->initialState[node];
	}

	// Propagates the input once through the network and calculates the output
	void ForwardPropagate(float* input, float* output)
	{
		uint64_t node, childNode;
		float sum;
		float* nextState = new float[NetworkNodes];

		for (node = 0; node < NetworkNodes; node++)
		{
			sum = parameters->networkBias[node];

			for (childNode = 0; childNode < InputNodes; childNode++)
				sum += parameters->inputWeights[node][childNode] * input[childNode];

			for (childNode = 0; childNode < NetworkNodes; childNode++)
				sum += parameters->networkWeights[node][childNode] * state[childNode];

			nextState[node] = Binary(sum);
		}

		// Delete the old state and set the new state
		delete[] state;
		state = nextState;

		for (node = 0; node < OutputNodes; node++)
		{
			sum = parameters->outputBias[node];

			for (childNode = 0; childNode < NetworkNodes; childNode++)
				sum += parameters->outputWeights[node][childNode] * state[childNode];

			output[node] = sum;
		}
	}

	// Exports the network to network.txt
	void Export()
	{
		parameters->Export();
	}

	// Imports the network from network.txt
	void Import()
	{
		parameters->Import();
	}
};