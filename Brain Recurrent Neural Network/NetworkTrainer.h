#pragma once
#include "NetworkParameters.h"

// Used for training the network. Backpropagation data is stored after every forwardpropagation.
class NetworkTrainer
{
private:
	NetworkParameters* parameters;

	vector<array<float, NetworkNodes>> interstates;				// Intermediate states of the network
	vector<array<float, NetworkNodes>> states;					// States of the network

	vector<array<float, InputNodes>>  inputs;					// Inputs to the network
	vector<array<float, OutputNodes>>  outputs;					// Outputs of the network
	vector<array<float, OutputNodes>>  expectedOutputs;			// Expected outputs of the network

	float initialStateGradientSum[NetworkNodes];
	float inputWeightGradientSums[NetworkNodes][InputNodes];
	float networkWeightGradientSums[NetworkNodes][NetworkNodes];
	float outputWeightGradientSums[OutputNodes][NetworkNodes];
	float networkBiasGradientSum[NetworkNodes];
	float outputBiasGradientSum[OutputNodes];

	uint64_t batchCount;
	uint64_t consecutiveOks;
	uint64_t currentIteration;

	// Restricts x between -1 and 1
	float Binary(float x)
	{
		return x < -1 ? -1 : x > 1 ? 1 : x;
	}

	// The gradient of Binary
	float BinaryGradient(float x)
	{
		return x < -1 ? 0 : x > 1 ? 0 : 1;
	}

	// The phantom gradient of Binary, it may promote the revitalization of a stale node
	float PhantomBinaryGradient(float x, float gradient)
	{
		return x < -1 ? (gradient < 0 ? 0 : gradient * PhantomCoefficient) : x > 1 ? (gradient > 0 ? 0 : gradient * PhantomCoefficient) : gradient;
	}

	// Clears all data related to the run
	void ResetRun()
	{
		interstates.clear();
		states.clear();
		inputs.clear();
		outputs.clear();
		expectedOutputs.clear();
		currentIteration = 0;
	}

	// Clears all data related to gradients
	void ResetGradients()
	{
		uint64_t node, childNode;

		for (node = 0; node < NetworkNodes; node++)
		{
			networkBiasGradientSum[node] = 0;
			initialStateGradientSum[node] = 0;

			for (childNode = 0; childNode < InputNodes; childNode++)
			{
				inputWeightGradientSums[node][childNode] = 0;
			}

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				networkWeightGradientSums[node][childNode] = 0;
			}
		}
		for (node = 0; node < OutputNodes; node++)
		{
			outputBiasGradientSum[node] = 0;

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				outputWeightGradientSums[node][childNode] = 0;
			}
		}

		batchCount = 0;
		consecutiveOks = 0;
	}

	void ApplyStableGradients()
	{
		uint64_t node, childNode;
		//cout << "Stable gradient reached after " << batchCount << " runs." << endl;
		for (node = 0; node < NetworkNodes; node++)
		{
			parameters->networkBias[node] += networkBiasGradientSum[node] / batchCount * LearningRate;
			parameters->initialState[node] += initialStateGradientSum[node] / batchCount * LearningRate;

			for (childNode = 0; childNode < InputNodes; childNode++)
			{
				parameters->inputWeights[node][childNode] += inputWeightGradientSums[node][childNode] / batchCount * LearningRate;
			}

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				parameters->networkWeights[node][childNode] += networkWeightGradientSums[node][childNode] / batchCount * LearningRate;
			}
		}
		for (node = 0; node < OutputNodes; node++)
		{
			parameters->outputBias[node] += outputBiasGradientSum[node] / batchCount * LearningRate;

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				parameters->outputWeights[node][childNode] += outputWeightGradientSums[node][childNode] / batchCount * LearningRate;
			}
		}
	}

public:
	NetworkTrainer(NetworkParameters* Parameters)
	{
		parameters = Parameters;
		ResetRun();
		ResetGradients();
	}

	// Propagates the input once through the network and calculates the output while storing the nessesary data for backpropagation. It is a single 'step' in a single 'run'.
	void ForwardPropagate(float* input, float* output, float* expectedOutput)
	{
		uint64_t node, childNode;
		float sum;
		array<float, NetworkNodes> interstateArray;
		array<float, NetworkNodes> stateArray;
		array<float, NetworkNodes> interstateGradientArray{};
		array<float, InputNodes> inputArray;
		array<float, OutputNodes> outputArray;
		array<float, OutputNodes> expectedOutputArray;

		// Adds the input to the input array for backpropagation
		for (node = 0; node < InputNodes; node++)
		{
			inputArray[node] = input[node];
		}
		inputs.push_back(inputArray);

		// Adds the expected output to the expected output array for backpropagation
		for (node = 0; node < OutputNodes; node++)
		{
			expectedOutputArray[node] = expectedOutput[node];
		}
		expectedOutputs.push_back(expectedOutputArray);

		// Use the stored state as the first state if just reset
		if (currentIteration != 0)
		{

			for (node = 0; node < NetworkNodes; node++)
			{
				sum = parameters->networkBias[node];

				for (childNode = 0; childNode < InputNodes; childNode++)
				{
					sum += parameters->inputWeights[node][childNode] * input[childNode];
				}

				// Uses the previous state to calculate the next state
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += parameters->networkWeights[node][childNode] * states[currentIteration - 1][childNode];
				}

				interstateArray[node] = sum;
				stateArray[node] = Binary(sum);
			}

			// Add the intermediate state to the intermediate states array for backpropagation
			interstates.push_back(interstateArray);

			// Add the state to the states array for backpropagation
			states.push_back(stateArray);

			for (node = 0; node < OutputNodes; node++)
			{
				sum = parameters->outputBias[node];

				// Uses the just computed state to calculate the output
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += parameters->outputWeights[node][childNode] * states[currentIteration][childNode];
				}

				outputArray[node] = sum;
				output[node] = sum;
			}

			// Add the output to the outputs array for backpropagation
			outputs.push_back(outputArray);
		}
		else
		{
			// Use the initial state as the first state if just reset
			for (node = 0; node < NetworkNodes; node++)
			{
				sum = parameters->networkBias[node];

				for (childNode = 0; childNode < InputNodes; childNode++)
				{
					sum += parameters->inputWeights[node][childNode] * input[childNode];
				}

				// Uses the initial state to calculate the next state
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += parameters->networkWeights[node][childNode] * parameters->initialState[childNode];
				}

				interstateArray[node] = sum;
				stateArray[node] = Binary(sum);
			}

			// Add the intermediate state to the intermediate states array for backpropagation
			interstates.push_back(interstateArray);

			// Add the state to the states array for backpropagation
			states.push_back(stateArray);

			for (node = 0; node < OutputNodes; node++)
			{
				sum = parameters->outputBias[node];

				// Uses the just computed state to calculate the output
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += parameters->outputWeights[node][childNode] * states[currentIteration][childNode];
				}

				outputArray[node] = sum;
				output[node] = sum;
			}

			// Add the output to the outputs array for backpropagation
			outputs.push_back(outputArray);
		}

		// Increment the iteration to define how many iterations have passed
		currentIteration++;
	}

	// Backpropagation is the same thing as reseting the entire 'run' and compiling changes. These gradients will not be applied until multiple backpropagations have been performed in which these gradients will be averaged and applied when stable.
	void BackPropagate()
	{
		float initialStateGradient[NetworkNodes]{};						// Initial state gradient
		float inputWeightGradients[NetworkNodes][InputNodes]{};			// Input weight gradients
		float networkWeightGradients[NetworkNodes][NetworkNodes]{};		// Network weight gradients
		float outputWeightGradients[OutputNodes][NetworkNodes]{};		// Output weight gradients
		float networkBiasGradient[NetworkNodes]{};						// Network bias gradient
		float outputBiasGradient[OutputNodes]{};						// Output bias gradient
		float outputGradient[OutputNodes];								// Output gradient
		float* interstateGradients = new float[NetworkNodes] {};		// Intermediate state gradient, stored as an address to optimize speed and memory usage when setting it to its 'previous' interstateGradient

		float standardDeviationSquared;										// The distance to the previous gradient averages to determine the average gradient to apply

		uint64_t layer, node, childNode;

		// loop for every iteration excluding the first one to calculate the gradients
		for (layer = currentIteration - 1; layer > 0; layer--)
		{
			float* previousInterstateGradient = new float[NetworkNodes] {};	// Previous intermediate state gradient

			// Calculate the output gradient as well as the output bias gradient
			for (node = 0; node < OutputNodes; node++)
			{
				outputGradient[node] = expectedOutputs[layer][node] - outputs[layer][node];
				outputBiasGradient[node] += outputGradient[node];
			}

			for (node = 0; node < NetworkNodes; node++)
			{
				// Calculate the output weight gradient as well as the intermediate state gradient
				for (childNode = 0; childNode < OutputNodes; childNode++)
				{
					interstateGradients[node] += parameters->outputWeights[childNode][node] * outputGradient[childNode];
					outputWeightGradients[childNode][node] += outputGradient[childNode] * states[layer][node];
				}

				// Calculate the network bias gradient as well as the intermediate state gradient
				//interstateGradients[node] *= BinaryGradient(interstates[layer][node]);
				interstateGradients[node] = PhantomBinaryGradient(interstates[layer][node], interstateGradients[node]);
				networkBiasGradient[node] += interstateGradients[node];

				// Calculate the network weight gradient
				for (childNode = 0; childNode < InputNodes; childNode++)
					inputWeightGradients[node][childNode] += inputs[layer][childNode] * interstateGradients[node];

				// Calculate the network weight gradient as well as the intermediate state gradient
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					previousInterstateGradient[childNode] += parameters->networkWeights[node][childNode] * interstateGradients[node];
					networkWeightGradients[node][childNode] += states[layer - 1][childNode] * interstateGradients[node];
				}
			}

			// delete the interstate gradient array and replace it with the 'previous' gradient (It is calculating the gradients from the last iteration to the first one)
			delete[] interstateGradients;
			interstateGradients = previousInterstateGradient;
		}

		// calculate the first iteration gradients using starting values, aka initial state
		for (node = 0; node < OutputNodes; node++)
		{
			outputGradient[node] = expectedOutputs[0][node] - outputs[0][node];
			outputBiasGradient[node] += outputGradient[node];
		}

		for (node = 0; node < NetworkNodes; node++)
		{
			for (childNode = 0; childNode < OutputNodes; childNode++)
			{
				interstateGradients[node] += parameters->outputWeights[childNode][node] * outputGradient[childNode];
				outputWeightGradients[childNode][node] += outputGradient[childNode] * states[0][node];
			}

			//interstateGradients[node] *= BinaryGradient(interstates[0][node]);
			interstateGradients[node] = PhantomBinaryGradient(interstates[0][node], interstateGradients[node]);
			networkBiasGradient[node] += interstateGradients[node];

			for (childNode = 0; childNode < InputNodes; childNode++)
				inputWeightGradients[node][childNode] += inputs[layer][childNode] * interstateGradients[node];

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				initialStateGradient[childNode] += parameters->networkWeights[node][childNode] * interstateGradients[node];
				networkWeightGradients[node][childNode] += parameters->initialState[childNode] * interstateGradients[node];
			}
		}

		delete[] interstateGradients;

		ResetRun();

		if (batchCount != 0)
		{
			standardDeviationSquared = 0;

			for (node = 0; node < NetworkNodes; node++)
			{
				standardDeviationSquared += pow((networkBiasGradientSum[node] + networkBiasGradient[node]) / (batchCount + 1) - (networkBiasGradientSum[node] / batchCount), 2);
				standardDeviationSquared += pow((initialStateGradientSum[node] + initialStateGradient[node]) / (batchCount + 1) - (initialStateGradientSum[node] / batchCount), 2);

				for (childNode = 0; childNode < InputNodes; childNode++)
					standardDeviationSquared += pow((inputWeightGradientSums[node][childNode] + inputWeightGradients[node][childNode]) / (batchCount + 1) - (inputWeightGradientSums[node][childNode] / batchCount), 2);

				for (childNode = 0; childNode < NetworkNodes; childNode++)
					standardDeviationSquared += pow((networkWeightGradientSums[node][childNode] + networkWeightGradients[node][childNode]) / (batchCount + 1) - (networkWeightGradientSums[node][childNode] / batchCount), 2);
			}
			for (node = 0; node < OutputNodes; node++)
			{
				standardDeviationSquared += pow((outputBiasGradientSum[node] + outputBiasGradient[node]) / (batchCount + 1) - (outputBiasGradientSum[node] / batchCount), 2);

				for (childNode = 0; childNode < NetworkNodes; childNode++)
					standardDeviationSquared += pow((outputWeightGradientSums[node][childNode] + outputWeightGradients[node][childNode]) / (batchCount + 1) - (outputWeightGradientSums[node][childNode] / batchCount), 2);
			}

			// Adds a count to the consecutive count if precision, else resets it.
			if (standardDeviationSquared < GradientPrecision * GradientPrecision)
			{
				consecutiveOks++;
			}
			else
			{
				consecutiveOks = 0;
			}
		}

		// Add all the gradients to the gradient sums.
		for (node = 0; node < NetworkNodes; node++)
		{
			networkBiasGradientSum[node] += networkBiasGradient[node];
			initialStateGradientSum[node] += initialStateGradient[node];
			for (childNode = 0; childNode < InputNodes; childNode++)
				inputWeightGradientSums[node][childNode] += inputWeightGradients[node][childNode];
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				networkWeightGradientSums[node][childNode] += networkWeightGradients[node][childNode];
		}
		for (node = 0; node < OutputNodes; node++)
		{
			outputBiasGradientSum[node] += outputBiasGradient[node];
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				outputWeightGradientSums[node][childNode] += outputWeightGradients[node][childNode];
		}

		// Increment the batchCount
		batchCount++;

		// If the consecutive count is greater than the required number of consecutive iterations, then we can apply the stable gradient and then reset the data related to the gradients.
		if (consecutiveOks >= RequiredConsecutiveOks)
		{
			ApplyStableGradients();
			ResetGradients();
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