#pragma once
#include "header.h"

// Used for training the network. Backpropagation data is stored after every forwardpropagation.
class NetworkTrainer
{
private:
	float initialState[NetworkNodes];							// Initial state of the network
	float inputWeights[NetworkNodes][InputNodes];				// Input weights
	float networkWeights[NetworkNodes][NetworkNodes];			// Network weights
	float outputWeights[OutputNodes][NetworkNodes];				// Output weights
	float networkBias[NetworkNodes];							// Network bias
	float outputBias[OutputNodes];								// Output bias

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

public:
	NetworkTrainer()
	{
		Initialize();
	}

	// Initializes the network with random weights and zeros the parameter gradients
	void Initialize()
	{
		uint64_t node, childNode;
		Random random = Random();

		interstates.clear();
		states.clear();
		inputs.clear();
		outputs.clear();
		expectedOutputs.clear();

		for (node = 0; node < NetworkNodes; node++)
		{
			networkBias[node] = random.DoubleRandom();
			networkBiasGradientSum[node] = 0;

			initialState[node] = random.DoubleRandom();
			initialStateGradientSum[node] = 0;

			for (childNode = 0; childNode < InputNodes; childNode++)
			{
				inputWeights[node][childNode] = random.DoubleRandom();
				inputWeightGradientSums[node][childNode] = 0;
			}

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				networkWeights[node][childNode] = random.DoubleRandom();
				networkWeightGradientSums[node][childNode] = 0;
			}
		}
		for (node = 0; node < OutputNodes; node++)
		{
			outputBias[node] = random.DoubleRandom();
			outputBiasGradientSum[node] = 0;

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				outputWeights[node][childNode] = random.DoubleRandom();
				outputWeightGradientSums[node][childNode] = 0;
			}
		}

		batchCount = 0;
		consecutiveOks = 0;
		currentIteration = 0;
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
		//cout << "Input array: ";
		for (node = 0; node < InputNodes; node++)
		{
			inputArray[node] = input[node];
			//cout << input[node] << " ";
		}
		inputs.push_back(inputArray);
		//cout << "\n\n";

		// Adds the expected output to the expected output array for backpropagation
		//cout << "Expected output array: ";
		for (node = 0; node < OutputNodes; node++)
		{
			expectedOutputArray[node] = expectedOutput[node];
			//cout << expectedOutput[node] << " ";
		}
		expectedOutputs.push_back(expectedOutputArray);
		//cout << "\n\n";

		// Use the stored state as the first state if just reset
		if (currentIteration != 0)
		{
			//cout << "Iteration: " << currentIteration << endl;

			for (node = 0; node < NetworkNodes; node++)
			{
				//cout << "Value for node " << node << " is " << networkBias[node] << endl;
				//cout << "----------------------\n";
				sum = networkBias[node];

				for (childNode = 0; childNode < InputNodes; childNode++)
				{
					sum += inputWeights[node][childNode] * input[childNode];
					//cout << "+ " << inputWeights[node][childNode] << " * " << input[childNode] << " = " << sum << endl;
				}
				//cout << "----------------------\n";

				// Uses the previous state to calculate the next state
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += networkWeights[node][childNode] * states[currentIteration - 1][childNode];
					//cout << "+ " << networkWeights[node][childNode] << " * " << states[currentIteration - 1][childNode] << " = " << sum << endl;
				}
				//cout << "----------------------\n";

				interstateArray[node] = sum;
				//cout << "The Intermediate State for node " << node << " is " << sum << endl;
				stateArray[node] = Binary(sum);
				//cout << "The State for node " << node << " is " << stateArray[node] << "\n\n\n";
			}

			// Add the intermediate state to the intermediate states array for backpropagation
			interstates.push_back(interstateArray);

			// Add the state to the states array for backpropagation
			states.push_back(stateArray);

			for (node = 0; node < OutputNodes; node++)
			{
				sum = outputBias[node];
				//cout << "Value for output node " << node << " is " << outputBias[node] << endl;
				//cout << "----------------------\n";

				// Uses the just computed state to calculate the output
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += outputWeights[node][childNode] * states[currentIteration][childNode];
					//cout << "+ " << outputWeights[node][childNode] << " * " << states[currentIteration][childNode] << " = " << sum << endl;
				}
				//cout << "----------------------\n";

				outputArray[node] = sum;
				output[node] = sum;
				//cout << "The Output for node " << node << " is " << outputArray[node] << "\n\n\n";
			}

			// Add the output to the outputs array for backpropagation
			outputs.push_back(outputArray);
		}
		else
		{
			//cout << "First iteration\n";

			// Use the initial state as the first state if just reset
			for (node = 0; node < NetworkNodes; node++)
			{
				//cout << "Value for node " << node << " is " << networkBias[node] << endl;
				//cout << "----------------------\n";
				sum = networkBias[node];

				for (childNode = 0; childNode < InputNodes; childNode++)
				{
					sum += inputWeights[node][childNode] * input[childNode];
					//cout << "+ " << inputWeights[node][childNode] << " * " << input[childNode] << " = " << sum << endl;
				}
				//cout << "----------------------\n";

				// Uses the initial state to calculate the next state
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += networkWeights[node][childNode] * initialState[childNode];
					//cout << "+ " << networkWeights[node][childNode] << " * " << initialState[childNode] << " = " << sum << endl;
				}
				//cout << "----------------------\n";

				interstateArray[node] = sum;
				//cout << "The Intermediate State for node " << node << " is " << sum << endl;
				stateArray[node] = Binary(sum);
				//cout << "The State for node " << node << " is " << stateArray[node] << "\n\n\n";
			}

			// Add the intermediate state to the intermediate states array for backpropagation
			interstates.push_back(interstateArray);

			// Add the state to the states array for backpropagation
			states.push_back(stateArray);

			for (node = 0; node < OutputNodes; node++)
			{
				sum = outputBias[node];
				//cout << "Value for output node " << node << " is " << outputBias[node] << endl;
				//cout << "----------------------\n";

				// Uses the just computed state to calculate the output
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					sum += outputWeights[node][childNode] * states[currentIteration][childNode];
					//cout << "+ " << outputWeights[node][childNode] << " * " << states[currentIteration][childNode] << " = " << sum << endl;
				}
				//cout << "----------------------\n";

				outputArray[node] = sum;
				output[node] = sum;
				//cout << "The Output for node " << node << " is " << outputArray[node] << "\n\n\n";
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

		float standardDeviation;										// The distance to the previous gradient averages to determine the average gradient to apply

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
					interstateGradients[node] += outputWeights[childNode][node] * outputGradient[childNode];
					outputWeightGradients[childNode][node] += outputGradient[childNode] * states[layer][node];
				}

				// Calculate the network bias gradient as well as the intermediate state gradient
				interstateGradients[node] *= BinaryGradient(interstates[layer][node]);
				networkBiasGradient[node] += interstateGradients[node];

				// Calculate the network weight gradient
				for (childNode = 0; childNode < InputNodes; childNode++)
					inputWeightGradients[node][childNode] += inputs[layer][childNode] * interstateGradients[node];

				// Calculate the network weight gradient as well as the intermediate state gradient
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					previousInterstateGradient[childNode] += networkWeights[node][childNode] * interstateGradients[node];
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
				interstateGradients[node] += outputWeights[childNode][node] * outputGradient[childNode];
				outputWeightGradients[childNode][node] += outputGradient[childNode] * states[0][node];
			}

			interstateGradients[node] *= BinaryGradient(interstates[0][node]);
			networkBiasGradient[node] += interstateGradients[node];

			for (childNode = 0; childNode < InputNodes; childNode++)
				inputWeightGradients[node][childNode] += inputs[layer][childNode] * interstateGradients[node];

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				initialStateGradient[childNode] += networkWeights[node][childNode] * interstateGradients[node];
				networkWeightGradients[node][childNode] += initialState[childNode] * interstateGradients[node];
			}
		}

		delete[] interstateGradients;

		// Clears the backpropagation data for the next run
		interstates.clear();
		states.clear();
		inputs.clear();
		outputs.clear();
		expectedOutputs.clear();

		// Reset the iteration.
		currentIteration = 0;

		if (batchCount != 0)
		{
			standardDeviation = 0;

			for (node = 0; node < NetworkNodes; node++)
			{
				standardDeviation += pow((networkBiasGradientSum[node] + networkBiasGradient[node]) / (batchCount + 1) - (networkBiasGradientSum[node] / batchCount), 2);
				standardDeviation += pow((initialStateGradientSum[node] + initialStateGradient[node]) / (batchCount + 1) - (initialStateGradientSum[node] / batchCount), 2);

				for (childNode = 0; childNode < InputNodes; childNode++)
					standardDeviation += pow((inputWeightGradientSums[node][childNode] + inputWeightGradients[node][childNode]) / (batchCount + 1) - (inputWeightGradientSums[node][childNode] / batchCount), 2);

				for (childNode = 0; childNode < NetworkNodes; childNode++)
					standardDeviation += pow((networkWeightGradientSums[node][childNode] + networkWeightGradients[node][childNode]) / (batchCount + 1) - (networkWeightGradientSums[node][childNode] / batchCount), 2);
			}
			for (node = 0; node < OutputNodes; node++)
			{
				standardDeviation += pow((outputBiasGradientSum[node] + outputBiasGradient[node]) / (batchCount + 1) - (outputBiasGradientSum[node] / batchCount), 2);

				for (childNode = 0; childNode < NetworkNodes; childNode++)
					standardDeviation += pow((outputWeightGradientSums[node][childNode] + outputWeightGradients[node][childNode]) / (batchCount + 1) - (outputWeightGradientSums[node][childNode] / batchCount), 2);
			}

			// sqrt(standardDeviation), distance equation, but we square both sides for optimization. Adds a count to the consecutive count if passes, else resets it.
			if (standardDeviation < GradientPrecision * GradientPrecision)
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
		if (consecutiveOks == RequiredConsecutiveOks)
		{
			//cout << "Stable gradient reached after " << batchCount << " iterations." << endl;
			for (node = 0; node < NetworkNodes; node++)
			{
				networkBias[node] += networkBiasGradientSum[node] / batchCount * LearningRate;
				networkBiasGradientSum[node] = 0;

				initialState[node] += initialStateGradientSum[node] / batchCount * LearningRate;
				initialStateGradientSum[node] = 0;

				for (childNode = 0; childNode < InputNodes; childNode++)
				{
					inputWeights[node][childNode] += inputWeightGradientSums[node][childNode] / batchCount * LearningRate;
					inputWeightGradientSums[node][childNode] = 0;
				}

				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					networkWeights[node][childNode] += networkWeightGradientSums[node][childNode] / batchCount * LearningRate;
					networkWeightGradientSums[node][childNode] = 0;
				}
			}
			for (node = 0; node < OutputNodes; node++)
			{
				outputBias[node] += outputBiasGradientSum[node] / batchCount * LearningRate;
				outputBiasGradientSum[node] = 0;

				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					outputWeights[node][childNode] += outputWeightGradientSums[node][childNode] / batchCount * LearningRate;
					outputWeightGradientSums[node][childNode] = 0;
				}
			}

			batchCount = 0;
			consecutiveOks = 0;
		}
	}

	// Exports the network to network.txt
	void Export()
	{
		uint64_t node, childNode;
		ofstream file;
		file.open("network.txt");
		file << "InputNodes " << InputNodes << "\n\n";
		file << "NetworkNodes " << NetworkNodes << "\n\n";
		file << "OutputNodes " << OutputNodes << "\n\n";
		file << "InitialState" << endl;
		for (node = 0; node < NetworkNodes; node++)
			file << initialState[node] << " ";
		file << "\n\n";
		file << "InputWeights" << endl;
		for (node = 0; node < NetworkNodes; node++)
		{
			for (childNode = 0; childNode < InputNodes; childNode++)
				file << inputWeights[node][childNode] << " ";
			file << endl;
		}
		file << endl;
		file << "NetworkWeights" << endl;
		for (node = 0; node < NetworkNodes; node++)
		{
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				file << networkWeights[node][childNode] << " ";
			file << endl;
		}
		file << endl;
		file << "OutputWeights" << endl;
		for (node = 0; node < OutputNodes; node++)
		{
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				file << outputWeights[node][childNode] << " ";
			file << endl;
		}
		file << endl;
		file << "NetworkBias" << endl;
		for (node = 0; node < NetworkNodes; node++)
			file << networkBias[node] << " ";
		file << "\n\n";
		file << "OutputBias" << endl;
		for (node = 0; node < OutputNodes; node++)
			file << outputBias[node] << " ";
		file << "\n\n";
		file.close();
	}

	// Imports the network from network.txt
	void Import()
	{
		string temp;
		uint64_t node, childNode, tempParam;
		ifstream file;
		file.open("network.txt");
		if (file.peek() == ifstream::traits_type::eof())
		{
			Initialize();
			cout << "Importing network..." << endl;
		}
		else
		{
			file >> temp;
			file >> tempParam;
			if (temp != "InputNodes" || tempParam != InputNodes)
				throw "Protocol error";
			if (tempParam != InputNodes)
				throw "InputNodes mismatch";
			cout << "InputNodes " << InputNodes << "\n\n";
			file >> temp;
			file >> tempParam;
			if (temp != "NetworkNodes" || tempParam != NetworkNodes)
				throw "Protocol error";
			if (tempParam != NetworkNodes)
				throw "NetworkNodes mismatch";
			cout << "NetworkNodes " << NetworkNodes << "\n\n";
			file >> temp;
			file >> tempParam;
			if (temp != "OutputNodes" || tempParam != OutputNodes)
				throw "Protocol error";
			if (tempParam != OutputNodes)
				throw "OutputNodes mismatch";
			cout << "OutputNodes " << OutputNodes << "\n\n";
			file >> temp;
			if (temp != "InitialState")
				throw "Protocol error";
			cout << "InitialState" << endl;
			for (node = 0; node < NetworkNodes; node++)
			{
				file >> initialState[node];
				cout << initialState[node] << " ";
			}
			cout << "\n\n";
			file >> temp;
			if (temp != "InputWeights")
				throw "Protocol error";
			cout << "InputWeights" << endl;
			for (node = 0; node < NetworkNodes; node++)
			{
				for (childNode = 0; childNode < InputNodes; childNode++)
				{
					file >> inputWeights[node][childNode];
					cout << inputWeights[node][childNode] << " ";
				}
				cout << endl;
			}
			cout << endl;
			file >> temp;
			if (temp != "NetworkWeights")
				throw "Protocol error";
			cout << "NetworkWeights" << endl;
			for (node = 0; node < NetworkNodes; node++)
			{
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					file >> networkWeights[node][childNode];
					cout << networkWeights[node][childNode] << " ";
				}
				cout << endl;
			}
			cout << endl;
			file >> temp;
			if (temp != "OutputWeights")
				throw "Protocol error";
			cout << "OutputWeights" << endl;
			for (node = 0; node < OutputNodes; node++)
			{
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					file >> outputWeights[node][childNode];
					cout << outputWeights[node][childNode] << " ";
				}
				cout << endl;
			}
			cout << endl;
			file >> temp;
			if (temp != "NetworkBias")
				throw "Protocol error";
			cout << "NetworkBias" << endl;
			for (node = 0; node < NetworkNodes; node++)
			{
				file >> networkBias[node];
				cout << networkBias[node] << " ";
			}
			cout << "\n\n";
			file >> temp;
			if (temp != "OutputBias")
				throw "Protocol error";
			cout << "OutputBias" << endl;
			for (node = 0; node < OutputNodes; node++)
			{
				file >> outputBias[node];
				cout << outputBias[node] << " ";
			}
			cout << "\n\n";
		}
		file.close();
	}
};