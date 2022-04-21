#pragma once
#include "header.h"

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
	vector<array<float, NetworkNodes>> interstateGradients;		// Intermediate gradients of the network
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
		interstateGradients.clear();
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

	// Propagates the input once through the network and calculates the output while storing the expected output
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
			inputArray[node] = input[node];
		inputs.push_back(inputArray);

		// Adds the expected output to the expected output array for backpropagation
		for (node = 0; node < OutputNodes; node++)
			expectedOutputArray[node] = expectedOutput[node];
		expectedOutputs.push_back(expectedOutputArray);

		// Add an empty array to the intermediate states for backpropagation
		interstateGradients.push_back(interstateGradientArray);

		// Use the stored state as the first state if just reset
		if (currentIteration != 0)
		{
			for (node = 0; node < NetworkNodes; node++)
			{
				sum = networkBias[node];

				for (childNode = 0; childNode < InputNodes; childNode++)
					sum += inputWeights[node][childNode] * input[childNode];

				// Uses the previous state to calculate the next state
				for (childNode = 0; childNode < NetworkNodes; childNode++)
					sum += networkWeights[node][childNode] * states[currentIteration - 1][childNode];

				interstateArray[node] = sum;
				stateArray[node] = Binary(sum);
			}

			// Add the intermediate state to the intermediate states array for backpropagation
			interstates.push_back(interstateArray);

			// Add the state to the states array for backpropagation
			states.push_back(stateArray);

			for (node = 0; node < OutputNodes; node++)
			{
				sum = outputBias[node];

				// Uses the just computed state to calculate the output
				for (childNode = 0; childNode < NetworkNodes; childNode++)
					sum += outputWeights[node][childNode] * states[currentIteration][childNode];

				outputArray[node] = Binary(sum);
				output[node] = outputArray[node];
			}

			// Add the output to the outputs array for backpropagation
			outputs.push_back(outputArray);
		}
		else
		{
			// Use the initial state as the first state if just reset
			for (node = 0; node < NetworkNodes; node++)
			{
				sum = networkBias[node];

				for (childNode = 0; childNode < InputNodes; childNode++)
					sum += inputWeights[node][childNode] * input[childNode];

				// Uses the initial state to calculate the next state
				for (childNode = 0; childNode < NetworkNodes; childNode++)
					sum += networkWeights[node][childNode] * initialState[childNode];

				interstateArray[node] = sum;
				stateArray[node] = Binary(sum);
			}

			// Add the intermediate state to the intermediate states array for backpropagation
			interstates.push_back(interstateArray);

			// Add the state to the states array for backpropagation
			states.push_back(stateArray);

			for (node = 0; node < OutputNodes; node++)
			{
				sum = outputBias[node];

				// Uses the just computed state to calculate the output
				for (childNode = 0; childNode < NetworkNodes; childNode++)
					sum += outputWeights[node][childNode] * states[currentIteration][childNode];

				outputArray[node] = Binary(sum);
				output[node] = outputArray[node];
			}

			// Add the output to the outputs array for backpropagation
			outputs.push_back(outputArray);
		}

		// Increment the iteration to define how many iterations have passed
		currentIteration++;
	}

	void BackPropagate()
	{
		float initialStateGradient[NetworkNodes]{};						// Initial state gradient
		float inputWeightGradients[NetworkNodes][InputNodes]{};			// Input weight gradients
		float networkWeightGradients[NetworkNodes][NetworkNodes]{};		// Network weight gradients
		float outputWeightGradients[OutputNodes][NetworkNodes]{};		// Output weight gradients
		float networkBiasGradient[NetworkNodes]{};						// Network bias gradient
		float outputBiasGradient[OutputNodes]{};						// Output bias gradient
		float outputGradient[OutputNodes];								// Output gradient
		float* interstateGradient = new float[NetworkNodes];			// Intermediate state gradient
		float* previousInterstateGradient = new float[NetworkNodes];	// Previous intermediate state gradient

		float standardDeviation;										// The distance to the previous gradient averages to determine the average gradient to apply

		uint64_t layer, node, childNode;

		// loop for every iteration excluding the first one to calculate the gradients
		for (layer = currentIteration - 1; layer > 0; layer--)
		{
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
					interstateGradients[layer][node] += outputWeights[childNode][node] * outputGradient[childNode];
					outputWeightGradients[childNode][node] += outputGradient[childNode] * states[layer][node];
				}

				// Calculate the network bias gradient as well as the intermediate state gradient
				interstateGradients[layer][node] *= BinaryGradient(interstates[layer][node]);
				networkBiasGradient[node] += interstateGradients[layer][node];

				// Calculate the network weight gradient
				for (childNode = 0; childNode < InputNodes; childNode++)
					inputWeightGradients[node][childNode] += inputs[layer][childNode] * interstateGradients[layer][node];

				// Calculate the network weight gradient as well as the intermediate state gradient
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					interstateGradients[layer - 1][childNode] += networkWeights[node][childNode] * interstateGradients[layer][node];
					networkWeightGradients[node][childNode] += states[layer - 1][childNode] * interstateGradients[layer][node];
				}
			}
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
				interstateGradients[0][node] += outputWeights[childNode][node] * outputGradient[childNode];
				outputWeightGradients[childNode][node] += outputGradient[childNode] * states[0][node];
			}

			interstateGradients[0][node] *= BinaryGradient(interstates[0][node]);
			networkBiasGradient[node] += interstateGradients[0][node];

			for (childNode = 0; childNode < InputNodes; childNode++)
				inputWeightGradients[node][childNode] += inputs[layer][childNode] * interstateGradients[0][node];

			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				initialStateGradient[childNode] += networkWeights[node][childNode] * interstateGradients[0][node];
				networkWeightGradients[node][childNode] += initialState[childNode] * interstateGradients[0][node];
			}
		}

		interstates.clear();
		states.clear();
		interstateGradients.clear();
		inputs.clear();
		outputs.clear();
		expectedOutputs.clear();

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

			if (standardDeviation < GradientPrecision * GradientPrecision)
			{
				consecutiveOks++;
			}
			else
			{
				consecutiveOks = 0;
			}
		}

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

		batchCount++;

		if (consecutiveOks == RequiredConsecutiveOks)
		{
			for (node = 0; node < NetworkNodes; node++)
			{
				networkBiasGradient[node] += networkBiasGradientSum[node] / batchCount * LearningRate;
				networkBiasGradientSum[node] = 0;
				initialStateGradient[node] += initialStateGradientSum[node] / batchCount * LearningRate;
				initialStateGradientSum[node] = 0;
				for (childNode = 0; childNode < InputNodes; childNode++)
					inputWeightGradients[node][childNode] += inputWeightGradientSums[node][childNode] / batchCount * LearningRate;
				inputWeightGradientSums[node][childNode] = 0;
				for (childNode = 0; childNode < NetworkNodes; childNode++)
					networkWeightGradients[node][childNode] += networkWeightGradientSums[node][childNode] / batchCount * LearningRate;
				networkWeightGradientSums[node][childNode] = 0;
			}
			for (node = 0; node < OutputNodes; node++)
			{
				outputBiasGradient[node] += outputBiasGradientSum[node] / batchCount * LearningRate;
				outputBiasGradientSum[node] = 0;
				for (childNode = 0; childNode < NetworkNodes; childNode++)
					outputWeightGradients[node][childNode] += outputWeightGradientSums[node][childNode] / batchCount * LearningRate;
				outputWeightGradientSums[node][childNode] = 0;
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