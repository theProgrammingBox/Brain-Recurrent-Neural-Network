#pragma once
#include "header.h"

class Network
{
private:
	float initialState[NetworkNodes];					// Initial state of the network
	float inputWeights[NetworkNodes][InputNodes];		// Input weights
	float networkWeights[NetworkNodes][NetworkNodes];	// Network weights
	float outputWeights[OutputNodes][NetworkNodes];		// Output weights
	float networkBias[NetworkNodes];					// Network bias
	float outputBias[OutputNodes];						// Output bias
	float* state;										// State of the network

	// Restricts x between -1 and 1
	float Binary(float x)
	{
		return x < -1 ? -1 : x > 1 ? 1 : x;
	}

public:
	Network()
	{
		Initialize();
	}

	// Initializes the network
	void Initialize()
	{
		uint64_t node, childNode;
		Random random = Random();
		delete[] state;
		state = new float[NetworkNodes];
		for (node = 0; node < NetworkNodes; node++)
		{
			networkBias[node] = random.DoubleRandom();
			initialState[node] = random.DoubleRandom();
			state[node] = initialState[node];
			for (childNode = 0; childNode < InputNodes; childNode++)
				inputWeights[node][childNode] = random.DoubleRandom();
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				networkWeights[node][childNode] = random.DoubleRandom();
		}
		for (node = 0; node < OutputNodes; node++)
		{
			outputBias[node] = random.DoubleRandom();
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				outputWeights[node][childNode] = random.DoubleRandom();
		}
	}

	// Resets the network to its initial state
	void Reset()
	{
		uint64_t node;
		for (node = 0; node < NetworkNodes; node++)
			state[node] = initialState[node];
	}

	// Propagates the input once through the network and calculates the output
	void ForwardPropagate(float* input, float* output)
	{
		uint64_t node, childNode;
		float sum;
		float* nextState = new float[NetworkNodes];
		for (node = 0; node < NetworkNodes; node++)
		{
			sum = networkBias[node];
			for (childNode = 0; childNode < InputNodes; childNode++)
				sum += inputWeights[node][childNode] * input[childNode];
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				sum += networkWeights[node][childNode] * state[childNode];
			nextState[node] = Binary(sum);
		}
		delete[] state;
		state = nextState;
		for (node = 0; node < OutputNodes; node++)
		{
			sum = outputBias[node];
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				sum += outputWeights[node][childNode] * state[childNode];
			output[node] = Binary(sum);
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