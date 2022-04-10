// Brain Recurrent Neural Network
#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <chrono>

using namespace std::chrono;
using namespace std;

class Random
{
private:
	uint64_t s[4];

	uint64_t rotl(const uint64_t x, int k)
	{
		return (x << k) | (x >> (64 - k));
	}

public:
	Random()
	{
		Seed();
		jump();
	}

	void jump(void) {
		static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		uint64_t s2 = 0;
		uint64_t s3 = 0;
		for (int i = 0; i < sizeof JUMP / sizeof * JUMP; i++)
			for (int b = 0; b < 64; b++) {
				if (JUMP[i] & UINT64_C(1) << b) {
					s0 ^= s[0];
					s1 ^= s[1];
					s2 ^= s[2];
					s3 ^= s[3];
				}
				ULongRandom();
			}

		s[0] = s0;
		s[1] = s1;
		s[2] = s2;
		s[3] = s3;
	}

	void Seed(
		uint64_t seed1 = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(),
		uint64_t seed2 = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count(),
		uint64_t seed3 = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count(),
		uint64_t seed4 = duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count())
	{
		s[0] = 0x76e15d3efefdcbbf ^ seed1;
		s[1] = 0xc5004e441c522fb3 ^ seed2;
		s[2] = 0x77710069854ee241 ^ seed3;
		s[3] = 0x39109bb02acbe635 ^ seed4;
	}

	uint64_t ULongRandom()
	{
		//const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
		const uint64_t t = s[1] << 17;

		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];
		s[2] ^= t;
		s[3] = rotl(s[3], 45);

		return rotl(s[0] + s[3], 23) + s[0];// result;
	}

	int64_t LongRandom()
	{
		return int64_t(ULongRandom());
	}

	double UDoubleRandom()	// 0 through 1
	{
		return ULongRandom() * 5.42101086243e-20;
	}

	double DoubleRandom()	// -1 through 1
	{
		return int64_t(ULongRandom()) * 1.08420217249e-19;
	}

	double NormalRandom(double mean, double standardDeviation)
	{
		double x, y, radius;
		do
		{
			x = DoubleRandom();
			y = DoubleRandom();

			radius = x * x + y * y;
		} while (radius >= 1.0 || radius == 0.0);

		return x * sqrt(-2.0 * log(radius) / radius) * standardDeviation + mean;
	}
};

#define random Random()					// Global random object
#define InputNodes uint64_t(1)			// Number of input nodes
#define NetworkNodes uint64_t(10)		// Number of nodes in the network
#define OutputNodes uint64_t(1)			// Number of output nodes
#define NetworkIterations uint64_t(10)	// Number of times to run the network before the output is calculates

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

	// Propagates the input once through the network and calculates the output after NetworkIterations iterations
	void ForwardPropagate(float* input, float* output)
	{
		uint64_t iterations, node, childNode;
		float sum;
		float* nextState = new float[NetworkNodes];
		float* temp;
		
		for (iterations = 0; iterations < NetworkIterations; iterations++)
		{
			for (node = 0; node < NetworkNodes; node++)
			{
				sum = networkBias[node];
			for (childNode = 0; childNode < InputNodes; childNode++)
				sum += inputWeights[node][childNode] * input[childNode];
				for (childNode = 0; childNode < NetworkNodes; childNode++)
					sum += networkWeights[node][childNode] * nextState[childNode];
				nextState[node] = Binary(sum);
			}
			temp = nextState;
			nextState = state;
			state = temp;
		}
		delete[] nextState;

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
		file << "NetworkIterations " << NetworkIterations << "\n\n";
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
			cout << "InputNodes " << InputNodes << endl;
			file >> temp;
			file >> tempParam;
			if (temp != "NetworkNodes" || tempParam != NetworkNodes)
				throw "Protocol error";
			if (tempParam != NetworkNodes)
				throw "NetworkNodes mismatch";
			cout << "NetworkNodes " << NetworkNodes << endl;
			file >> temp;
			file >> tempParam;
			if (temp != "OutputNodes" || tempParam != OutputNodes)
				throw "Protocol error";
			if (tempParam != OutputNodes)
				throw "OutputNodes mismatch";
			cout << "OutputNodes " << OutputNodes << endl;
			file >> temp;
			file >> tempParam;
			if (temp != "NetworkIterations" || tempParam != NetworkIterations)
				throw "Protocol error";
			if (tempParam != NetworkIterations)
				throw "NetworkIterations mismatch";
			cout << "NetworkIterations " << NetworkIterations << endl;
			file >> temp;
			if (temp != "InitialState")
				throw "Protocol error";
			for (node = 0; node < NetworkNodes; node++)
			{
				file >> initialState[node];
				cout << initialState[node] << " ";
			}
			cout << endl;
			file >> temp;
			if (temp != "InputWeights")
				throw "Protocol error";
			for (node = 0; node < NetworkNodes; node++)
			{
				for (childNode = 0; childNode < InputNodes; childNode++)
				{
					file >> inputWeights[node][childNode];
					cout << inputWeights[node][childNode] << " ";
				}
				cout << endl;
			}
			file >> temp;
			if (temp != "NetworkWeights")
				throw "Protocol error";
			for (node = 0; node < NetworkNodes; node++)
			{
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					file >> networkWeights[node][childNode];
					cout << networkWeights[node][childNode] << " ";
				}
				cout << endl;
			}
			file >> temp;
			if (temp != "OutputWeights")
				throw "Protocol error";
			for (node = 0; node < OutputNodes; node++)
			{
				for (childNode = 0; childNode < NetworkNodes; childNode++)
				{
					file >> outputWeights[node][childNode];
					cout << outputWeights[node][childNode] << " ";
				}
				cout << endl;
			}
			file >> temp;
			if (temp != "NetworkBias")
				throw "Protocol error";
			for (node = 0; node < NetworkNodes; node++)
			{
				file >> networkBias[node];
				cout << networkBias[node] << " ";
			}
			cout << endl;
			file >> temp;
			if (temp != "OutputBias")
				throw "Protocol error";
			for (node = 0; node < OutputNodes; node++)
			{
				file >> outputBias[node];
				cout << outputBias[node] << " ";
			}
			cout << endl;
		}
		file.close();
	}
};

class NetworkTrainer
{
private:
	float initialState[NetworkNodes];							// Initial state of the network
	float inputWeights[NetworkNodes][InputNodes];				// Input weights
	float networkWeights[NetworkNodes][NetworkNodes];			// Network weights
	float outputWeights[OutputNodes][NetworkNodes];				// Output weights
	float networkBias[NetworkNodes];							// Network bias
	float outputBias[OutputNodes];								// Output bias
	float initialStateGradient[NetworkNodes];					// Initial state gradient
	float inputWeightGradients[NetworkNodes][InputNodes];		// Input weight gradients
	float networkWeightGradients[NetworkNodes][NetworkNodes];	// Network weight gradients
	float outputWeightGradients[OutputNodes][NetworkNodes];		// Output weight gradients
	float networkBiasGradient[NetworkNodes];					// Network bias gradient
	float outputBiasGradient[OutputNodes];						// Output bias gradient
	vector<array<float, InputNodes>> interstates;				// Intermediate states of the network
	vector<array<float, InputNodes>> states;					// States of the network
	vector<array<float, InputNodes>> interstateGradients;		// Intermediate gradients of the network

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
		for (node = 0; node < NetworkNodes; node++)
		{
			networkBias[node] = random.DoubleRandom();
			initialState[node] = random.DoubleRandom();
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
		for (node = 0; node < NetworkNodes; node++)
		{
			networkBiasGradient[node] = 0;
			initialStateGradient[node] = 0;
			for (childNode = 0; childNode < InputNodes; childNode++)
				inputWeightGradients[node][childNode] = 0;
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				networkWeightGradients[node][childNode] = 0;
		}
		for (node = 0; node < OutputNodes; node++)
		{
			outputBiasGradient[node] = 0;
			for (childNode = 0; childNode < NetworkNodes; childNode++)
				outputWeightGradients[node][childNode] = 0;
		}
	}

	// Clears the states of the network and applies the parameter gradients to the network
	void Reset()
	{
		uint64_t node, childNode;
		interstates.clear();
		states.clear();
		interstateGradients.clear();
		for (node = 0; node < NetworkNodes; node++)
		{
			initialState[node] += initialStateGradient[node];
			initialStateGradient[node] = 0;
			for (childNode = 0; childNode < InputNodes; childNode++)
			{
				inputWeights[node][childNode] += inputWeightGradients[node][childNode];
				inputWeightGradients[node][childNode] = 0;
			}
			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				networkWeights[node][childNode] += networkWeightGradients[node][childNode];
				networkWeightGradients[node][childNode] = 0;
			}
		}
		for (node = 0; node < OutputNodes; node++)
		{
			outputBias[node] += outputBiasGradient[node];
			outputBiasGradient[node] = 0;
			for (childNode = 0; childNode < NetworkNodes; childNode++)
			{
				outputWeights[node][childNode] += outputWeightGradients[node][childNode];
				outputWeightGradients[node][childNode] = 0;
			}
		}
	}
};

int main()
{
	Network network;

	network.Import();
	network.Export();
	
	return 0;
}