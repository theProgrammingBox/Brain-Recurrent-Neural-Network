// Brain Recurrent Neural Network
#include <iostream>
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

#define random Random()		 // Global random object
#define InputNodes 1		 // Number of input nodes
#define NetworkNodes 10		 // Number of nodes in the network
#define OutputNodes 1		 // Number of output nodes
#define NetworkIterations 10 // Number of times to run the network before the output is sampled

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

	float Binary(float x)	// Restricts x between -1 and 1
	{
		return x < -1 ? -1 : x > 1 ? 1 : x;
	}

public:
	Network()
	{
		Initialize();
	}

	void Initialize()	// Initializes the network
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

	void Reset()	// Resets the network to its initial state
	{
		uint64_t node;
		for (node = 0; node < NetworkNodes; node++)
			state[node] = initialState[node];
	}

	void ForwardPropagate(float* input, float* output)
	{
		uint64_t iterations, node, childNode;
		float sum;
		float* nextState = new float[NetworkNodes];

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
			float* temp = nextState;
			nextState = state;
			state = temp;
		}
	}
};

int main()
{
	return 0;
}