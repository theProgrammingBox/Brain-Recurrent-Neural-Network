#include "Network.h"
#include "NetworkTrainer.h"

void DoARun(NetworkTrainer& network)
{
	Random random = Random();
	float input[InputNodes]{};
	float output[OutputNodes]{};
	float expectedOutput[OutputNodes]{};

	for (int i = 0; i < 10; i++)
	{
		float sum = 0;
		for (uint64_t i = 0; i < InputNodes; i++)
		{
			input[i] = random.DoubleRandom();
			sum += input[i];
		}
		for (uint64_t i = 0; i < OutputNodes; i++)
		{
			expectedOutput[i] = sum + i;
		}
		network.ForwardPropagate(input, output, expectedOutput);
	}
	network.BackPropagate();
}

float GetError(NetworkTrainer& network, uint64_t iterations)
{
	float error = 0;
	Random random = Random();
	float input[InputNodes]{};
	float output[OutputNodes]{};
	float expectedOutput[OutputNodes]{};
	for (int iteration = 0; iteration < iterations; iteration++)
	{
		for (int i = 0; i < 10; i++)
		{
			float sum = 0;
			for (uint64_t i = 0; i < InputNodes; i++)
			{
				input[i] = random.DoubleRandom();
				sum += input[i];
			}
			for (uint64_t i = 0; i < OutputNodes; i++)
			{
				expectedOutput[i] = sum + i;
			}
			network.ForwardPropagate(input, output, expectedOutput);

			for (uint64_t i = 0; i < OutputNodes; i++)
			{
				error += abs(expectedOutput[i] - output[i]);
			}
		}
	}

	return error / iterations;
}

int main()
{
	// This is required for it to run on repl.it
	/*ifstream file("network.txt");
	string in;
	file >> in;
	file.close();*/

	NetworkTrainer network = NetworkTrainer();
	network.Import();
	float error = 1;
	uint64_t iterations = 0;

	while (error > 0.001)
	{
		iterations++;
		DoARun(network);

		if (iterations % 1000 == 0)
		{
			error = GetError(network, 1000);
			cout << "Iteration: " << iterations << ", Error: " << error << "\n\n";
			network.Export();
		}
	}
	cout << "Iteration: " << iterations << ", Error: " << GetError(network, 1000) << "\n\n";
	network.Export();

	return 0;
}