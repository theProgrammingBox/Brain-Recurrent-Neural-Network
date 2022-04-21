#include "Network.h"
#include "NetworkTrainer.h"


int main()
{
	// This is required for it to run on repl.it
	/*ifstream file("network.txt");
	string in;
	file >> in;
	file.close();*/
	Random random = Random();

	float input[InputNodes]{};
	float output[OutputNodes]{};
	float expectedOutput[OutputNodes]{};

	NetworkTrainer network = NetworkTrainer();
	//network.Import();
	uint64_t iterations = 0;
	while (true || iterations < 10000)
	{
		iterations++;
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
		network.BackPropagate();
		if (iterations % 1000 == 0)
		{
			float error = 0;

			cout << "Iteration: " << iterations << endl;

			cout << "Output: ";
			for (uint64_t i = 0; i < OutputNodes; i++)
			{
				cout << output[i] << " ";
			}
			cout << endl;

			for (uint64_t i = 0; i < OutputNodes; i++)
			{
				error += abs(expectedOutput[i] - output[i]);
			}

			cout << "Error: " << error << "\n\n";
		}
	}
	/*for (int i = 0; i < InputNodes; i++)
	{
		input[i] = random.DoubleRandom();
	}
	for (int i = 0; i < OutputNodes; i++)
	{
		expectedOutput[i] = random.DoubleRandom();
	}
	network.ForwardPropagate(input, output, expectedOutput);*/
	network.Export();
	return 0;
}