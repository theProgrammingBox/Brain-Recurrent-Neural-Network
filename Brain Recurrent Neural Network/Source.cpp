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

	uint64_t RequiredConsecutiveOksArr[4] = { 10, 6, 3, 1 };

	float GradientPrecisionArr[4] = { 1, 0.5, 0.1, 0.05 };

	float LearningRateArr[4] = { 0.05, 0.01, 0.005, 0.001 };

	ofstream fout("Data.txt");

	for (int item = 0; item < 64; item++)
	{
		RequiredConsecutiveOks = RequiredConsecutiveOksArr[item & 3];
		GradientPrecision = GradientPrecisionArr[int((item & 15) / 4.0)];
		LearningRate = LearningRateArr[int(item / 16.0)];

		fout << "RequiredConsecutiveOks: " << RequiredConsecutiveOks << endl;
		fout << "GradientPrecision: " << GradientPrecision << endl;
		fout << "LearningRate: " << LearningRate << endl;

		for (int j = 0; j < 10; j++)
		{
			cout << (10 * item + j) / 6.40 << "%" << endl;
			fout << "Run " << j << endl;

			NetworkTrainer network = NetworkTrainer();
			uint64_t iterations = 0;
			float error;

			while (iterations < 10000)
			{
				iterations++;
				DoARun(network);

				if (iterations % 100 == 0)
				{
					error = GetError(network, 100);
					fout << "Iteration: " << iterations << " Error: " << error << endl;
					/*cout << "Iteration: " << iterations << ", Error: " << error << "\n\n";
					network.Export();*/
				}
			}
		}
		/*cout << "Iteration: " << iterations << ", Error: " << GetError(network, 1000) << "\n\n";
		network.Export();*/
	}

	fout.close();

	return 0;
}