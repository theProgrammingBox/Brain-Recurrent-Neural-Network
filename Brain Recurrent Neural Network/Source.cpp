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
			expectedOutput[i] = sum / 2.0;
		}
		network.ForwardPropagate(input, output, expectedOutput);
	}
	network.BackPropagate();
}

float GetError(Network& network)
{
	const uint64_t ErrorRequiredConsecutiveOks = 10;
	const float ErrorGradientPrecision = 0.01;

	uint64_t node;
	Random random = Random();
	float input[InputNodes]{};
	float output[OutputNodes]{};
	float expectedOutput[OutputNodes]{};
	float sum;

	float error;
	float errorSum = 0;
	uint64_t batchCount = 0;
	uint64_t consecutiveOks = 0;
	float standardDeviation;

	while (consecutiveOks < ErrorRequiredConsecutiveOks)
	{
		network.Reset();
		error = 0;

		for (int i = 0; i < 10; i++)
		{
			sum = 0;

			for (node = 0; node < InputNodes; node++)
			{
				input[node] = random.DoubleRandom();
				// cout << "Input " << node << ": " << input[node] << endl;
				sum += input[node];
				// cout << "Sum " << sum << endl;
			}

			for (node = 0; node < OutputNodes; node++)
			{
				expectedOutput[node] = sum / 2.0;
				// cout << "Expected " << node << ": " << expectedOutput[node] << endl;
			}
			network.ForwardPropagate(input, output);

			for (node = 0; node < OutputNodes; node++)
			{
				// cout << "Output " << node << ": " << output[node] << endl;
				error += abs(expectedOutput[node] - output[node]);
			}
			// cout << endl;
		}

		if (batchCount != 0)
		{
			standardDeviation = (errorSum + error) / (batchCount + 1) - (errorSum / batchCount);

			if (abs(standardDeviation) < ErrorGradientPrecision)
			{
				consecutiveOks++;
			}
			else
			{
				consecutiveOks = 0;
			}
		}

		errorSum += error;
		batchCount++;
	}

	return errorSum / batchCount;
}

int main()
{
	cout << left;
	uint64_t RequiredConsecutiveOksItem = 0;
	uint64_t GradientPrecisionItem = 0;
	uint64_t LearningRateItem = 0;
	uint64_t PhantomCoefficientItem = 0;

	uint64_t item;
	uint64_t run;

	uint64_t iterations;
	uint64_t training;

	ofstream fout("Data.txt");

	for (item = 0; item < NumItems; item++)
	{
		RequiredConsecutiveOks = RequiredConsecutiveOksArr[RequiredConsecutiveOksItem];
		GradientPrecision = GradientPrecisionArr[GradientPrecisionItem];
		LearningRate = LearningRateArr[LearningRateItem];
		PhantomCoefficient = PhantomCoefficientArr[PhantomCoefficientItem];

		fout << "RequiredConsecutiveOks: " << RequiredConsecutiveOks << endl;
		fout << "GradientPrecision: " << GradientPrecision << endl;
		fout << "LearningRate: " << LearningRate << endl;
		fout << "PhantomCoefficient: " << PhantomCoefficient << endl;

		for (run = 0; run < NumRuns; run++)
		{
			cout
				<< fixed << setprecision(2) << setw(10) << "Progress: " << right
				<< float(NumRuns * item + run) / (NumItems * (NumItems - 1) + (NumRuns - 1)) * 100 << "%"
				<< left << setprecision(6) << endl;
			fout << "Run " << run << endl;
			cout.unsetf(ios::fixed);

			NetworkParameters parameter = NetworkParameters();
			NetworkTrainer networkTrainer = NetworkTrainer(&parameter);
			Network network = Network(&parameter);
			iterations = 0;

			for (iterations = 0; iterations < NumTrainingPerRuns; iterations++)
			{
				for (training = 0; training < NumTrainingBeforeSample; training++)
				{
					DoARun(networkTrainer);
				}

				fout << "NumTraining: " << (iterations + 1) * NumTrainingBeforeSample << " Error: " << GetError(network) << endl;
			}
		}

		if (++RequiredConsecutiveOksItem == NumRequiredConsecutiveOks)
		{
			RequiredConsecutiveOksItem = 0;

			if (++GradientPrecisionItem == NumGradientPrecision)
			{
				GradientPrecisionItem = 0;

				if (++LearningRateItem == NumLearningRate)
				{
					LearningRateItem = 0;

					if (++PhantomCoefficientItem == NumPhantomCoefficient)
					{
						PhantomCoefficientItem = 0;
					}
				}
			}
		}
	}

	fout.close();

	/*NetworkParameters parameter = NetworkParameters();
	parameter.Import();
	NetworkTrainer networkTrainer = NetworkTrainer(&parameter);
	Network network = Network(&parameter);
	for (;;)
	{
		cout << " Error: " << GetError(network) << endl;
		for (int i = 0; i < 100000; i++)
		{
			DoARun(networkTrainer);
		}
		network.Export();
	}*/

	/*NetworkParameters parameter = NetworkParameters();
	parameter.Import();
	Network network = Network(&parameter);

	float input[InputNodes]{};
	float output[OutputNodes]{};
	float expectedOutput[OutputNodes]{};

	for (;;)
	{
		cout << "Enter num1: ";
		cin >> input[0];
		cout << "Enter num2: ";
		cin >> input[1];

		for (uint64_t i = 0; i < OutputNodes; i++)
		{
			expectedOutput[i] = (input[0] + input[1]) / 2.0;
			cout << "Expected " << i << ": " << expectedOutput[i] << endl;
		}

		network.ForwardPropagate(input, output);

		for (uint64_t i = 0; i < OutputNodes; i++)
		{
			cout << "Output " << i << ": " << output[i] << endl;
		}
	}*/

	return 0;
}