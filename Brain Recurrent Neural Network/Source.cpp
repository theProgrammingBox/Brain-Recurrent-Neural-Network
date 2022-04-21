#include "Network.h"
#include "NetworkTrainer.h"


int main()
{
	ifstream file("network.txt");
	string in;
	file >> in;
	file.close();

	NetworkTrainer network;
	network.Import();
	network.Export();
	return 0;
}