#include "Network.h"
#include "NetworkTrainer.h"


int main()
{
	// This is required for it to run on repl.it
	/*ifstream file("network.txt");
	string in;
	file >> in;
	file.close();*/

	NetworkTrainer network;
	network.Import();
	network.Export();
	return 0;
}