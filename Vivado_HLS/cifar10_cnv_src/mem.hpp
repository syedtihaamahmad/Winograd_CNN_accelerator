#ifndef MEM_H
#define MEM_H
#include <string>
#include <fstream>
#include <iostream>
using namespace std;

struct layer_features{
	unsigned int ch;
	unsigned int row;
	unsigned int col;
};

template<typename layer_type>
void load(string path,layer_features layer,layer_type ***weight )
{
	ifstream in_file(path);
	    if(!in_file.is_open()){
	        cout<< "Failed to open file"<<endl;
	        assert(0);
	    }



	string myString;
	string line;


	for(int ch=0; ch<layer.ch; ch++) {
		for(int row=0; row<layer.row; row++){
			for(int col=0; col<layer.col; col++){
				getline(in_file, line);
				stringstream ss(line);
				getline(ss, myString, ',');
				weight[ch][row][col] = stol(myString, nullptr, 0);
		//		cout << hex << unum << endl;
			}
		}
	}

	in_file.close();



	}






#endif
