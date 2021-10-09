#include <iostream>
#include <fstream>
using namespace std;

ofstream in_sdk_file("../dbg/input_sdk.txt");
ofstream out_sdk_file("../dbg/out_L2_sdk.txt");

#include "core.h"
//#include "data_loader.cpp"
//#include "interpret.hpp"

/*
 * Mode 1: Output of maxpool
 *
 * Mode 2: Output without maxpool
 *
 * */
int mode = 2;

struct out_features{
	const int dim;
	const int ch;
	const int reps;
};


int main()
{
	cout<<"Vals per input = "<< Vals_per_Input <<", Input depth = "<<INPUT_depth<<endl;
	out_features OUT = {OFM_DIM_C2, OFM_CH_C2, NUM_REPS};

//	load_parameters(2);
//	int state = read_parameters();
//	if(state == 1)
//		return 1;
//	std::cout.setstate(std::ios_base::failbit);
//	hls::stream<ap_uint<simd*8*4*4>> inStream("tb_input_stream");
//	hls::stream<ap_uint<8*out_parallel_C2>> outStream("tb_output_stream");
	cout<< "output features: "<<endl;
	cout <<"dim: " <<OUT.dim <<", ch: "<<OUT.ch<<", reps: "<<OUT.reps<<endl;
	TO_32 outStream[((OUT.dim) * (OUT.dim) * (OUT.ch) * NUM_REPS)/4] = {0};

//	ap_uint<simd*8*4*4> input[ifmChannels/simd];

	int rep = 1;

	cout<<"Loading image ..."<<endl;
	ap_uint<bitw> image[NUM_REPS][IMG_CH][IMG_DIM][IMG_DIM] = {
//			#include "7_padded.txt"
			#include "../data/vgg_11_pack/input_image.txt"
			};
	cout<<"Packing image ..."<<endl;
	ap_uint<MEM_BANDWIDTH> packed_image[INPUT_depth * NUM_REPS] = {0};
	unsigned packed_iter=0;
	unsigned int num=0;
	ap_uint<MEM_BANDWIDTH> temp1 = 0;
	for(int rep=0; rep < NUM_REPS; rep++)
		for(int r = 0; r<IMG_DIM; r++){
			for(int c = 0; c<IMG_DIM; c++){
				for(int dp=0; dp<IMG_CH; dp++){

					unsigned int lowBit = num * 8;
					unsigned int highBit = (num+1)*8 - 1;
					temp1.range(highBit,lowBit) = image[rep][dp][r][c];

					if(++num == (MEM_BANDWIDTH/8)){
						packed_image[packed_iter++] = temp1;
						in_sdk_file <<hex<<temp1<<","<<endl;
						cout<<(packed_iter-1)<<" ";
						num = 0;
						temp1 = 0;
					}
				}
			}
		}

	in_sdk_file.close();
	cout <<"Running Image..."<<endl;
	nn_top(packed_image, outStream, false,
			0, 0, 0,
			0, 0, NUM_REPS);

	cout<<"Run completed."<<endl;
int diff = 0;
int temp_diff = 0;
int c_pool = 0;
//int pool1_gold[3136] = {
//#include "pool1(gold).txt"
//};



int pool2_gold[(OUT.dim) * (OUT.dim) * (OUT.ch) * (OUT.reps)]={
#include "../data/vgg_11_pack/relu_2(gold)"
};
//	cout.clear();
	int outData[OUT.reps][OUT.ch][OUT.dim][OUT.dim] = {0};
//	int outData_sdk[(OFM_DIM_C2*OFM_DIM_C2*OFM_CH_C2)] = {0};
//	cout<<"Output stream size = "<<outStream.size()<<endl;
//	if(outStream.empty())
//		cout<<"Output Stream empty"<<endl;
//	int out_size = outStream.size();

	TO_32 temp_out = 0;
	packed_iter = 0;
	int depth_iter=0;
	int row_iter=0;
	int col_iter=0;

	if(mode == 2)
	{
		temp_out = 0;
		packed_iter = 0;
		for(int rep=0; rep<OUT.reps; rep++)
			for(int r = 0; r<OUT.dim; r+=2)
				for(int c = 0; c<OUT.dim; c+=2)
					for(int i = 0; i<(OUT.ch); i++)
					{
						temp_out = outStream[packed_iter++];
						out_sdk_file << hex<<temp_out<<endl;
						for(int jj =0; jj < 2; jj++)
						{	for(int jc=0; jc < 2; jc++)
							{
							unsigned int lowBit = (jj*2 + jc) * 8;
							unsigned int highBit = (jj*2 + jc + 1)*8 - 1;
							outData[rep][i][r+jj][c+jc] = temp_out.range(highBit,lowBit);
							}
						}
					}
		cout<<"output unpacked ..."<<endl;
		out_sdk_file.close();
	}
	else if (mode == 1)
	{

		temp_out = 0;
		packed_iter = 0;
		for(int rep=0; rep<OUT.reps; rep++)
			for(int r = 0; r<OUT.dim; r++)
				for(int c = 0; c<OUT.dim; c++)
					for(int i = 0; i<(OUT.ch)/4; i++)
						{
		//				temp_out = outStream.read();
						temp_out = outStream[packed_iter++];
						out_sdk_file << hex<<temp_out<<endl;
							for(int j =0; j<4; j++)
								{
									unsigned int lowBit = j * 8;
									unsigned int highBit = (j+1)*8 - 1;
									outData[rep][i*4 + j][r][c] = temp_out.range(highBit,lowBit);
								}
						}
		cout<<"output unpacked ..."<<endl;
		out_sdk_file.close();
	}

		int count = 0;
	//	cout.clear();
		cout<<"Printing Output"<<endl;
		c_pool = 0;
		for(int rep=0; rep<OUT.reps; rep++)
			for(int i = 0; i<OUT.ch; i++)
				{
					for(int r = 0; r<OUT.dim; r++)
						{	for(int c = 0; c<OUT.dim; c++)
								{cout<<outData[rep][i][r][c];
									temp_diff = abs(outData[rep][i][r][c] - pool2_gold[c_pool++]);
									if(temp_diff > 35)
										{cout<<"Diff = "<< temp_diff <<" @"<<c_pool-1;
										diff++;
										}
									cout<<endl;
								}
						}
				}
		cout.clear();
		cout<<"Differences calculated"<<endl;
		cout<<"Total Differences = "<<diff<<endl;

		for(int rep=0; rep<OUT.reps; rep++){
			cout<<"Image: "<<rep+1<<endl;
			for(int r=0;r<OUT.dim; r++)
				{
					for(int c=0; c<OUT.dim; c++)
						cout<<outData[rep][0][r][c]<<" ";
					cout<<endl;
				}
		}


if (diff == 0)
	return 0;
else
	return 1;
}
