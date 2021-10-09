#include <iostream>
using namespace std;


#include "core.h"
//#include "interpret.hpp"


int main()
{
	cout <<"***** Running Windowing function testbench ******"<<endl;
	cout<<"Vals per input = "<< Vals_per_Input <<", Input depth = "<<INPUT_depth<<endl;
	std::cout.setstate(std::ios_base::failbit);
//	hls::stream<ap_uint<simd*8*4*4>> inStream("tb_input_stream");
	hls::stream<ap_uint<8*simd*4*4>> outStream("tb_output_stream");

//	ap_uint<8*simd*4*4> outStream[(IFM_DIM_C1/2)*(IFM_DIM_C1/2)] = {0};

//	ap_uint<simd*8> input[4*4];
	ap_uint<simd*8*4*4> input[ifmChannels/simd];

	int rep = 1;

	ap_uint<bitw> image[IMG_CH][IMG_DIM][IMG_DIM] = {
//			#include "7_padded.txt"
			#include "../data/7_test.txt"
			};
	ap_uint<MEM_BANDWIDTH> packed_image[INPUT_depth] = {0};
	unsigned packed_iter=0;
	unsigned int num=0;
	ap_uint<MEM_BANDWIDTH> temp1 = 0;
	for(int r = 0; r<IMG_DIM; r++){
		for(int c = 0; c<IMG_DIM; c++){
			for(int dp=0; dp<IMG_CH; dp++){

				unsigned int lowBit = num * 8;
				unsigned int highBit = (num+1)*8 - 1;
				temp1.range(highBit,lowBit) = image[dp][r][c];

				if(++num == (MEM_BANDWIDTH/8)){
					packed_image[packed_iter++] = temp1;
					num = 0;
					temp1 = 0;
				}
			}
		}
	}


	nn_top_window(packed_image, outStream, 1);

int diff = 0;
int temp_diff = 0;
int c_pool = 0;

	cout.clear();
	int outData[IFM_DIM_C1/2][IFM_DIM_C1/2][IMG_CH][4][4] = {0};

	cout<<"Output stream size = "<<dec<<outStream.size()<<endl;
	if(outStream.empty())
		cout<<"Output Stream empty"<<endl;
	int out_size = outStream.size();

	ap_uint<8*simd*4*4> temp_out = 0;
	packed_iter = 0;
	for(int r = 0; r<IMG_DIM/2; r++)
		for(int c = 0; c<IMG_DIM/2; c++){
							temp_out = outStream.read();
//			temp_out = outStream[packed_iter++];
			for(int i = 0; i<IMG_CH; i++) //Not used for MNIST
				{
					for(int rw =0; rw<4; rw++)
						{
							for(int cw=0;cw<4;cw++){
								unsigned int lowBit = (cw+rw*4) * 8;
								unsigned int highBit = ((cw+rw*4)+1)*8 - 1;
								outData[r][c][i][rw][cw] = temp_out.range(highBit,lowBit);
							}
						}
				}
		}


	int count = 0;
	cout.clear();
	cout<<"Printing window"<<endl;

	//Total windows = IMG_DIM/2 = 14*14
	int window_num = 80;
	for(int r=0; r<4; r++)
	{
		for(int c=0; c<4; c++)
		{
			cout << dec <<outData[4][4][0][r][c] <<" ";
			if (outData[4][4][0][r][c] != image[0][7+r][7+c])
				diff++;
		}
		cout<<endl;
	}



	cout<<"Total Differences = "<<diff<<endl;



if (diff == 0)
	return 0;
else
	return 1;
}
