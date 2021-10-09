#include <iostream>
using namespace std;


#include "core.h"
//#include "interpret.hpp"


int main()
{
	cout<<"Vals per input = "<< Vals_per_Input <<", Input depth = "<<INPUT_depth<<endl;
	std::cout.setstate(std::ios_base::failbit);
	hls::stream<ap_uint<simd*8*4*4>> inStream("tb_input_stream");
//	hls::stream<ap_uint<8*out_parallel>> outStream("tb_output_stream");

	ap_uint<8*out_parallel> outStream[(14*14*16)/out_parallel] = {0};

//	ap_uint<simd*8> input[4*4];
	ap_uint<simd*8*4*4> input[ifmChannels/simd];

	int rep = 1;

	ap_uint<bitw> image[IMG_CH][IMG_DIM][IMG_DIM] = {
//			#include "7_padded.txt"
			#include "../data/weights_bn_v1_pack/input_image.txt"
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


	nn_top_layer1(packed_image, outStream, 1);

int diff = 0;
int temp_diff = 0;
int c_pool = 0;
int pool1_gold[3136] = {
#include "../data/weights_bn_v1_pack/pool1(gold).txt"
};

	cout.clear();
	int outData[OFM_CH_C1][OFM_DIM_C1][OFM_DIM_C1] = {0};
//	cout<<"Output stream size = "<<outStream.size()<<endl;
//	if(outStream.empty())
//		cout<<"Output Stream empty"<<endl;
//	int out_size = outStream.size();

	ap_uint<8*out_parallel> temp_out = 0;
	packed_iter = 0;
	for(int r = 0; r<OFM_DIM_C1; r++)
		for(int c = 0; c<OFM_DIM_C1; c++)
			for(int i = 0; i<OFM_CH_C1/out_parallel; i++)
				{
//				temp_out = outStream.read();
				temp_out = outStream[packed_iter++];
					for(int j =0; j<out_parallel; j++)
						{
							unsigned int lowBit = j * 8;
							unsigned int highBit = (j+1)*8 - 1;
							outData[i*out_parallel + j][r][c] = temp_out.range(highBit,lowBit);

						}

				}


	int count = 0;
//	cout.clear();
	cout<<"Printing Output"<<endl;
	c_pool = 0;
	for(int i = 0; i<OFM_CH_C1; i++)
		{
			for(int r = 0; r<OFM_DIM_C1; r++)
				{	for(int c = 0; c<OFM_DIM_C1; c++)
						{cout<<outData[i][r][c];
							temp_diff = abs(outData[i][r][c] - pool1_gold[c_pool++]);
							if(temp_diff > 6)
								{cout<<" - Diff = "<< temp_diff <<" @"<<c_pool-1;
								diff++;
								}
							cout<<endl;

						}
//					cout<<endl;
				}
//			break;
		}
	cout.clear();

	cout<<"Total Differences = "<<diff<<endl;

	for(int r=0;r<OFM_DIM_C1; r++)
		{
			for(int c=0; c<OFM_DIM_C1; c++)
				cout<<outData[2][r][c]<<" ";
			cout<<endl;
		}


if (diff == 0)
	return 0;
else
	return 1;
}
