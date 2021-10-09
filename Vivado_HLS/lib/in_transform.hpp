#ifndef INTRANSFORM_H
#define INTRANSFORM_H

#include <iostream>
using namespace std;


template<
unsigned next_SIMDi,
unsigned int BIT_W,
unsigned int IFMdim,
unsigned int IFM_ch,
unsigned int trans_bit
>
void in_transform(hls::stream<ap_uint<8*next_SIMDi*4*4>> &in,
			hls::stream<ap_uint<trans_bit*next_SIMDi*4*4>> &out, const unsigned int numReps)
{
	ap_uint<8*next_SIMDi*4*4>  inElem;
	ap_uint<BIT_W> inElem_partitioned[next_SIMDi][16];
#pragma HLS ARRAY_PARTITION variable=inElem_partitioned complete dim=2

	ap_int<trans_bit> inElem_transformed[next_SIMDi][16];
#pragma HLS ARRAY_PARTITION variable=inElem_transformed complete dim=2
int tile_num = 0;


ap_uint<trans_bit*next_SIMDi*4*4> outElem = 0;


	for(unsigned i=0; i<(IFMdim/2 - 1)*(IFMdim/2 - 1)*IFM_ch/next_SIMDi*numReps; i++)
	{
	#pragma HLS PIPELINE II=1
		inElem = in.read();
		tile_num++;




		//Transferring input to partitioned array
		for(unsigned  pe = 0; pe < 16; pe++)
		{
			for(unsigned sim=0; sim < next_SIMDi; sim++){
	#pragma HLS UNROLL
			inElem_partitioned[sim][pe] = inElem.range(BIT_W-1, 0);
			inElem = inElem >> BIT_W;
			}
		}

//		if(tile_num == 79){
//			cout.clear();
//			cout<<"Testing trans"<<endl;
//			for(int i=0; i<4; i++){
//				for(int j=0; j<4; j++)
//					cout<< dec<<inElem_partitioned[0][i*4+j]<<" ";
//			cout<<endl;
//				}
//			std::cout.setstate(std::ios_base::failbit);
//
//		}


		/* Transform the input */

		for(unsigned sim=0; sim<next_SIMDi; sim++){
			ap_uint<trans_bit> temp_val_1 = 0;
			int idx =0;
			in_tran:for (int h = 0; h < 16/4; h++)
			{
			#pragma HLS UNROLL
				idx = 4 * h;
				inElem_transformed[sim][idx] 	= inElem_partitioned[sim][idx] - inElem_partitioned[sim][idx + 2];
				inElem_transformed[sim][idx + 1] = inElem_partitioned[sim][idx + 1] + inElem_partitioned[sim][idx + 2];
				inElem_transformed[sim][idx + 2] = inElem_partitioned[sim][idx + 2] - inElem_partitioned[sim][idx + 1];
				inElem_transformed[sim][idx + 3] = inElem_partitioned[sim][idx + 1] - inElem_partitioned[sim][idx + 3];
			}


			for (int h = 0; h < 16/4; h++)
			{
		#pragma HLS UNROLL
				temp_val_1 = inElem_transformed[sim][h + 4];
				inElem_transformed[sim][h] = inElem_transformed[sim][h] - inElem_transformed[sim][h + 8];
				inElem_transformed[sim][h + 4] = inElem_transformed[sim][h + 4] + inElem_transformed[sim][h + 8];
				inElem_transformed[sim][h + 8] = inElem_transformed[sim][h + 8] - temp_val_1;
				inElem_transformed[sim][h + 12] = temp_val_1 - inElem_transformed[sim][h + 12];
			}


//			if(tile_num == 137)
//			{
//				cout.clear();
//				cout<<"trans: Interesting tile"<<endl;
//
//				cout<<"Tile"<<endl;
//					    for(int a =0; a<4;a++){
//					    	for(int b=0;b<4;b++)
//					    		cout << inElem_partitioned[sim][4*a+b]<<" ";
//					    	cout<<endl;
//					    }
//						std::cout.setstate(std::ios_base::failbit);
//
//			}

			for(unsigned j=0; j<16; j++)
			{
				unsigned int lowBit = j *trans_bit*next_SIMDi + sim*trans_bit;
//				unsigned int highBit = (j+1)*(BIT_W+2+extra_bit)*next_SIMDi + (BIT_W+2+extra_bit) - 1 + sim*(BIT_W+2+extra_bit);
				unsigned int highBit = j *trans_bit*next_SIMDi + sim*trans_bit + trans_bit - 1;
				outElem.range(highBit,lowBit) = inElem_transformed[sim][j].range(trans_bit-1, 0);
			}
		}

		out.write(outElem);
	}
}


#endif

