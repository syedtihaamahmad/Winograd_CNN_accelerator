#include <iostream>
using namespace std;

#include "core.h"

ap_uint<64> weights_C1[OFM_CH_C1/out_parallel][ifmChannels/simd][pe_num] ={
#include "../data/weights_bn_v1_pack/conv1_packed_out4.txt"
};

ap_uint<64> bias_C1[OFM_CH_C1/out_parallel][out_parallel] = {
#include "../data/weights_bn_v1_pack/qconv1_b.txt"
};

ap_uint<64> weights_C2[OFM_CH_C2/out_parallel_C2][OFM_CH_C1/simd_C2][pe_num_C2] ={
#if out_parallel_C2 == 2
		#include "../data/weights_bn_v1_pack/conv2_packed_out2.txt"
#elif
		assert(1)
#endif
};

 ap_uint<64> bias_C2[OFM_CH_C2/out_parallel_C2][out_parallel_C2] = {
#include "../data/weights_bn_v1_pack/qconv2_b.txt"
};



void load_parameters(int model){

	//These 2 are useless. Just used to run nn_top
	TO_32 outStream[(OFM_DIM_C2*OFM_DIM_C2*OFM_CH_C2*NUM_REPS)/4];
	TI packed_image[INPUT_depth*NUM_REPS];

	switch(model){
	/*
	 *
	case 1: //without bias
		cout<<"***** Loading lenet without biases ******"<<endl;
		ap_uint<simd*bitw> weights_C1[OFM_CH_C1/out_parallel][ifmChannels/simd][pe_num] ={
		#include "../data/lenet_orig/conv1_packed_out4.txt"
		};

		ap_uint<simd_C2*bitw> weights_C2[OFM_CH_C2/out_parallel_C2][OFM_CH_C1/simd_C2][pe_num_C2] ={
		#if out_parallel_C2 == 2
				#include "../data/lenet_orig/conv2_packed_out2.txt"
		#elif out_parallel_C2 == 1
				#include "../data/lenet_orig/qconv2_packed_simd2_out1.txt"
		#elif out_parallel_C2 ==4
				#include "../data/lenet_orig/qconv2_packed_simd2_out4.txt"
		#elif
				assert(1)
		#endif
		};

		//Initializing weight_C1
		for(int i=0; i<(OFM_CH_C1/out_parallel); i++)
			for(int r=0; r<ifmChannels/simd; r++)
				for(int c=0; c<pe_num; c++)
					nn_top(packed_image, outStream, true , false,
							1, i, r, c, weights_C1[i][r][c], 1);

		//Initializing weight_C2
		for(int i=0; i<(OFM_CH_C2/out_parallel_C2); i++)
			for(int r=0; r<OFM_CH_C1/simd_C2; r++)
				for(int c=0; c<pe_num_C2; c++)
					nn_top(packed_image, outStream, true , false,
							2, i, r, c, weights_C2[i][r][c], 1);

		break;
		*/
	case 2:
		cout<<"***** Loading lenet with biases ******"<<endl;


		cout<<"Initializing weight_C1"<<endl;
		for(int i=0; i<(OFM_CH_C1/out_parallel); i++)
			for(int r=0; r<ifmChannels/simd; r++)
				for(int c=0; c<pe_num; c++)
					nn_top(packed_image, outStream, true ,
							1, i, r, c, weights_C1[i][r][c], 1);

		cout<<"Initializing weight_C2"<<endl;
		for(int i=0; i<(OFM_CH_C2/out_parallel_C2); i++)
			for(int r=0; r<OFM_CH_C1/simd_C2; r++)
				for(int c=0; c<pe_num_C2; c++)
					nn_top(packed_image, outStream, true,
							3, i, r, c, weights_C2[i][r][c], 1);

		cout<<"Initializing bias_C1"<<endl;
		for(int r=0; r<OFM_CH_C1/out_parallel; r++)
			for(int c=0; c<out_parallel; c++)
				nn_top(packed_image, outStream, true,
							2, 0, r, c, bias_C1[r][c], 1);

		cout<<"Initializing bias_C2"<<endl;
		for(int r=0; r<OFM_CH_C2/out_parallel_C2; r++)
			for(int c=0; c<out_parallel_C2; c++)
				nn_top(packed_image, outStream, true,
					4, 0, r, c, bias_C2[r][c], 1);

		cout<<"Done Initializations ..."<<endl;

	}
}


/*Depreciated*/
//int read_parameters(){
//	TO_32 outStream[(OFM_DIM_C2*OFM_DIM_C2*OFM_CH_C2)/4];
//	TI packed_image[INPUT_depth];
//	ap_uint<64> val=1234;
//
//	cout<<"Reading weight_C1"<<endl;
//	for(int i=0; i<(OFM_CH_C1/out_parallel); i++)
//		for(int r=0; r<ifmChannels/simd; r++)
//			for(int c=0; c<pe_num; c++)
//				{
//					nn_top(packed_image, outStream, false ,true,
//						1, i, r, c, &val, 1);
//					if(val != weights_C1[i][r][c])
//						return 1;
//				}
//
//	cout<<"Reading weight_C2"<<endl;
//	for(int i=0; i<(OFM_CH_C2/out_parallel_C2); i++)
//		for(int r=0; r<OFM_CH_C1/simd_C2; r++)
//			for(int c=0; c<pe_num_C2; c++){
//				nn_top(packed_image, outStream, false ,true,
//						3, i, r, c, &val, 1);
//				if(val != weights_C2[i][r][c])
//						return 1;
//			}
//
//	cout<<"Reading bias_C1"<<endl;
//	for(int r=0; r<OFM_CH_C1/out_parallel; r++)
//		for(int c=0; c<out_parallel; c++)
//			{
//				nn_top(packed_image, outStream, false ,true,
//						2, 0, r, c, &val, 1);
//				if(B_C1(val) != bias_C1[r][c])
//						return 1;
//			}
//
//	cout<<"Reading bias_C2"<<endl;
//	for(int r=0; r<OFM_CH_C2/out_parallel_C2; r++)
//		for(int c=0; c<out_parallel_C2; c++)
//			{
//				nn_top(packed_image, outStream, false ,true,
//						4, 0, r, c, &val, 1);
//				if(B_C2(val) != bias_C2[r][c])
//						return 1;
//			}
//
//	cout<<"Done Readings ..."<<endl;
//	return 0;
//
//}

