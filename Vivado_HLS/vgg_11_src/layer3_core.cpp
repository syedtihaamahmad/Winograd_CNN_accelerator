#include <iostream>
using namespace std;

#include "core.h"


//static ap_uint<simd_C0*bitw> weights_C0[OFM_CH_C0/out_parallel_C0][IMG_CH/simd_C0][pe_num_C0] = {
//#include "../data/vgg_11_pack/conv0_packed_out1.txt"
//};
static B_C0 bias_C0[OFM_CH_C0/out_parallel_C0][out_parallel_C0] = {
#include "../data/vgg_11_pack/qconv0_b.txt"
};
//static ap_uint<simd_C1*bitw> weights_C1[OFM_CH_C1/out_parallel_C1][OFM_CH_C0/simd_C1][pe_num_C1] = {
//#include "../data/vgg_11_pack/conv1_packed_out1.txt"
//};
static B_C1 bias_C1[OFM_CH_C1/out_parallel_C1][out_parallel_C1] = {
#include "../data/vgg_11_pack/qconv1_b.txt"
};
//static ap_uint<simd_C2*bitw> weights_C2[OFM_CH_C2/out_parallel_C2][OFM_CH_C1/simd_C2][pe_num_C2] = {
//#include "../data/vgg_11_pack/conv2_packed_out1.txt"
//};
static B_C2 bias_C2[OFM_CH_C2/out_parallel_C2][out_parallel_C2] = {
#include "../data/vgg_11_pack/qconv2_b.txt"
};

#if !__SYNTHESIS__

#include "mem.hpp"
ap_uint<simd_C0*bitw> ***weights_C0;
ap_uint<simd_C1*bitw> ***weights_C1;
ap_uint<simd_C2*bitw> ***weights_C2;

#else
static ap_uint<simd_C0*bitw> weights_C0[OFM_CH_C0/out_parallel_C0][IMG_CH/simd_C0][pe_num_C0] = {
#include "../data/vgg_11_pack/conv0_packed_out1.txt"
};
static ap_uint<simd_C1*bitw> weights_C1[OFM_CH_C1/out_parallel_C1][OFM_CH_C0/simd_C1][pe_num_C1] = {
#include "../data/vgg_11_pack/conv1_packed_out1.txt"
};
static ap_uint<simd_C2*bitw> weights_C2[OFM_CH_C2/out_parallel_C2][OFM_CH_C1/simd_C2][pe_num_C2] = {
#include "../data/vgg_11_pack/conv2_packed_out1.txt"
};
#endif

//static ap_uint<simd_C2*bitw> weights_C2[OFM_CH_C2/out_parallel_C2][OFM_CH_C1/simd_C2][pe_num_C2];
//static B_C2 bias_C2[OFM_CH_C2/out_parallel_C2][out_parallel_C2];
//static Fixed_Weights<simd*bitw, OFM_CH_C1/out_parallel, ifmChannels/simd, pe_num> weights_C1;
//static Fixed_Weights<simd_C2*bitw, OFM_CH_C2/out_parallel_C2, OFM_CH_C1/simd_C2, pe_num_C2> weights_C2;




//void nn_top(ap_uint<128> *in, hls::stream<TO> &out, unsigned int reps){

void doCompute( TI *in, TO_32 *out,int const reps){

#pragma HLS DATAFLOW
hls::stream<TI> inStream_0;
//	#pragma HLS STREAM variable=inStream_0 depth=30
hls::stream<ap_uint<bitw*1>> inStream_1("unit_str_docomp"); //After adjusting width of stream
//	#pragma HLS STREAM variable=inStream_1 depth=30
hls::stream<ap_uint<bitw*IMG_CH>> c0_in_padded;
//	#pragma HLS STREAM variable=c1_in_padded depth=450
hls::stream<ap_uint<bitw*out_parallel_C0>> c0_outstream;
//Conv1
hls::stream<ap_uint<bitw*OFM_CH_C0>> c1_in_padded;
hls::stream<ap_uint<bitw*out_parallel_C1>> c1_outstream;
//Conv2
hls::stream<ap_uint<bitw*OFM_CH_C1>> c2_in_padded;
hls::stream<ap_uint<out_size_C2>> c2_outstream; //*4 due to no maxpool

hls::stream<TO_32> outstream("outstream");


//static ap_uint<simd*bitw> weights_C1[OFM_CH_C1/out_parallel][ifmChannels/simd][pe_num] ={
//#include "../data/weights_bn_v1_pack/conv1_packed_out4.txt"
//};
//
//static B_C1 bias_C1[OFM_CH_C1/out_parallel][out_parallel] = {
//#include "../data/weights_bn_v1_pack/qconv1_b.txt"
//};
//
//static ap_uint<simd_C2*bitw> weights_C2[OFM_CH_C2/out_parallel_C2][OFM_CH_C1/simd_C2][pe_num_C2] ={
//#if out_parallel_C2 == 2
//		#include "../data/weights_bn_v1_pack/conv2_packed_out2.txt"
////#elif out_parallel_C2 == 1
////		#include "../data/qconv2_packed_simd2_out1.txt"
////#elif out_parallel_C2 ==4
////		#include "../data/qconv2_packed_simd2_out4.txt"
//#elif
//		assert(1)
//#endif
//};
//
//static B_C2 bias_C2[OFM_CH_C2/out_parallel_C2][out_parallel_C2] = {
//#include "../data/weights_bn_v1_pack/qconv2_b.txt"
//};


	Mem2Stream<MEM_BANDWIDTH, INPUT_depth>(in, inStream_0, reps);

	//stream width=MEM_BANDWIDTH
	StreamingDataWidthConverter_Batch<TI::width, bitw*1, INPUT_depth>
	(inStream_0, inStream_1, reps);

	//----------------------------------------//
#if !__SYNTHESIS__
	layer_features l = {OFM_CH_C0/out_parallel_C0, IMG_CH/simd_C0, pe_num_C0};
	weights_C0 = new ap_uint<simd_C0*bitw> **[l.ch];
	for(int r=0; r< l.ch;r++){
		weights_C0[r] = new ap_uint<simd_C0*bitw> *[l.row];

		for(int col=0; col<l.row; col++)
			weights_C0[r][col] = new ap_uint<simd_C0*bitw> [l.col];
	}

	load<ap_uint<simd_C0*bitw>>("/home/ab/projects/hls_projects/slide_fast_vgg_11/data/vgg_11_pack/conv0_packed_out1.txt", l, weights_C0);

#endif
	//----------------------------------------//

	//stream width=bitw*IMG_CH
	Streaming_pad<IMG_DIM, 3, 1, IMG_CH, 1, IMG_CH,
		ap_uint<bitw>>(inStream_1, c0_in_padded, reps);

	convlayer<
		4,
		2,
		bitw,
		IFM_DIM_C0,
		OFM_DIM_C0,
		IMG_CH,
		trans_bit_C0, OFM_CH_C0, simd_C0, pe_num_C0,IMG_CH,
		out_shift_C0, out_size_C0,
		ap_int<8>, ap_int<8>, ap_int<8>,has_bias_C0,
		ap_int<32>
		>(c0_in_padded, c0_outstream, weights_C0, bias_C0, reps);

	//---------------- Conv1 -----------------//
#if !__SYNTHESIS__
	// deallocate memory for conv0
	for (int i = 0; i < l.ch; i++)
	{
		for (int j = 0; j < l.row; j++)
			delete[] weights_C0[i][j];

		delete[] weights_C0[i];
	}
	delete[] weights_C0;
	//Allocating Weights for conv1
	l = {OFM_CH_C1/out_parallel_C1, OFM_CH_C0/simd_C1, pe_num_C1};
	weights_C1 = new ap_uint<simd_C1*bitw> **[l.ch];
		for(int r=0; r< l.ch;r++){
			weights_C1[r] = new ap_uint<simd_C1*bitw> *[l.row];

			for(int col=0; col<l.row; col++)
				weights_C1[r][col] = new ap_uint<simd_C1*bitw> [l.col];
		}

	load<ap_uint<simd_C1*bitw>>("/home/ab/projects/hls_projects/slide_fast_vgg_11/data/vgg_11_pack/conv1_packed_out1.txt", l, weights_C1);

#endif

	//----------------------------------------//
	Streaming_pad<OFM_DIM_C0, 3, 1, OFM_CH_C0, out_parallel_C0,
		OFM_CH_C0, ap_uint<bitw>>(c0_outstream, c1_in_padded, reps);

	convlayer<
	4,
	2,
	bitw,
	IFM_DIM_C1,
	OFM_DIM_C1,
	OFM_CH_C0,
	trans_bit_C1, OFM_CH_C1, simd_C1,pe_num_C1,OFM_CH_C0,
	out_shift_C1, out_size_C1,
	ap_int<8>, ap_int<8>, ap_int<8>, has_bias_C1,
	ap_int<32>
	>(c1_in_padded, c1_outstream, weights_C1, bias_C1, reps);

	//---------------- Conv2 -----------------//
#if !__SYNTHESIS__
	// deallocate memory for conv1
	for (int i = 0; i < l.ch; i++)
	{
		for (int j = 0; j < l.row; j++)
			delete[] weights_C1[i][j];

		delete[] weights_C1[i];
	}
	delete[] weights_C1;
	//Allocating Weights for conv2
	l = {OFM_CH_C2/out_parallel_C2, OFM_CH_C1/simd_C2, pe_num_C2};
	weights_C2 = new ap_uint<simd_C2*bitw> **[l.ch];
		for(int r=0; r< l.ch;r++){
			weights_C2[r] = new ap_uint<simd_C2*bitw> *[l.row];

			for(int col=0; col<l.row; col++)
				weights_C2[r][col] = new ap_uint<simd_C2*bitw> [l.col];
		}

	load<ap_uint<simd_C1*bitw>>("/home/ab/projects/hls_projects/slide_fast_vgg_11/data/vgg_11_pack/conv2_packed_out1.txt", l, weights_C2);


#endif


	//----------------------------------------//
	Streaming_pad<OFM_DIM_C1, 3, 1, OFM_CH_C1, out_parallel_C1,
		OFM_CH_C1, ap_uint<bitw>>(c1_outstream, c2_in_padded, reps);

	convlayer<
	4,
	2,
	bitw,
	IFM_DIM_C2,
	OFM_DIM_C2,
	OFM_CH_C1,
	trans_bit_C2, OFM_CH_C2, simd_C2,pe_num_C2,OFM_CH_C1,
	out_shift_C2, out_size_C2,
	ap_int<8>, ap_int<8>, ap_int<8>, has_bias_C2,
	ap_int<32>
	>(c2_in_padded, c2_outstream, weights_C2, bias_C2, reps);

	//----------------------------------------//

#if !__SYNTHESIS__
	// deallocate memory for conv1
	for (int i = 0; i < l.ch; i++)
	{
		for (int j = 0; j < l.row; j++)
			delete[] weights_C2[i][j];

		delete[] weights_C2[i];
	}
	delete[] weights_C2;
#endif


	//----------------------------------------//



	StreamingDataWidthConverter_Batch<TO::width, TO_32::width, (OFM_DIM_C2*OFM_DIM_C2*OFM_CH_C2)/(out_parallel_C2*4)>(c2_outstream, outstream, reps);
	Stream2Mem<TO_32::width,(OFM_DIM_C2*OFM_DIM_C2*OFM_CH_C2)/(out_parallel_C2*4) >(outstream, out, reps);


}

void DoMemInit(unsigned int targetmem,unsigned int target_ch,
		unsigned int target_row, unsigned int target_col,ap_uint<64> val){

	switch(targetmem){
	case 1:
		weights_C0[target_ch][target_row][target_col] = val;
		break;
	case 2:
		bias_C0[target_row][target_col] = val(B_C0::width-1,0);
		break;
	case 3:
		weights_C1[target_ch][target_row][target_col] = val;
		break;
	case 4:
		bias_C1[target_row][target_col] = val(B_C1::width-1,0);
		break;
	case 5:
		weights_C2[target_ch][target_row][target_col] = val;
		break;
	case 6:
		bias_C2[target_row][target_col] = val(B_C2::width-1,0);
		break;

	}
}

//
//void Read_Vals(unsigned int targetmem,unsigned int target_ch,
//		unsigned int target_row, unsigned int target_col,ap_uint<64> val){
//
//	switch(targetmem){
//	case 1:
//		val = weights_C1[target_ch][target_row][target_col];
//		break;
//	case 2:
//		val(B_C1::width-1,0) = bias_C1[target_row][target_col];
//		break;
//	case 3:
//		val = weights_C2[target_ch][target_row][target_col];
//		break;
//	case 4:
//		val(B_C2::width-1,0) = bias_C2[target_row][target_col];
//		break;
//	}
//
//}


void nn_top( TI *in, TO_32 *out,
		bool doInit,
		unsigned int targetmem,
		unsigned int target_ch, unsigned int target_row, unsigned int target_col,
		ap_uint<64> val, unsigned int numReps){

	//void nn_top(hls::stream<TI> &in, hls::stream<TO> &out){
	//#pragma HLS INTERFACE axis port=in
	//#pragma HLS INTERFACE axis port=out
	//#pragma HLS INTERFACE s_axilite port=reps bundle=ctrl_bus
	//#pragma HLS INTERFACE s_axilite port=return bundle=ctrl_bus

// signals to be mapped to the AXI master ports
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=inmem depth=384
#pragma HLS INTERFACE s_axilite port=in bundle=ctrl_bus
	//depth = (OFM_DIM_C2*OFM_DIM_C2*OFM_CH_C2)/(MEM_BANDWIDTH/8)+1
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=outmem depth=4096 //*****change when changing out_parallel***//
#pragma HLS INTERFACE s_axilite port=out bundle=ctrl_bus

//#pragma HLS INTERFACE m_axi offset=slave port=val bundle=valmem

// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=doInit bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=targetmem bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=target_ch bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=target_row bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=target_col bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=val bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=numReps bundle=ctrl_bus

//Partitioning arrays
#pragma HLS ARRAY_PARTITION variable=weights_C0 complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights_C1 complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights_C2 complete dim=3

#pragma HLS ARRAY_PARTITION variable=bias_C0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias_C1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias_C2 complete dim=2

//Array Resources
#pragma HLS RESOURCE variable=weights_C0 core=RAM_2P
#pragma HLS RESOURCE variable=weights_C1 core=RAM_2P
#pragma HLS RESOURCE variable=weights_C2 core=RAM_2P

#pragma HLS RESOURCE variable=bias_C0 core=RAM_2P_LUTRAM
#pragma HLS RESOURCE variable=bias_C1 core=RAM_2P_LUTRAM
#pragma HLS RESOURCE variable=bias_C2 core=RAM_2P_LUTRAM


	  if (doInit) {
	    DoMemInit(targetmem, target_ch, target_row, target_col, val);
	  }
	  else {
	    doCompute(in, out, numReps);
	  }


}

