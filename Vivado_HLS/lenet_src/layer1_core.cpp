#include <iostream>
using namespace std;

#include "core.h"



//void nn_top_layer1(TI *in, hls::stream<TO_layer1> &out, unsigned int reps){

void nn_top_layer1(TI *in, TO_layer1 *out, unsigned int reps){

//void nn_top_layer1(hls::stream<TI> &in, hls::stream<TO> &out){
//#pragma HLS INTERFACE axis port=in
//#pragma HLS INTERFACE axis port=out
//#pragma HLS INTERFACE s_axilite port=reps bundle=ctrl_bus
//#pragma HLS INTERFACE s_axilite port=return bundle=ctrl_bus

#pragma HLS INTERFACE m_axi offset=slave port=in bundle=inmem depth=98
#pragma HLS INTERFACE s_axilite port=in bundle=ctrl_bus
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=outmem depth=784 //*****change when changing out_parallel***//
#pragma HLS INTERFACE s_axilite port=out bundle=ctrl_bus

#pragma HLS INTERFACE s_axilite port=reps bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl_bus




#pragma HLS DATAFLOW
hls::stream<TI> inStream_0;
	#pragma HLS STREAM variable=inStream_0 depth=30
hls::stream<ap_uint<bitw*IMG_CH>> inStream_1; //After adjusting width of stream
	#pragma HLS STREAM variable=inStream_1 depth=30
hls::stream<ap_uint<bitw*IMG_CH>> c1_in_padded;
	#pragma HLS STREAM variable=c1_in_padded depth=450

hls::stream<ap_uint<8*out_parallel>> c1_outstream;
	#pragma HLS STREAM variable=c1_outstream depth=450

//conv2 streams

hls::stream<ap_uint<8*OFM_CH_C1>> c2_in_padded;
	#pragma HLS STREAM variable=c2_in_padded depth=450
hls::stream<ap_uint<8*out_parallel_C2>> c2_outstream;
	#pragma HLS STREAM variable=c2_outstream depth=450



static ap_uint<simd*bitw> weights_C1[OFM_CH_C1/out_parallel][ifmChannels/simd][pe_num] ={
#include "../data/weights_bn_v1_pack/conv1_packed_out4.txt"
};

static B_C1 bias_C1[OFM_CH_C1/out_parallel][out_parallel] = {
#include "../data/weights_bn_v1_pack/qconv1_b.txt"
};



	Mem2Stream<MEM_BANDWIDTH, INPUT_depth>(in, inStream_0);

	//stream width=MEM_BANDWIDTH
	StreamingDataWidthConverter_Batch<TI::width, bitw*IMG_CH, INPUT_depth>
	(inStream_0, inStream_1, reps);

	//stream width=bitw*IMG_CH
	Streaming_pad<IMG_DIM, 3, 1, IMG_CH, IMG_CH, IMG_CH, ap_uint<bitw>>(inStream_1, c1_in_padded);

	convlayer<
		4,
		2,
		bitw,
		IFM_DIM_C1,
		OFM_DIM_C1,
		IMG_CH,
		extra_bit_C1, OFM_CH_C1, simd, pe_num,IMG_CH,
		6, out_size_C1,
		ap_int<8>, ap_int<8>, ap_int<8>,has_bias_C1,
		ap_int<32>
		>(c1_in_padded, c1_outstream, weights_C1, bias_C1, reps);




	//*********** Replace "out" with "c1_outstream" above ***************//

	//This padding function is working fine
//	Streaming_pad<OFM_DIM_C1, 3, 1, OFM_CH_C1, out_parallel, OFM_CH_C1, ap_uint<bitw>>(c1_outstream, c2_in_padded);
//
//	convlayer<
//	4,
//	2,
//	bitw,
//	IFM_DIM_C2,
//	OFM_DIM_C2,
//	OFM_CH_C1,
//	extra_bit_C2, OFM_CH_C2, simd_C2,pe_num_C2,OFM_CH_C1,
//	7,out_size_C2,
//	ap_int<8>, ap_int<8>, ap_int<8>,
//	ap_int<32>
//	>(c2_in_padded, c2_outstream, weights_C2, reps);



	Stream2Mem<TO_layer1::width, (OFM_DIM_C1*OFM_DIM_C1*OFM_CH_C1)/out_parallel>(c1_outstream, out);


}

