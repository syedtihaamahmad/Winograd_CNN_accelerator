#include <iostream>
using namespace std;

#include "core.h"

void nn_top_window(TI *in, hls::stream<TO_window> &out, unsigned int reps){

//void nn_top_window(TI *in, ap_uint<8*simd*4*4> *out, unsigned int reps){

//void nn_top_window(hls::stream<TI> &in, hls::stream<TO> &out){
//#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
//#pragma HLS INTERFACE s_axilite port=reps bundle=ctrl_bus
//#pragma HLS INTERFACE s_axilite port=return bundle=ctrl_bus

#pragma HLS INTERFACE m_axi offset=slave port=in bundle=inmem depth=900
#pragma HLS INTERFACE s_axilite port=in bundle=ctrl_bus
//#pragma HLS INTERFACE m_axi offset=slave port=out bundle=outmem depth=900 //*****change when changing out_parallel***//
//#pragma HLS INTERFACE s_axilite port=out bundle=ctrl_bus

#pragma HLS INTERFACE s_axilite port=reps bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl_bus




#pragma HLS DATAFLOW
hls::stream<TI> inStream_0;
	#pragma HLS STREAM variable=inStream_0 depth=900
hls::stream<ap_uint<bitw*IMG_CH>> inStream_1; //After adjusting width of stream
	#pragma HLS STREAM variable=inStream_1 depth=900
hls::stream<ap_uint<bitw*IMG_CH>> c1_in_padded;
	#pragma HLS STREAM variable=c1_in_padded depth=900

hls::stream<ap_uint<8*out_parallel>> c1_outstream;
	#pragma HLS STREAM variable=c1_outstream depth=900

//conv2 streams

hls::stream<ap_uint<8*OFM_CH_C1>> c2_in_padded;
	#pragma HLS STREAM variable=c2_in_padded depth=900
hls::stream<ap_uint<8*out_parallel_C2>> c2_outstream;
	#pragma HLS STREAM variable=c2_outstream depth=900




	Mem2Stream<MEM_BANDWIDTH, INPUT_depth>(in, inStream_0);

	//stream width=MEM_BANDWIDTH
	StreamingDataWidthConverter_Batch<TI::width, bitw*IMG_CH, INPUT_depth>
	(inStream_0, inStream_1, reps);

	//stream width=bitw*IMG_CH
	Streaming_pad<IMG_DIM, 3, 1, IMG_CH, IMG_CH, IMG_CH, ap_uint<bitw>>(inStream_1, c1_in_padded);

//	hls::stream<ap_uint<(8+2+extra_bit_C1)*simd*4*4>> inStream_tran("out_of_intransform");
//		#pragma HLS STREAM variable=inStream_tran depth=900

//	hls::stream<ap_uint<8*simd*4*4>> window_out("out_of_window");
//		#pragma HLS STREAM variable=window_out depth=900

	SlidingWindow<4, IMG_CH, 8, IFM_DIM_C1, OFM_DIM_C1, 2, IMG_CH, simd>
	(c1_in_padded, out, reps);

	//in_transform<simd, 8, IFM_DIM_C1, IMG_CH, extra_bit_C1>(window_out, inStream_tran);




//	Stream2Mem<8*simd*4*4, ((IFM_DIM_C1/2)*(IFM_DIM_C1/2))>(window_out, out);


}


