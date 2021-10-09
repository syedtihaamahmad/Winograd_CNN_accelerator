#ifndef MY_CORE_H
#define MY_CORE_H

//Max bitwidth
#define AP_INT_MAX_W (8*16*16*16)

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_fixed.h>
#include <cmath>

#define IMG_CH 3
#define NUM_REPS 1
#define IMG_DIM 32
#define MEM_BANDWIDTH 64
#define Vals_per_Input (MEM_BANDWIDTH/8) //These brackets are important
#define INPUT_depth ((IMG_DIM*IMG_DIM*IMG_CH)/Vals_per_Input)
#define bitw 8

//Conv0
#define simd_C0 1
#define out_parallel_C0 1 //This is adjustable ***Must be 1,2,4,8 i.e. divisible by filter number
#define pe_num_C0 (16*out_parallel_C0)
#define IFM_DIM_C0 34
#define OFM_DIM_C0 16
#define OFM_CH_C0 32
#define trans_bit_C0 (bitw+2+1)
#define out_shift_C0 6
#define out_size_C0 (8*pe_num_C0/16)
#define has_bias_C0 true
typedef ap_int<(out_shift_C0 +8+6)> B_C0;



//Conv1
#define simd_C1 1
#define out_parallel_C1 1 //This is adjustable ***Must be 1,2,4,8 i.e. divisible by filter number
#define pe_num_C1 (16*out_parallel_C1)
#define IFM_DIM_C1 18
#define OFM_DIM_C1 8
#define OFM_CH_C1 32
#define trans_bit_C1 (bitw+2+1)
#define out_shift_C1 7
#define out_size_C1 (8*pe_num_C1/16)
#define has_bias_C1 true
typedef ap_int<(out_shift_C1 +8+6)> B_C1;


//Conv2
#define simd_C2 1
#define out_parallel_C2 1 //This is adjustable ***Must be 1,2,4,8 i.e. divisible by filter number
#define pe_num_C2 (16*out_parallel_C2)
#define IFM_DIM_C2 10
#define OFM_DIM_C2 4
#define OFM_CH_C2 32
#define trans_bit_C2 (bitw+2+1)
#define out_shift_C2 8
#define out_size_C2 (8*pe_num_C2/16) 
#define has_bias_C2 true
typedef ap_int<(out_shift_C2 +8+6)> B_C2;


//Conv3
#define simd_C3 1
#define out_parallel_C3 1 //This is adjustable ***Must be 1,2,4,8 i.e. divisible by filter number
#define pe_num_C3 (16*out_parallel_C3)
#define IFM_DIM_C3 6
#define OFM_DIM_C3 2
#define OFM_CH_C3 64
#define trans_bit_C3 (bitw+2+1)
#define out_shift_C3 8
#define out_size_C3 (8*pe_num_C3/16)
#define has_bias_C3 true
typedef ap_int<(out_shift_C3+8+6)> B_C3;


//Conv4
#define simd_C4 1
#define out_parallel_C4 1 //This is adjustable ***Must be 1,2,4,8 i.e. divisible by filter number
#define pe_num_C4 (16*out_parallel_C4)
#define IFM_DIM_C4 4
#define OFM_DIM_C4 1
#define OFM_CH_C4 128
#define trans_bit_C4 (bitw+2+1)
#define out_shift_C4 8
#define out_size_C4 (8*pe_num_C4/16)
#define has_bias_C4 true
typedef ap_int<(out_shift_C4+8+6)> B_C4;




typedef ap_uint<MEM_BANDWIDTH> TI;
typedef ap_uint<out_size_C4> TO;
typedef ap_uint<32> TO_32;
typedef ap_int<8> TW;
//, typename R


//template<
//  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned int IFMChannels,
//  typename TSrcI , typename TDstI , typename TWeightI ,
//  typename Accum_type,
//  typename TI, typename TO//, typename TW//, typename R
//>
//void wino_conv(
//		hls::stream<TI> &in,
//		hls::stream<TO> &out,
//		int  reps);

//template<unsigned next_SIMDi>
//void in_transform(hls::stream<ap_uint<8*next_SIMDi*4*4>> &in,
//		hls::stream<ap_uint<(8+2)*next_SIMDi*4*4>> &out);
//
//
//
//template<unsigned int DataWidth, unsigned int numBytes>
//void Mem2Stream(ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out);
//
//template<unsigned int DataWidth, unsigned int numBytes>
//void Stream2Mem(hls::stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out);
//
//template<unsigned int InWidth, unsigned int OutWidth, unsigned int NumInWords>
//void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth> > & in,
//		hls::stream<ap_uint<OutWidth> > & out, const unsigned int numReps);

//void nn_top(ap_uint<128> *in, hls::stream<TO> &out, unsigned int reps);
//void nn_top(TI *in, TO *out, unsigned int reps);
//void nn_top(hls::stream<TI> &in, hls::stream<TO> &out);
void nn_top( TI *in, TO_32 *out, bool doInit,
		unsigned int targetmem,
		unsigned int target_ch, unsigned int target_row,
		unsigned int target_col, ap_uint<64> val, unsigned int numReps);

//Other function testbenches
//#define TO_window ap_uint<8*simd_C1*4*4>
//#define TO_layer1 ap_uint<8*out_parallel_C1>

//void nn_top_window(TI *in, hls::stream<TO_window> &out, unsigned int reps);
//void nn_top_layer1(TI *in, TO_layer1 *out, unsigned int reps);
//void nn_top_layer1(TI *in, hls::stream<TO_layer1> &out, unsigned int reps);




//My libraries
#include "../lib/stream_utils.hpp"
#include "../lib/in_transform.hpp"
//#include "../lib/slidingwindow.hpp"
//#include "../lib/electric_slide.hpp"
#include "../lib/slidingwindow_orig.hpp"
#include "../lib/wino_conv.hpp"
#include "../lib/weight.hpp"

#endif
