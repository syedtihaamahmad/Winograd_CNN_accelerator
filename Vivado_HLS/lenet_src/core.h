#ifndef MY_CORE_H
#define MY_CORE_H

//Max bitwidth
#define AP_INT_MAX_W (4*4*8*16)

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_fixed.h>
#include <cmath>

#define IMG_CH 1
#define NUM_REPS 1
#define IMG_DIM 28
#define MEM_BANDWIDTH 64
#define Vals_per_Input (MEM_BANDWIDTH/8) //These brackets are important
#define INPUT_depth ((IMG_DIM*IMG_DIM*IMG_CH)/Vals_per_Input)


#define bitw 8
#define simd 1
#define out_parallel 2 //This is adjustable ***Must be 1,2,4,8 i.e. divisible by filter number
#define pe_num (16*out_parallel)
#define ifmChannels 1
#define IFM_DIM_C1 30
#define OFM_DIM_C1 14
#define OFM_CH_C1 16
#define trans_bit_C1 (bitw+2+0)
//#define extra_bit_C1 0
#define out_shift_C1 6
#define out_size_C1 (8*pe_num/16)
#define has_bias_C1 true
typedef ap_int<(out_shift_C1+8)> B_C1;

//Conv2 parameters
#define simd_C2 2 //This needs to be 1 for
#define out_parallel_C2 2 //This is also adjustable but weights will have to be generated again. Divisible by filter number
#define pe_num_C2 (16*out_parallel_C2)
#define IFM_DIM_C2 16
#define OFM_DIM_C2 7
#define OFM_CH_C2 20
#define trans_bit_C2 (bitw+2+1)
#define extra_bit_C2 1
#define out_shift_C2 7
#define out_size_C2 (8*pe_num_C2/16)
#define has_bias_C2 true
typedef ap_int<(out_shift_C2+8)> B_C2;

//These are dummy values to test 1st layer only
//#define simd_C2 2 //This needs to be 1 for
//#define out_parallel_C2 OFM_CH_C1
//#define pe_num_C2 16*out_parallel_C2
//#define IFM_DIM_C2 30
//#define OFM_DIM_C2 16
//#define OFM_CH_C2 16


//#define MatrixW 1
//#define MatrixH 16
//#define SIMD 1
//#define PE 16
//#define TILES 32/(PE*SIMD)
//#define IFMChannels 1
#define next_simd 1
//typedef ap_int<8> TSrcI;
//typedef ap_int<8> TDstI;
//typedef ap_int<8> TWeightI;
//typedef ap_int<32> Accum_type;
typedef ap_uint<MEM_BANDWIDTH> TI;
typedef ap_uint<8*out_parallel_C2> TO;
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
#define TO_window ap_uint<8*simd*4*4>
#define TO_layer1 ap_uint<8*out_parallel>

void nn_top_window(TI *in, hls::stream<TO_window> &out, unsigned int reps);
void nn_top_layer1(TI *in, TO_layer1 *out, unsigned int reps);
//void nn_top_layer1(TI *in, hls::stream<TO_layer1> &out, unsigned int reps);




//My libraries
#include "../lib/stream_utils.hpp"
#include "../lib/in_transform.hpp"
#include "../lib/slidingwindow.hpp"
#include "../lib/wino_conv.hpp"
#include "../lib/weight.hpp"

#endif
