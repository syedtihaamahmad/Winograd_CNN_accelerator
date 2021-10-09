#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H
#include <string>
#include <iostream>
#include <fstream>
using namespace std;


template<unsigned int ConvKernelDim,
		 unsigned int Input_precision,
		 unsigned int IFMDim,
		 unsigned int SIMD,
		 unsigned int Stride,
		 unsigned int nextSIMD>
void slide(
		hls::stream<ap_uint<SIMD*Input_precision> > & in,
		hls::stream<ap_uint<Input_precision*nextSIMD*4*4> > & out,
		const unsigned int numReps){

#if !__SYNTHESIS__
	string name;
	if(IFMDim == 30)
		name = "../dbg/window_c1.txt";
	else if(IFMDim == 16)
		name = "../dbg/window_c2.txt";

	ofstream window_out(name);
#endif

	ap_uint<SIMD*Input_precision> in_buf[ConvKernelDim][IFMDim] = {0};
#pragma HLS ARRAY_PARTITION variable=in_buf complete dim=0
//#pragma HLS RESOURCE variable=in_buf core=RAM_2P_LUTRAM

//#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=6 dim=2

//#pragma HLS ARRAY_PARTITION variable=in_buf complete dim=0


#if !__SYNTHESIS__
	assert(SIMD%nextSIMD == 0);
#endif

	const unsigned FOLD = SIMD/nextSIMD;

	ap_uint<2> buf_row = 0;
	ap_uint<8> buf_col = 0;
	unsigned next_col = 0;

	ap_uint<1> out_writing = 0; //1 means writing output
	ap_uint<1> unpacked = 1; //1 means output is unpacked
	ap_uint<1> initial = 0; //Initial values passed
	ap_uint<1> col_begin = 0; //Initial column passed

	//tile ready variables
	unsigned tile_row = 0;
	unsigned tile_col = 0; //Do not make this ap_uint<8>



	const unsigned TILE_ver=((IFMDim-2)/2);
	const unsigned TILE_hor=((IFMDim-2)/2);
	unsigned fold=0;

	unsigned int TOTAL_ITER = (FOLD==1)? (IFMDim*IFMDim): (IFMDim*IFMDim+TILE_ver*TILE_hor*(FOLD-1));


	cout<<"Starting slider"<<endl;
	//Bigger overall loop
	for(int iter = 0; iter < TOTAL_ITER; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS dependence array intra RAW true

//		unsigned fold;
		ap_uint<Input_precision*nextSIMD*4*4> outElem;
//		ap_uint<Input_precision*nextSIMD*4*4> out_array[FOLD];
//#pragma HLS RESOURCE variable=out_array core=RAM_2P_BRAM

		if(!(out_writing)){


//		buf_row = (total/IFMDim) % ConvKernelDim;
//		buf_col = total % IFMDim;


		in_buf[buf_row][buf_col]=in.read();




#pragma AP dependence variable=buf_row intra false
//#pragma AP dependence variable=buf_col intra false




		if(iter >= IFMDim*3+2)
			initial = 1;

		if((buf_col)>=3)
			col_begin = 1;
		else
			col_begin = 0;

		if(initial && ((buf_row+1)&0x1)==0 && ((buf_col+1)&0x1)==0 && col_begin)
				{
						out_writing = 1;
						unpacked = 0;

				}

		if(++buf_col >= IFMDim){
							buf_col = 0;
							buf_row++;
							//Because buf_row is 2 bit
//							if(++buf_row == ConvKernelDim)
//								buf_row = 0;
						}

		}



		if(out_writing){

			if(((tile_row>>1) & 0x1) == 0) //If tile row is even
					 for(int row=0; row<4; row++)
						for(int col = 0; col<4; col++)
						{
#pragma HLS UNROLL
							unsigned int lowBit = (col+row*4)*nextSIMD*Input_precision;
							unsigned int highBit = (col+row*4+1)*nextSIMD*Input_precision - 1;

							unsigned int lowbit_simd = fold*nextSIMD*Input_precision;
							unsigned int highbit_simd = (fold+1)*nextSIMD*Input_precision-1;

							outElem.range(highBit, lowBit) =
									in_buf[(0+row)][(tile_col+col)].range(highbit_simd,lowbit_simd);
						}
			else //If tile row is odd
			{
				 for(int row=0; row<2; row++)
					for(int col = 0; col<4; col++)
					{
#pragma HLS UNROLL
						unsigned int lowBit = (col+row*4)*nextSIMD*Input_precision;
						unsigned int highBit = (col+row*4+1)*nextSIMD*Input_precision - 1;

						unsigned int lowbit_simd = fold*nextSIMD*Input_precision;
						unsigned int highbit_simd = (fold+1)*nextSIMD*Input_precision-1;
						outElem.range(highBit, lowBit) =
								in_buf[(0+row+2)][(tile_col+col)].range(highbit_simd,lowbit_simd);
					}

				 for(int row=2; row<4; row++)
					for(int col = 0; col<4; col++)
					{
#pragma HLS UNROLL
						unsigned int lowBit = (col+row*4)*nextSIMD*Input_precision;
						unsigned int highBit = (col+row*4+1)*nextSIMD*Input_precision - 1;

						unsigned int lowbit_simd = fold*nextSIMD*Input_precision;
						unsigned int highbit_simd = (fold+1)*nextSIMD*Input_precision-1;
						outElem.range(highBit, lowBit) =
								in_buf[(0+row-2)][(tile_col+col)].range(highbit_simd,lowbit_simd);
					}

			}



					 out.write(outElem);

#pragma AP dependence variable=unpacked intra false




//			out.write(out_array[fold]);
//#if !__SYNTHESIS__
//				 window_out << hex << out_array[fold] <<","<<endl;
//#endif

			 if(++fold == FOLD)
				 {
				 	 tile_col+=Stride;
				 	 fold = 0;
				 	 out_writing = 0;
				 	 if(tile_col >= IFMDim-2){
				 		 tile_col = 0;
				 		 tile_row+=Stride;
				 	 }
				 }
#pragma AP dependence variable=fold intra false


		}
//#pragma AP dependence variable=out_writing intra false

	}

#if !__SYNTHESIS__
				 window_out.close();
#endif

}



template<unsigned int ConvKernelDim,
		 unsigned int IFMChannels,
		 unsigned int Input_precision,
		 unsigned int IFMDim,
		 unsigned int OFMDim,
		 unsigned int Stride,
		 unsigned int SIMD, //This is equal to IFMChannels
		 unsigned int nextSIMD>
void SlidingWindow(
		hls::stream<ap_uint<SIMD*Input_precision> > & in,
		hls::stream<ap_uint<Input_precision*nextSIMD*4*4> > & out,
		const unsigned int numReps){
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=numReps bundle=ctrl_bus
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl_bus

#pragma HLS INLINE
//#pragma HLS DATAFLOW

		slide< 4, Input_precision, IFMDim, SIMD, Stride, nextSIMD>( in, out, numReps);

}


#endif



