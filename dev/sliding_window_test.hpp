#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;


template<
	unsigned OFM_dim,
	unsigned IFM_channels, //All channels of input must be packed in input
	unsigned vals_per_output,
	unsigned int Input_pre
>
void window_slicer(
		hls::stream<ap_uint<IFM_channels*Input_pre*4*4> > & in,
				hls::stream<ap_uint<vals_per_output*Input_pre*4*4> > & out){

#if !__SYNTHESIS__
	assert(IFM_channels%vals_per_output == 0);
#endif
	const unsigned int FOLD = IFM_channels/vals_per_output;

	ap_uint<IFM_channels*Input_pre*4*4> inElem, temp_holder;
	ap_uint<vals_per_output*Input_pre*4*4> inElem_partitioned[FOLD]; //Partitioned along depth
#pragma HLS ARRAY_PARTITION variable=inElem_partitioned complete dim=0

	ap_uint<vals_per_output*Input_pre*4*4> outElem;

	const unsigned int TOTAL_ITER = OFM_dim * OFM_dim * FOLD;



	unsigned tile = 0;
	unsigned r=0, c=0;
	unsigned nf = FOLD;


	for(unsigned int i=0; i<TOTAL_ITER; i++){
#pragma HLS PIPELINE II=1
		ap_uint<Input_pre*vals_per_output*4*4> outElem;

		if (nf == FOLD){
			nf=0;
			inElem = in.read();
			//*Input_buf = temp_holder;
			//cout<<Input_buf[0][1];

			for(int f=0; f<FOLD; f++){
				temp_holder = inElem;
				inElem = inElem >> f*vals_per_output*Input_pre;
				for(int d=0; d<16; d++)//******** 16 due to size of window ***********//
				{
	#pragma HLS UNROLL
					unsigned int lowBit = d*vals_per_output*Input_pre;// + jdx*8*next_SIMD has value of next neuron
					unsigned int highBit = (d+1)*vals_per_output*Input_pre -1;
					inElem_partitioned[f].range(highBit, lowBit) = inElem.range(vals_per_output*Input_pre-1, 0);
					inElem = inElem >> IFM_channels*Input_pre;
				}
				inElem = temp_holder;
			}
		}


		out.write(inElem_partitioned[nf++]);


	}

}





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

	int in_count=0;
	unsigned buf_row = 0;
	unsigned buf_col = 0;
	unsigned next_col = 0;

	ap_uint<1> out_writing = 0; //1 means writing output
	ap_uint<1> unpacked = 1; //1 means output is unpacked

	//tile ready variables
	int tile_count = 0;
	unsigned tile_row = 0;
	unsigned tile_col = 0;
	unsigned output_buf_row = 0;
	unsigned output_buf_col = 0;



	const unsigned TILE_ver=((IFMDim-2)/2);
	const unsigned TILE_hor=((IFMDim-2)/2);
//	const unsigned mask = log2(IFMDim);
	unsigned total=0;
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
//			unsigned col_rem = total;
//			while(col_rem >= IFMDim){
//				col_rem = col_rem - IFMDim;
//			}


		buf_row = (total/IFMDim) % ConvKernelDim;
//		buf_col = total % IFMDim;
		buf_col = next_col;


		in_buf[buf_row][buf_col]=in.read();

//		if(iter == (16*3+2))
//			cout<<"Interesting"<<endl;



#pragma AP dependence variable=buf_row intra false
#pragma AP dependence variable=buf_col intra false
#pragma AP dependence variable=total intra false


		total++;
//		buf_row++;
		next_col = buf_col;
		next_col++;
//		if(buf_row >= IFMDim)
//			buf_row=0;


		}

		if(total >= IFMDim*3+2 && ((buf_row+1)&0x1)==0 && ((buf_col+1)&0x1)==0 && (buf_col)>=3)
		{
				out_writing = 1;
				unpacked = 0;

		}

		if(next_col >= IFMDim)
			next_col=0;


//		if(buf_col >= IFMDim){
//					buf_col = 0;
//		//			buf_row ++;
//				}

#pragma AP dependence variable=next_col intra false


//		if(!out_writing)
//			fold=0;

//		cout<<"in"<<total+1<<":"<<hex<<in_buf[buf_row][buf_col]<<" "<<dec;

//		if(in_buf[buf_row][buf_col] == 42 || in_buf[buf_row][buf_col] == 33)
//			cout<<"Start"<<endl;

		if(out_writing){


//			cout<<endl;
//			 cout<<"Tile: "<<dec<<tile_count+1<<endl;

//			 if(tile_count+1 == 62)
//				 cout<<"interest"<<endl;

//			if(!unpacked){
//				 int r=tile_count / TILE_ver;
//				 int c=tile_count % TILE_hor;
//				 int r=tile_row;
//				 int c=tile_col;


//				 for(int sim=0; sim<FOLD; sim++){
//			cout<<"tile_row>>1 &0x1: "<< ((tile_row>>1) & 0x1) <<endl;

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
//					 out_array[sim] = outElem;
	//				 cout<<hex<<outElem<<endl;
//				 }
//				 unpacked = 1;
//			}
#pragma AP dependence variable=unpacked intra false




//			out.write(out_array[fold]);
//#if !__SYNTHESIS__
//				 window_out << hex << out_array[fold] <<","<<endl;
//#endif

//				 fold++;
			 if(++fold == FOLD)
				 {
				 	 tile_count++;
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

	if(IFMChannels != nextSIMD){
		hls::stream<ap_uint<SIMD*Input_precision*4*4>> inter_stream("stream between slide and slice");
		//#pragma HLS STREAM variable=inter_stream depth=196

		//****** Pass NextSIMD=SIMD in slide and uncomment slicer for ******//
		//******     task parallelism. WARNING: huge inter_stream     *****//
		slide< 4, Input_precision, IFMDim, SIMD, Stride, nextSIMD>( in, out, numReps);


//		window_slicer<OFMDim, SIMD, nextSIMD, Input_precision>(inter_stream, out);
	}
	else{

		slide< 4, Input_precision, IFMDim, SIMD, Stride, nextSIMD>(in, out, numReps);
	}

}


#endif


