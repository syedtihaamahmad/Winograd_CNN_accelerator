#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H
#include <string>
#include <iostream>
#include <fstream>
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
	unsigned nf = 0;


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
//#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=6 dim=2

//#pragma HLS ARRAY_PARTITION variable=in_buf complete dim=0

//#pragma HLS RESOURCE variable in_buf core=RAM_2P

#if !__SYNTHESIS__
	assert(SIMD%nextSIMD == 0);
#endif

	const unsigned FOLD = SIMD/nextSIMD;

	int in_count=0;
	int buf_row = 0;
	int buf_col = 0;

	//tile ready variables
	int tile_count=0;
	int TILE_ver=((IFMDim-2)/2);
	int TILE_hor=((IFMDim-2)/2);

	cout<<"Starting slider"<<endl;
	//Bigger overall loop
	for(int total = 0; total < IFMDim*IFMDim; total++){
#pragma HLS PIPELINE II=1

		ap_uint<Input_precision*nextSIMD*4*4> outElem;
		ap_uint<Input_precision*nextSIMD*4*4> out_array[FOLD];

		buf_row = (total/IFMDim) % ConvKernelDim;
		buf_col = total % IFMDim;

		in_buf[buf_row][buf_col]=in.read();

//		cout<<"in"<<total+1<<":"<<hex<<in_buf[buf_row][buf_col]<<" "<<dec;

//		if(in_buf[buf_row][buf_col] == 42 || in_buf[buf_row][buf_col] == 33)
//			cout<<"Start"<<endl;

		if(total >= IFMDim*3+3 && (buf_row+1)%Stride==0 && (buf_col+1)%Stride==0 && (buf_col+1)>=3){
//			cout<<endl;
//			 cout<<"Tile: "<<dec<<tile_count+1<<endl;

//			 if(tile_count+1 == 62)
//				 cout<<"interest"<<endl;

			 int r=tile_count / TILE_ver;
			 int c=tile_count % TILE_hor;
			 for(int sim=0; sim<FOLD; sim++){

				 for(int row=0; row<4; row++)
					for(int col = 0; col<4; col++)
					{
#pragma HLS UNROLL
						unsigned int lowBit = (col+row*4)*nextSIMD*Input_precision;
						unsigned int highBit = (col+row*4+1)*nextSIMD*Input_precision - 1;

						unsigned int lowbit_simd = sim*nextSIMD*Input_precision;
						unsigned int highbit_simd = (sim+1)*nextSIMD*Input_precision-1;
						outElem.range(highBit, lowBit) =
								in_buf[(r*Stride+row)%4][(c*Stride+col)%IFMDim].range(highbit_simd,lowbit_simd);
					}

//				 out.write(outElem);
				 out_array[sim] = outElem;
//				 cout<<hex<<outElem<<endl;
			 }


			 for(int p=0; p<FOLD; p++)
			 {
				 out.write(out_array[p]);
#if !__SYNTHESIS__
				 window_out << hex << out_array[p] <<","<<endl;
#endif
			 }
		tile_count++;
		}
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

