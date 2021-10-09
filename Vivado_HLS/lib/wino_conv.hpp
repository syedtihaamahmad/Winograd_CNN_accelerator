#ifndef WINO_CONV_H
#define WINO_CONV_H

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

//ofstream conv1_out("./dbg/conv1_out.txt");
//ofstream conv2_out("./dbg/conv2_out.txt");


template<unsigned N, typename T, typename TC, typename TD>
T mac(T const &a, TC const &c, TD const &d);

template<
  unsigned int trans_bit, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned int IFMChannels,
  unsigned int IFM_dim, unsigned int out_shift, unsigned int output_size,//depends on maxpool
  typename TSrcI , typename TDstI , typename TWeightI ,
  typename Accum_type = ap_int<64>,bool has_bias,
  typename TI, typename TO, typename TW, typename TB//, typename R
>
void wino_conv(
		hls::stream<TI> &in,
		hls::stream<TO> &out,
		TW  const &weights,
		TB const &bias,
		int const reps){

#if !__SYNTHESIS__
	string out_name;
	string in_name;
	if(MatrixH == 128){
		out_name = "../dbg/conv4_out.txt";
		in_name = "../dbg/conv4_in.txt";
	}
	else if(MatrixH == 64){
		out_name = "../dbg/conv3_out.txt";
		in_name = "../dbg/conv3_in.txt";
	}

	ofstream conv_out(out_name);
	ofstream conv_in(in_name);
#endif

	//MatrixH represents Outputs i.e. how many times to apply filters e.g. PE=2, MatrixH=16,
//runs 8 times and 2 outputs per run
	unsigned const  NF = MatrixH/(PE/16);
	//MatrixW represents 1 tile but complete IFM channels (size = 4x4xIFM_channel)
	//unsigned const  SF = MatrixW / SIMD;
	//Depth fold: how many times to repeat to cover complete depth of input
	unsigned const DF = IFMChannels / SIMD;

	TI  inputBuf[DF];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

	Accum_type accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

	//ap_uint<8> output; //Each of size 8bit * 4

	unsigned  nf   = 0;
//#pragma HLS RESOURCE variable=nf core=AddSubnS
	unsigned  df   = 0;
	unsigned  tile = 0; // invariant: tile = nf*SF + sf
	unsigned tile_num = 0;

	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelining the way we want
	unsigned const TOTAL_FOLD = NF * DF * (IFM_dim/2 - 1) * (IFM_dim/2 - 1); //*** Not always same as OFMDim
//	cout<<"Conv layer: TOTAL_FOLD = "<<TOTAL_FOLD<<endl;
	for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS LOOP_TRIPCOUNT min=2704 max=2704
//#pragma HLS UNROLL factor=1
#pragma HLS PIPELINE II=1
		TI  inElem, temp_inElem;
		ap_uint<SIMD*trans_bit> inElem_partitioned[PE];
#pragma HLS ARRAY_PARTITION variable=inElem_partitioned complete dim=0


		if(nf == 0) {
		  // This will read in stream of size <8bit * SIMD * 4*4>
		  inElem = in.read(); //Input stream
#if !__SYNTHESIS__
		  conv_in <<hex<<inElem<<endl;
#endif
		  tile_num++;
//		  cout<<"Tile_number: "<<tile_num<<endl;
		  // store in appropriate buffer for reuse
		  inputBuf[df] = inElem; //A single value of inputBuf has SIMD input values
		}
		else {
		  // reuse buffered input
		  inElem = inputBuf[df];
		}

//		if(tile_num == 137 || tile_num == 138){
//			    	    		cout<<"Interesting tile: TotalFOLD = "<<TOTAL_FOLD<<endl;
//			    	    		cout.clear();
//			    	    	}
//			    else
//			    	cout.setstate(std::ios_base::failbit);
//		if(tile_num >= 79 and tile_num <= 81 ){
//			cout.clear();
//			cout<<"Interesting tile: TotalFOLD = "<<TOTAL_FOLD<<endl;
//		}
//		else
//			cout.setstate(std::ios_base::failbit);


		//Transferring input to partitioned array
temp_inElem = inElem;
for(unsigned ndx=0; ndx<PE/16; ndx++)
	{
	for(unsigned  pe = 0; pe < 16; pe++)
		{
#pragma HLS UNROLL
//			unsigned int lowBit = pe * 8*SIMD;
//			unsigned int highBit = (pe+1)*8*SIMD - 1;
//			inElem_partitioned[pe] = inElem.range(highBit, lowBit);
			inElem_partitioned[pe+16*ndx] = inElem.range(trans_bit*SIMD-1, 0);
			inElem = inElem >> trans_bit*SIMD;
		}
	inElem = temp_inElem;
	}




//	    cout<<"Tile"<<endl;
//	    for(int a =0; a<4;a++){
//	    	for(int b=0;b<4;b++)
//	    		cout << inElem_partitioned[4*a+b]<<" ";
//	    	cout<<endl;
//	    }


	    // Threshold Initialisation
	    if(df == 0) {
	      for(unsigned  pe = 0; pe < PE; pe++) {
	#pragma HLS UNROLL
		    accu[pe] = 0;
	      }
	    }

		//fetch a tile of weight. 1 tile of weight = first SIMD weights of all filters
		auto const w = weights[nf][tile];
//	    auto const w = weights.get_col(nf, tile);
		auto const b = bias[nf];
//	    auto const &w = weights.weights(tile);
//		for(int f=0; f<PE/16; f++)
//	    {cout<<"Filter: "<< nf <<endl;
//			for(int a = 0; a<16;a++)
//			cout<< hex <<w[a+f*16]<<" "<<dec;
//		cout<<endl;
//	    }
//		if(nf == 0)
//			cout<<"Interesting filter"<<endl;

	    for(unsigned  pe = 0; pe < PE; pe++) {
	#pragma HLS UNROLL
		    	auto const  wgt = &w[pe];
			auto const  act = &inElem_partitioned[pe];
			accu[pe] = mac<SIMD, trans_bit>(accu[pe], wgt, act);
//			cout <<pe<<"- weight: "<<*wgt<<" act: "<<*act<<" res = "<<accu[pe]<<endl;
	    }

	    ++tile;
	    if(++df == DF)
	    {

	    	// we have 4x4 dot product ready. Now output transform
	    	//for 1 PE
	    	ap_uint<output_size> packed_out;

	    	for(int jdx = 0; jdx<PE/16; jdx++)
	    	{
#pragma HLS UNROLL
	    		int idx = 0;
	    		Accum_type temp2[16 / 2];//No need to initialize to 0
	    		Accum_type temp3[4];
	            for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
	                idx = 4 * j;
	                temp2[j] = accu[idx + jdx*16] + accu[idx + 1 + jdx*16] + accu[idx + 2 + jdx*16];
	                temp2[4 + j] = accu[idx + 1 + jdx*16] - accu[idx + 2 + jdx*16] - accu[idx + 3 + jdx*16];
	            }

	            temp3[0] = temp2[0] + temp2[1] + temp2[2];
	            temp3[1] = temp2[4] + temp2[5] + temp2[6];
	            temp3[2] = temp2[1] - temp2[2] - temp2[3];
	            temp3[3] = temp2[5] - temp2[6] - temp2[7];

	            //Maxpool Operation

	            if(output_size == 8*PE/16){ //This means that maxpool is applied
				Accum_type max_val = temp3[0];
//					if(has_bias)
//						max_val = -b[jdx];
//					else
//						max_val = 0;

				for(int idj=1; idj<4;idj++)
					{
						if(temp3[idj]>max_val)
							max_val = temp3[idj];
					}
				if(has_bias){
//					cout << "bias = " << b[jdx]<<endl;
					max_val = max_val + b[jdx];
				}


				max_val = max_val>>out_shift;


	            //Clipped Relu
	            if(max_val < 0)
	            	max_val = 0;
	            else if(max_val > 255)
	            	max_val = 255;

	            //Because max_val=0 in maxpool, just clamp here
//	            if(max_val > 255)
//					max_val = 255;


	            //Relu Activation
//	            for(int i =0; i<4; i++){
//#pragma HLS UNROLL
//	            	if(temp3[i]<0)
//	            		temp3[i] = 0;
//	            }





//				cout<<"output before pack = "<< hex <<(output[output_idx])<<dec<<endl;

				unsigned int lowBit = jdx*8;// + jdx*8*next_SIMD has value of next neuron
				unsigned int highBit = (jdx+1)*8 -1;
	            packed_out.range(highBit, lowBit) = max_val.range(7,0);

	            }
	            else{ //If maxpool is not applied, output_size = 8-bit * 4-elems * PE/16

	            	for(int idj=0; idj<4;idj++)
						{
							temp3[idj] = temp3[idj] + b[jdx];
							temp3[idj] = temp3[idj]>>out_shift;

							//clipped relu
							if(temp3[idj] < 0)
								temp3[idj] = 0;
							else if(temp3[idj] > 255)
								temp3[idj] = 255;
							//****************************************************************//
							//****************************************************************//
							//******** Isn't this same as bit_slicing differently??? **********//
							//****************************************************************//
							//************But this may save error in case of opposite shift*****************//
							//****************************************************************//
						}

	            	unsigned int lowBit = jdx*8*4;// + jdx*8*next_SIMD has value of next neuron
					unsigned int highBit = (jdx+1)*8*4 -1;
#if !__SYNTHESIS__
						assert(!(out_shift<0 || out_shift>20));
#endif
					//shift values by out_shift and relu
					packed_out.range(highBit, lowBit) = temp3[3].range(7,0)<<24
							| temp3[2].range(7,0)<<16
							| temp3[1].range(7,0)<<8
							| temp3[0].range(7,0);
	            }

	    	}
#if !__SYNTHESIS__
	    	conv_out << hex << packed_out <<","<<endl;
#endif
	    	out.write(packed_out);

	    	df = 0;
	    	tile = 0;
	    	if(++nf == NF) {
	    		    nf   = 0;
	    		    tile = 0;
	    	      }
#pragma AP dependence variable=nf intra false
	    }



	}
#if !__SYNTHESIS__
	conv_out.close();
	conv_in.close();
#endif
}



//- Request DSP48
template<typename TC, typename TD>
auto mul(TC const &c, TD const &d) -> decltype(c*d){

#pragma HLS inline
  decltype(c*d)  res = c*d;
  cout<<"mul:c = "<< c <<", d = "<< d <<"res = "<<res<<endl;

//#pragma HLS RESOURCE variable=res core=DMul_fullDSP
//#pragma HLS RESOURCE variable=res core=AddSub_DSP
//#pragma HLS RESOURCE variable=res core=Mul
  return  res;
}

//- MAC with selectable implementation resource
template<unsigned N, unsigned int trans_bit,typename T, typename TC, typename TD>
T mac(T const &a, TC const &c, TD const &d) {
#pragma HLS inline
  T  mac_res = a;
  ap_int<bitw> simd_c_lane[N];   //Typename should be TW for more generic
  ap_int<trans_bit> simd_d_lane[N];
#pragma HLS ARRAY_PARTITION variable=simd_c_lane complete dim=0
#pragma HLS ARRAY_PARTITION variable=simd_d_lane complete dim=0

  for(unsigned  i = 0; i < N; i++)
  {
#pragma HLS unroll
	unsigned int lowBit = i * bitw; //Hardcoded for 8 bit weights
	unsigned int highBit = (i+1)*bitw-1;
	unsigned int lowBit_d = i * trans_bit; //Hardcoded for 8 bit weights
	unsigned int highBit_d = (i+1)*trans_bit-1;


	simd_c_lane[i] = c->range(highBit,lowBit);
	simd_d_lane[i] = d->range(highBit_d,lowBit_d);
  }

  for(unsigned  i = 0; i < N; i++) {
#pragma HLS unroll
	  T mul_temp;
	  mul_temp = simd_c_lane[i] * simd_d_lane[i];
	  mac_res += mul_temp;
#pragma HLS RESOURCE variable=mul_temp core=Mul
#pragma HLS RESOURCE variable=mac_res core=AddSub

  }
  return  mac_res;
}



template<
unsigned int ConvKernelDim,
unsigned int Stride,
unsigned int Input_precision,
unsigned int IFMDim,
unsigned int OFMDim,
unsigned int vals_per_input, //This is used to separate individual stream values
unsigned int trans_bit, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned int IFMChannels,
unsigned int out_shift,unsigned int output_size,// current_Maxpool,
typename TSrcI , typename TDstI , typename TWeightI , bool has_bias,
typename Accum_type = ap_int<64>,
typename TW, typename TB//, typename R
>
void convlayer(hls::stream<ap_uint<vals_per_input*Input_precision> > & in,
		hls::stream<ap_uint<output_size>> &out,
		TW  const &weights,
		TB const &bias,
		int const reps )
{
#pragma HLS INLINE
	hls::stream<ap_uint<trans_bit*SIMD*4*4>> inStream_tran("out_of_intransform");
		//#pragma HLS STREAM variable=inStream_tran depth=450

	hls::stream<ap_uint<Input_precision*SIMD*4*4>> window_out("out_of_window");
		//#pragma HLS STREAM variable=window_out depth=450

	SlidingWindow<ConvKernelDim, IFMChannels, Input_precision, IFMDim, OFMDim, Stride,
	vals_per_input, SIMD>(in, window_out, reps);

	//	SlidingWindow<4, OFM_CH_C1, bitw, IFM_DIM_C2, OFM_DIM_C2, 2, OFM_CH_C1, simd_C2>(c2_in_padded, c2_window_out, reps);
	//	in_transform<simd_C2, bitw, IFM_DIM_C2, OFM_CH_C1, extra_bit_C2>(c2_window_out, c2_inStream_tran);
	//	wino_conv<extra_bit_C2, OFM_CH_C2, simd_C2, pe_num_C2, OFM_CH_C1,
	//		OFM_DIM_C2, 7,
	//		ap_int<8> , ap_int<8> , ap_int<8> ,
	//		ap_int<32>>(c2_inStream_tran, out, weights_C2, reps);

//	bool prevMaxpool = true;
//	bool current_Maxpool = true;
//	unsigned int output_size = 8*PE/16;
//
//	if (!current_Maxpool)
//		output_size=8*PE/16*4;


	in_transform<SIMD, Input_precision, IFMDim, IFMChannels, trans_bit>(window_out, inStream_tran, reps);
	wino_conv<trans_bit, MatrixH, SIMD, PE, IFMChannels,
	IFMDim, out_shift, output_size,
	TSrcI , TDstI , TWeightI ,
	Accum_type, has_bias>(inStream_tran, out, weights, bias, reps);

}




#endif
