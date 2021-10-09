#ifndef STREAM_UTILS_H
#define STREAM_UTILS_H


//datawidth=64 to utilize max bandwidth
//numBytes = 7*8
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream(volatile ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out,
		const unsigned int numReps) {
  //CASSERT_DATAFLOW(DataWidth % 8 == 0); //assertion raised if false
  const unsigned int numWords = numBytes;
  //CASSERT_DATAFLOW(numWords != 0);
  for (unsigned int i = 0; i < numWords*numReps; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in[i];
    out.write(e);
  }
}

template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem(hls::stream<ap_uint<DataWidth> > & in,
		volatile  ap_uint<DataWidth> * out, const unsigned int numReps) {
  //CASSERT_DATAFLOW(DataWidth % 8 == 0);
  const unsigned int numWords = numBytes;
  //CASSERT_DATAFLOW(numWords != 0);
  for (unsigned int i = 0; i < numReps*numWords; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in.read();
	out[i] = e;
  }
}

template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords		// number of input words to process
>
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth> > & in,
		hls::stream<ap_uint<OutWidth> > & out, const unsigned int numReps) {
  if (InWidth > OutWidth) {
    // emit multiple output words per input word read
#if !__SYNTHESIS__
	  assert(InWidth % OutWidth == 0);
#endif
//    CASSERT_DATAFLOW(InWidth % OutWidth == 0);
    const unsigned int outPerIn = InWidth / OutWidth;
    const unsigned int totalIters = NumInWords * outPerIn * numReps;
    unsigned int o = 0;
    ap_uint<InWidth> ei = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS LOOP_TRIPCOUNT
#pragma HLS PIPELINE II=1
      // read new input word if current out count is zero
      if (o == 0) {
        ei = in.read();
	  }
      // pick output word from the rightmost position
      ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
      out.write(eo);
      // shift input to get new output word for next iteration
      ei = ei >> OutWidth;
      // increment written output count
      o++;
      // wraparound indices to recreate the nested loop structure
      if (o == outPerIn) {
        o = 0;
      }
    }
  } else if (InWidth == OutWidth) {
    // straight-through copy
    for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS LOOP_TRIPCOUNT
#pragma HLS PIPELINE II=1
      ap_uint<InWidth> e = in.read();
      out.write(e);
    }
  } else { // InWidth < OutWidth
    // read multiple input words per output word emitted
//    CASSERT_DATAFLOW(OutWidth % InWidth == 0);
#if !__SYNTHESIS__
	  assert(OutWidth % InWidth == 0);
#endif
    const unsigned int inPerOut = OutWidth / InWidth;
    const unsigned int totalIters = NumInWords * numReps;
    unsigned int i = 0;
    ap_uint<OutWidth> eo = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS LOOP_TRIPCOUNT
#pragma HLS PIPELINE II=1
      // read input and shift into output buffer
      ap_uint<InWidth> ei = in.read();
      eo = eo >> InWidth;
      eo(OutWidth - 1, OutWidth - InWidth) = ei;
      // increment read input count
      i++;
      // wraparound logic to recreate nested loop functionality
      if (i == inPerOut) {
        i = 0;
        out.write(eo);
      }
    }
  }
}



/**
 * \brief   Stream Padding - Padds the input with zeroes for when the sliding window is
 *          centered on border pixels
 *
 * Used to add padding to the input with zeroes in case the sliding window is
 * centered on border pixels
 *
 * \tparam     ImgDim          Size of the input feature map
 * \tparam     KernelDim       Size of the sliding window
 * \tparam     Stride          Stride of the sliding window
 * \tparam     NumChannels     Amount of channels of the input feature map
 * \tparam     In_t            Input datatype
 * \tparam     PaddingStyle    Type of padding that will be applied
 *
 * \param      in              Input stream
 * \param      out             Output stream
 *
 */
template<	unsigned int ImgDim,
			unsigned int KernelDim,
			unsigned int Stride,
			unsigned int NumChannels,
			unsigned int Prev_out_channels,
			unsigned int Out_t, //This is same as Prev_out_channels
			typename In_t,
      unsigned int PaddingStyle=2>
void Streaming_pad_basic(hls::stream<ap_uint<Prev_out_channels* In_t::width> > &in,
		hls::stream<ap_uint<Out_t* In_t::width> > &out, const unsigned int numReps){

	// Number of "same" windows over the input data
	constexpr unsigned int SameWindows = (ImgDim) / Stride + ((ImgDim % Stride) > 0);

	// Number of elements to generate as output per dimension
	constexpr unsigned int OutputDim = KernelDim + Stride * (SameWindows - 1);

	// Padding
	constexpr unsigned int Padding = OutputDim - ImgDim;
	constexpr unsigned int FOLD = NumChannels/Prev_out_channels;

//	cout<<"SameWindow = "<< SameWindows<<endl;
//	cout<<"OutputDim = "<< OutputDim<<endl;
//	cout<<"Padding = " << Padding<<endl;

	// Padding Up and Left
  constexpr unsigned int PaddingUp = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);
  constexpr unsigned int PaddingLeft = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);

	// Padding Down and Right (might be 1 element more than up and left in case of odd padding)
	constexpr unsigned int PaddingDown = Padding - PaddingUp;
	constexpr unsigned int PaddingRight = Padding - PaddingLeft;

	cout<<"Starting padding"<<endl;
	for(unsigned int rep=0; rep< numReps; rep++)
	for(unsigned int y = 0; y<OutputDim; y++){
		for(unsigned int x=0; x < OutputDim; x++){
				for(unsigned int z=0; z<FOLD; z++){
	#pragma HLS PIPELINE II=1
				ap_uint<Prev_out_channels* In_t::width> inData;
				ap_uint<Out_t* In_t::width> outData;

				// Padding Rows
				if(y < PaddingUp || y >= (OutputDim - PaddingDown)){
					outData = 0;
				}
				// Padding Cols
				else if(x < PaddingLeft || x >= (OutputDim - PaddingRight)){
					outData = 0;
				}
				// No Padding
				else{
					inData = in.read();
					outData = inData;
				}

				out.write(outData);
	//			cout<<outData<<" ";
				}
	//		cout<<endl;
		}
	}
}

/*
 * This function wraps the basic version.
 *
 * Because, the above version only accepts input packed
 * along its depth i.e. NumChannels*In_t::width
 *
 * ------------- Hard-coded for padding of 1 ---------------
 * */

template<	unsigned int ImgDim,
			unsigned int KernelDim,
			unsigned int Stride,
			unsigned int NumChannels,
			unsigned int vals_per_input,
			unsigned int vals_per_output,
			typename In_t,
      unsigned int PaddingStyle=2>
void Streaming_pad(hls::stream<ap_uint<vals_per_input* In_t::width> > &in,
		hls::stream<ap_uint<vals_per_output* In_t::width> > &out, const unsigned int numReps){
#pragma HLS INLINE
//	hls::stream<ap_uint<NumChannels* In_t::width>> depth_packed_stream("depth packed stream");
	hls::stream<ap_uint<vals_per_input* In_t::width>> depth_pp_stream("depth pack&pad stream");

//	StreamingDataWidthConverter_Batch<vals_per_input* In_t::width,NumChannels* In_t::width,
//		(ImgDim*ImgDim*NumChannels)/vals_per_input>(in, depth_packed_stream, 1);


#if !__SYNTHESIS__
	assert( vals_per_output%vals_per_input == 0);
#endif

	if(vals_per_input!= vals_per_output)
	{


		//This is IFM_dim before padding???
		Streaming_pad_basic<ImgDim,KernelDim,Stride,NumChannels,vals_per_input,vals_per_input,
							In_t, PaddingStyle>(in, depth_pp_stream, numReps);

		StreamingDataWidthConverter_Batch<vals_per_input* In_t::width,vals_per_output* In_t::width,
		((ImgDim+2)*(ImgDim+2)*(NumChannels/vals_per_input))>(depth_pp_stream, out, numReps);
	}
	else{
		//This is IFM_dim before padding???
		Streaming_pad_basic<ImgDim,KernelDim,Stride,NumChannels,vals_per_input,vals_per_output,
							In_t, PaddingStyle>(in, out, numReps);
	}



}


#endif
