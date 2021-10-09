/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *******************************************************************************/

 /******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  \file slidingwindow.h
 *
 *  Library of templated HLS functions for BNN deployment.
 *  This file lists a set of convenience funtions used to implement
 *  Sliding window generator for convolutions
 *
 *****************************************************************************/

#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H

#define MAX(x, y) (((x) > (y)) ? (x) : (y)) /* \brief Maximum value between x and y*/
#define MIN(x, y) (((x) > (y)) ? (y) : (x)) /* !< \brief Minimum value between x and y*/

/**
 * \brief Sliding Window unit that produces output vectors for feeding
 * a Matrix_Vector_Activate_Batch, implementing the im2col algorithm. To be used only if
 * ConvKernelDim%Stride = 0
 *
 * \tparam ConvKernelDim    Dimension of the convolutional kernel (assumed square)
 * \tparam IFMChannels      Number of Input Feature Maps
 * \tparam Input_precision  Number bits per pixel
 * \tparam IFMDim           Width and Heigth of the Input Feature Map (assumed square)
 * \tparam OFMDim           Width and Heigth of the Output Feature Map (assumed square)
 * \tparam SIMD             Number of input columns computed in parallel
 * \tparam Stride           Stride of the convolutional kernel
 *
 * \param in                Input hls::stream
 * \param out               Output hls::stream
 * \param numReps           Number of time the function has to be repeatedly executed (e.g. number of images)
 */


template<unsigned int ConvKernelDim,
		 unsigned int IFMChannels,
		 unsigned int Input_precision,
		 unsigned int IFMDim,
		 unsigned int OFMDim,
		 unsigned int Stride,
		 unsigned int SIMD,
		 unsigned int nextSIMD>
void ConvolutionInputGenerator(
		hls::stream<ap_uint<SIMD*Input_precision> > & in,
		hls::stream<ap_uint<SIMD*Input_precision*nextSIMD> > & out,
		const unsigned int numReps = 1) {
//  //CASSERT_DATAFLOW(IFMChannels % SIMD == 0);
 // //CASSERT_DATAFLOW(ConvKernelDim % Stride == 0);

  const unsigned int multiplying_factor = IFMChannels/SIMD;
//  const unsigned int multiplying_factor = 1;
  const unsigned int folds = 1 ;

  const unsigned int number_blocks = ConvKernelDim/Stride+1   ;
  ap_uint<SIMD*Input_precision> inputBuf[number_blocks][Stride* IFMDim *folds*multiplying_factor];

#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
#pragma HLS RESOURCE variable inputBuf core=RAM_2P
  const unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim *(folds)* multiplying_factor);
  const unsigned int cycles_read_block =Stride * IFMDim *folds* multiplying_factor;
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
  const unsigned int baseIter = IFMDim * ConvKernelDim *folds*multiplying_factor// Initial buffer
			                  + OFMDim * MAX(cycles_write_block,cycles_read_block);

//  cout<<"cycles_write_block :"<<cycles_write_block<<","<<"cycles_read_block :"<<cycles_read_block<<","
//		  <<"max_cycles :"<<max_cycles<<","<<"baseIter :"<<baseIter<<","<<endl;

  unsigned int counter_internal_block = 0;
  unsigned int current_block_write = 0;
  unsigned int next_block_write = 0;
  unsigned int current_line = 0;
  unsigned int read_block = 0;
  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, count_simd =0;
int yo=0,b=16;
ap_uint<nextSIMD*Input_precision> out_16;

#pragma HLS reset variable=inp
  for (unsigned int count_image = 0; count_image < numReps; count_image++) {
#pragma HLS LOOP_TRIPCOUNT
	  for (unsigned int i = 0; i < baseIter; i++) {
#pragma HLS LOOP_TRIPCOUNT
#pragma HLS PIPELINE II=1
      if (inp < IFMDim * ConvKernelDim*multiplying_factor ) {// Initial buffer of ConvKernelDim lines
        ap_uint<SIMD*Input_precision> inElem;
        inElem = in.read();
       // cout<<"current_block_write : "<<current_block_write<<",\n";
        inputBuf[current_block_write][current_line] = inElem;
       // for(int i=0;i<number_blocks;i++)
        //	for(int j=0;j<Stride * IFMDim *multiplying_factor;j++)
        		//cout<<hex<<inputBuf[i][j]<<",";
        //cout<<endl;
        //cout<<hex<<inElem<<",,";
        current_line++;
        inp++;
        if (current_line == Stride * IFMDim * multiplying_factor) {
          current_line = 0;
          current_block_write++;
          if (current_block_write == number_blocks) {
            current_block_write=0;
          }
          read_block++;
          counter_internal_block = 0;
        }
      } else {
        if (counter_internal_block < cycles_write_block -1) { // We are writing output, MMV IFMChan per cycle
          unsigned int current_block_read = (current_block_write  +1+ k_y /Stride);
          if (current_block_read >= number_blocks) {
            current_block_read-= number_blocks;
		  }
         // cout<<"current_block_read "<<current_block_read<<",	";

          unsigned int current_line_in_block = ((k_y%Stride) * IFMDim + ofm_x*Stride + k_x)*multiplying_factor + count_simd;
          //cout<<"current_line_in_block"<<current_line_in_block<<",";

          ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read][(current_line_in_block)];

//           cout<<hex<<outElem<<",";

          if(yo<nextSIMD+1){
        	  unsigned int lowBit = (yo)*SIMD*Input_precision;
        	  unsigned int highBit = (yo+1)*SIMD*Input_precision-1;
        	  out_16.range(highBit,lowBit)=outElem;
        	  yo++;

        	  if(yo==nextSIMD){
            	  yo=0;
//            	  cout<<"TILE	";
//                  cout<<hex<<out_16<<",";
                  out.write(out_16);
        	  }
          }



          count_simd++;
          if (count_simd == multiplying_factor) {
            count_simd=0;
            k_x++;
            if (k_x == ConvKernelDim) {
//            	cout<<endl;

              k_x = 0;
              k_y++;
              if (k_y == ConvKernelDim) {
//            		cout<<endl;

                k_y = 0;
                ofm_x ++;
                if (ofm_x == OFMDim) {
                  ofm_x = 0;
                  ofm_y++;
                  if (ofm_y == OFMDim) {
                    ofm_y = 0;
                    inp = 0;
                  }
                }
              }
            }
          }
        }
       if ((counter_internal_block < cycles_read_block-1) && (read_block<IFMDim/Stride))  { // In parallel we write in the buffer, in the current block write if we still need to
       //   cout<<" pcurrent_block_write :"<<current_block_write<<",";
          ap_uint<SIMD*Input_precision> inElem;
          inElem = in.read();
          //cout<<hex<<inElem<<",,,";

          inputBuf[current_block_write][current_line] = inElem;
#pragma AP dependence variable=inputBuf intra false
#pragma AP dependence variable=inputBuf inter false
          current_line++;
          if (current_line == Stride * IFMDim *multiplying_factor) {// We read the whole block, we change the next block in which we want to we
            // We filled up a block, let's not read until
            current_line = 0;

            read_block++;
            current_block_write++;
            if (current_block_write == number_blocks) {
              current_block_write=0;
			}
#pragma AP dependence variable=current_block_write intra false
          }
        }
        counter_internal_block++; // = (counter_internal_block +1) % max_cycles;
        if (counter_internal_block == (max_cycles -1)) {
          counter_internal_block = 0;
        }
      }
      //cout<<"read_block : "<<read_block<<",";

    } // End base_iter
	read_block = 0;
  } // End count_image
} // End generator




template<
	unsigned IFM_dim,
	unsigned IFM_channels,
	unsigned SIMDi,
	unsigned int Input_pre
>
void window_pack(
		hls::stream<ap_uint<SIMDi*Input_pre> > & in,
				hls::stream<ap_uint<Input_pre*SIMDi*4*4> > & out, const unsigned int numReps){

	ap_uint<SIMDi*Input_pre> Input_buf[IFM_channels/SIMDi][4][4];
#pragma HLS ARRAY_PARTITION variable=Input_buf complete dim=3
#pragma HLS ARRAY_PARTITION variable=Input_buf complete dim=2
#pragma HLS RESOURCE variable=Input_buf core=RAM_2P_BRAM
//#pragma HLS dependence variable=Input_buf intra RAW true

	const unsigned int TOTAL_ITER = (IFM_dim/2-1) * (IFM_dim/2-1) * IFM_channels/SIMDi * 4*4;
	unsigned tile = 0;
	unsigned r=0, c=0;
	for(unsigned int i=0; i<numReps*TOTAL_ITER; i++){
#pragma HLS PIPELINE II=1
#pragma HLS dependence variable=Input_buf array intra RAW true

		ap_uint<Input_pre*SIMDi*4*4> outElem;

		Input_buf[tile][r][c] = in.read();


		// ************* LOOK CAREFULLY ***************//
		if(r == 3 && c == 3)
		{
			for(int row=0; row<4; row++)
				for(int col = 0; col<4; col++)
				{
					unsigned int lowBit = (col+row*4)*SIMDi*Input_pre;
					unsigned int highBit = (col+row*4+1)*SIMDi*Input_pre - 1;
					outElem.range(highBit, lowBit) = Input_buf[tile][row][col];
				}

			out.write(outElem);
		}


		tile++;

		if(tile == (IFM_channels/SIMDi)){
			c++;
			tile = 0;

			if(c == 4){
				r++;
				c=0;
				tile=0;
				if(r == 4){
					r=0;
					c=0;
					tile=0;
				}
			}
		}

	}

}



/*
 * As the above function can only accept separate values, this will serve as a wrapper.
 * */


template<unsigned int ConvKernelDim,
		 unsigned int IFMChannels,
		 unsigned int Input_precision,
		 unsigned int IFMDim,
		 unsigned int OFMDim,
		 unsigned int Stride,
		 unsigned int SIMD, //This is used to separate individual stream values
		 unsigned int nextSIMD>
void SlidingWindow(
		hls::stream<ap_uint<SIMD*Input_precision> > & in,
		hls::stream<ap_uint<Input_precision*nextSIMD*4*4> > & out,
		const unsigned int numReps = 1){
#pragma HLS INLINE
	hls::stream<ap_uint<Input_precision>> unit_stream("unit window in sliding window"); //This stream has width of 1 value
#pragma HLS STREAM variable=unit_stream depth=900
	hls::stream<ap_uint<Input_precision*nextSIMD>> window_out("basic window out");
#pragma HLS STREAM variable=window_out depth=900
	const unsigned int sim = 1;
	StreamingDataWidthConverter_Batch<SIMD*Input_precision,Input_precision,(IFMDim*IFMDim*IFMChannels)/SIMD>(in, unit_stream, numReps);

	ConvolutionInputGenerator<ConvKernelDim, IFMChannels, Input_precision, IFMDim,
	(IFMDim/2-1), Stride, sim, nextSIMD>(unit_stream, window_out,numReps);

#if !__SYNTHESIS__
	assert( IFMChannels%nextSIMD == 0);
#endif

	//This will not work as the values are coming in depth order
	//Packing all values in 1 value
//	StreamingDataWidthConverter_Batch<nextSIMD*Input_precision,nextSIMD*Input_precision*4*4,(OFMDim*OFMDim)*(IFMChannels/nextSIMD)*4*4>(window_out, out, numReps);
	window_pack<IFMDim, IFMChannels, nextSIMD, Input_precision>(window_out, out, numReps);
}


/*

template<unsigned int ConvKernelDim,
		 unsigned int IFMChannels,
		 unsigned int Input_precision,
		 unsigned int IFMDim,
		 unsigned int OFMDim,
		 unsigned int Stride,
		 unsigned int SIMD, //This is used to separate individual stream values
		 unsigned int nextSIMD>
void SlidingWindow(
		hls::stream<ap_uint<SIMD*Input_precision> > & in,
		hls::stream<ap_uint<Input_precision*nextSIMD*4*4> > & out,
		const unsigned int numReps = 1){

	ap_uint<SIMD*Input_precision> in_buf[4][IFMDim] = {0};
#pragma HLS ARRAY_PARTITION variable=in_buf complete dim=0

//**** It has been tested that either of these does not increase speed *****
//#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=6 dim=2
//#pragma HLS RESOURCE variable in_buf core=RAM_2P


	int in_count=0;
	int buf_row = 0;
	int buf_col = 0;

	//tile ready variables
	int tile_count=0;
	int TILE_ver=((IFMDim-2)/2);
	int TILE_hor=((IFMDim-2)/2);


	//Bigger overall loop
	for(int total = 0; total < IFMDim*IFMDim; total++){
#pragma HLS PIPELINE II=1

		ap_uint<Input_precision*nextSIMD*4*4> outElem;


		buf_row = (total/IFMDim) % 4;
		buf_col = total % IFMDim;

		in_buf[buf_row][buf_col]=in.read();

//		if(in_buf[buf_row][buf_col] == 42 || in_buf[buf_row][buf_col] == 33)
//			cout<<"Start"<<endl;

		if(total >= IFMDim*3+3 && (buf_row+1)%Stride==0 && (buf_col+1)%Stride==0 && (buf_col+1)>=3){
//			 cout<<"Tile: "<<dec<<tile_count+1<<" ";

//			 if(tile_count+1 == 62)
//				 cout<<"interest"<<endl;

			 int r=tile_count / TILE_ver;
			 int c=tile_count % TILE_hor;

			 for(int row=0; row<4; row++)
				for(int col = 0; col<4; col++)
				{
					unsigned int lowBit = (col+row*4)*nextSIMD*Input_precision;
					unsigned int highBit = (col+row*4+1)*nextSIMD*Input_precision - 1;
					outElem.range(highBit, lowBit) = in_buf[(r*Stride+row)%4][(c*Stride+col)%IFMDim];
				}

//			 cout<<hex<<outElem<<endl;
		out.write(outElem);

		tile_count++;
		}
	}


}

*/


#endif

