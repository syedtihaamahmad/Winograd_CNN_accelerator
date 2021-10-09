#ifndef MY_CORE_H
#define MY_CORE_H

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



#endif
