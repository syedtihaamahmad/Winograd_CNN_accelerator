#ifndef MY_CORE_H
#define MY_CORE_H

#define IMG_CH 1
#define IMG_DIM 28
#define MEM_BANDWIDTH 64
#define Vals_per_Input (MEM_BANDWIDTH/8) //These brackets are important
#define INPUT_depth ((IMG_DIM*IMG_DIM*IMG_CH)/Vals_per_Input)


#define bitw 8
//Conv1 parameters
#define simd_C1 1
#define out_parallel_C1 4 //This is adjustable ***Must be 1,2,4,8 i.e. divisible by filter number
#define pe_num_C1 (16*out_parallel_C1)
#define IFM_DIM_C1 30
#define OFM_DIM_C1 14
#define OFM_CH_C1 16
#define extra_bit_C1 0
#define out_shift_C1 6
#define out_size_C1 (8*pe_num/16)
#define has_bias_C1 true

//Conv2 parameters
#define simd_C2 2 //This needs to be 1 for
#define out_parallel_C2 2 //This is also adjustable but weights will have to be generated again. Divisible by filter number
#define pe_num_C2 (16*out_parallel_C2)
#define IFM_DIM_C2 16
#define OFM_DIM_C2 7
#define OFM_CH_C2 20
#define extra_bit_C2 1
#define out_shift_C2 7
#define out_size_C2 (8*pe_num_C2/16)
#define has_bias_C2 true


#endif
