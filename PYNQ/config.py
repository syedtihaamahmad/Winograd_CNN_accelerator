
IMG_CH=1
IMG_DIM=28
MEM_BANDWIDTH=64
Vals_per_Input=(MEM_BANDWIDTH/8)
INPUT_depth=((IMG_DIM*IMG_DIM*IMG_CH)/Vals_per_Input)


bitw=8
#Conv1parameters
simd_C1=1
out_parallel_C1=4
pe_num_C1=(16*out_parallel_C1)
IFM_DIM_C1=30
OFM_DIM_C1=14
OFM_CH_C1=16
extra_bit_C1=0
out_shift_C1=6
out_size_C1=(8*pe_num/16)
has_bias_C1=true

#Conv2_parameters
simd_C2=2
out_parallel_C2=2
pe_num_C2=(16*out_parallel_C2)
IFM_DIM_C2=16
OFM_DIM_C2=7
OFM_CH_C2=20
extra_bit_C2=1
out_shift_C2=7
out_size_C2=(8*pe_num_C2/16)
has_bias_C2=true

