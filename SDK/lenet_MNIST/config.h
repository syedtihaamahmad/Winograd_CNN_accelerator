#ifndef CONFIG_H
#define CONFIG_H

#include "core.h"

struct layer{
	unsigned int layer_no;
	//weight parameters
	unsigned int weight_ch;
	unsigned int weight_row;
	unsigned int weight_col;
	//bias parameters
	bool hasBias;
	unsigned int bias_row;
	unsigned int bias_col;
};

const unsigned int totalLayers = 2;

const layer C1 = {1,
		OFM_CH_C1/out_parallel_C1, IMG_CH/simd_C1, pe_num_C1,
		has_bias_C1, OFM_CH_C1/out_parallel_C1, out_parallel_C1};

const layer C2 = {2,
		OFM_CH_C2/out_parallel_C2, OFM_CH_C1/simd_C2, pe_num_C2,
		has_bias_C2, OFM_CH_C2/out_parallel_C2, out_parallel_C2};

layer layers[totalLayers] = {C1, C2};

unsigned long long W_C1[OFM_CH_C1/out_parallel_C1][IMG_CH/simd_C1][pe_num_C1]={
#include "./data/conv1_packed_out4.txt"
};

unsigned long long W_C2[OFM_CH_C2/out_parallel_C2][OFM_CH_C1/simd_C2][pe_num_C2]={
#include "./data/conv2_packed_out2.txt"
};

//Biases are signed because they are not packed
long long B_C1[OFM_CH_C1/out_parallel_C1][out_parallel_C1]={
#include "./data/qconv1_b.txt"
};

long long B_C2[OFM_CH_C2/out_parallel_C2][out_parallel_C2]={
#include "./data/qconv2_b.txt"
};


// function to set individual layer parameters
void Load_Val_Weight(XNn_top *xbbj, unsigned int targetmem, unsigned int
		target_ch, unsigned int target_row,unsigned int target_col, unsigned long long val)
{
	// Set loading mode
	XNn_top_Set_doInit(xbbj, 1);
	// position
	XNn_top_Set_targetmem(xbbj, targetmem);
	XNn_top_Set_target_ch(xbbj, target_ch);
	XNn_top_Set_target_row(xbbj, target_row);
	XNn_top_Set_target_col(xbbj, target_col);

	XNn_top_Set_val_V(xbbj, val);

//	XNn_top_WriteReg(xbbj->Ctrl_bus_BaseAddress,
//			XNN_TOP_CTRL_BUS_ADDR_VAL_V_DATA, (val));
//	XNn_top_WriteReg(xbbj->Ctrl_bus_BaseAddress,
//			XNN_TOP_CTRL_BUS_ADDR_VAL_V_DATA + 4, (val >> 32));
	// exec
	XNn_top_Start(xbbj);
	while(!XNn_top_IsDone(xbbj));
	XNn_top_Set_doInit(xbbj, 0);
}

//Difference b/w this and above function is type of val
void Load_Val_Bias(XNn_top *xbbj, unsigned int targetmem, unsigned int
		target_ch, unsigned int target_row,unsigned int target_col, long long val)
{
	// Set loading mode
	XNn_top_Set_doInit(xbbj, 1);
	// position
	XNn_top_Set_targetmem(xbbj, targetmem);
	XNn_top_Set_target_ch(xbbj, target_ch);
	XNn_top_Set_target_row(xbbj, target_row);
	XNn_top_Set_target_col(xbbj, target_col);

	XNn_top_Set_val_V(xbbj, val);

//	XNn_top_WriteReg(xbbj->Ctrl_bus_BaseAddress,
//			XNN_TOP_CTRL_BUS_ADDR_VAL_V_DATA, (val));
//	XNn_top_WriteReg(xbbj->Ctrl_bus_BaseAddress,
//			XNN_TOP_CTRL_BUS_ADDR_VAL_V_DATA + 4, (val >> 32));
	// exec
	XNn_top_Start(xbbj);
	while(!XNn_top_IsDone(xbbj));
	XNn_top_Set_doInit(xbbj, 0);
}


// function to load layer parameters
void Load_Weights(XNn_top *xbbj)
{
		layer l = layers[0];
		xil_printf("Initializing weights layer 1\n");
		for(unsigned int i=0; i<l.weight_ch; i++)
			for(unsigned int r=0; r<l.weight_row; r++)
				for(unsigned int c=0; c<l.weight_col; c++)
					Load_Val_Weight(xbbj, 1, i,r,c, W_C1[i][r][c]);

		l = layers[1];
		xil_printf("Initializing weights layer 2\n");
		for(unsigned int i=0; i<l.weight_ch; i++)
			for(unsigned int r=0; r<l.weight_row; r++)
				for(unsigned int c=0; c<l.weight_col; c++)
					Load_Val_Weight(xbbj, 3, i,r,c, W_C2[i][r][c]);


}


//layerNo starts from 1
void Load_Bias(XNn_top *xbbj)
{

		layer l = layers[0];
		xil_printf("Initializing bias layer 1\n");
				for(unsigned int r=0; r<l.bias_row; r++)
					for(unsigned int c=0; c<l.bias_col; c++)
						Load_Val_Bias(xbbj, 2, 0,r,c, B_C1[r][c]);

		l = layers[1];
		xil_printf("Initializing bias layer 2\n");
				for(unsigned int r=0; r<l.bias_row; r++)
					for(unsigned int c=0; c<l.bias_col; c++)
						Load_Val_Bias(xbbj, 4, 0,r,c, B_C2[r][c]);


}


void load_parameters(XNn_top *xbbj){

	xil_printf("***** Loading lenet with biases ******\n");
	Load_Weights(xbbj);
	Load_Bias(xbbj);
	xil_printf("Done Initializations ...\n");

}


#endif
