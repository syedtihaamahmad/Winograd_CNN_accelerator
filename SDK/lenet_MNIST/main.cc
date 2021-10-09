#include <stdio.h>

// driver files for IP
#include <xparameters.h>
#include "xnn_top.h"
#include "xnn_top_hw.h"
#include <xil_cache.h>

// driver for our axi timer
#include <xtmrctr.h>

// driver for Interrupt Controller
#include <xscugic.h>
#include <xil_exception.h>

// Extra utilities
#include <xil_types.h>
#include <xil_printf.h>

// Loading configurations
#include "config.h"

// Creating accelerator object
XNn_top nn_top_IP;
XNn_top_Config *nn_top_cfg;

// Creating Interrupt Controller
XScuGic INTC_Inst;
XScuGic_Config *INTC_cfg;

// Input and Output Arrays
#define SIZE_IMAGE 98
#define SIZE_OUT 490

u16 outData[SIZE_OUT] = {0};
unsigned long long image[SIZE_IMAGE] = {
#include "./data/input_sdk.txt"
};

u16 output_gold[SIZE_OUT]={
#include "./data/conv2_out.txt"
};

unsigned INT_count=0;

void Nn_top_InterruptHandler(){
//	xil_printf("\n **** Interrupt Raised ****\n");
	//Step-1: disable Interrupt
//	XNn_top_InterruptDisable(&nn_top_IP, 0x1);
	//Step-2: Acknowledge interrupt
	XNn_top_InterruptClear(&nn_top_IP, 0x1);
	INT_count++;
	//Step-3: Enable Interrupt
//	XNn_top_InterruptEnable(&nn_top_IP, 0x1);
//	xil_printf("\n Interrupt Enabled\n");

}




void initPeripherals(){
	xil_printf("initializing Nn_top\n");
	int status = 0;
	nn_top_cfg = XNn_top_LookupConfig(XPAR_NN_TOP_0_DEVICE_ID);
	if(nn_top_cfg)
	{
		status = XNn_top_CfgInitialize(&nn_top_IP, nn_top_cfg);
		if(status != XST_SUCCESS)
		{
			xil_printf("Error initializing  XNn_top IP\n");
		}
	}


	xil_printf("Initializing Global INT Controller\n");
	INTC_cfg = XScuGic_LookupConfig(XPAR_PS7_SCUGIC_0_DEVICE_ID);
	if(INTC_cfg)
	{
		int status = XScuGic_CfgInitialize(&INTC_Inst, INTC_cfg, INTC_cfg->CpuBaseAddress);
		if(status != XST_SUCCESS)
		{
			xil_printf("Error initializing  INT controller\n");
		}
	}


	xil_printf("Setting up Interrupt Controller\n");
	// Exception handling
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
								(Xil_ExceptionHandler)XScuGic_InterruptHandler,
								&INTC_Inst);
	Xil_ExceptionEnable();
	// Connect our Custom Nn_top IP to GIC and out handler in code
	status = XScuGic_Connect(	&INTC_Inst,
			XPAR_FABRIC_NN_TOP_0_INTERRUPT_INTR,
			(Xil_ExceptionHandler)Nn_top_InterruptHandler,
			(void *)&nn_top_IP);
	if(status != XST_SUCCESS)
			{
				xil_printf("Error initializing  INT controller\n");
			}



}

void enableInterrupt(){
	XNn_top_InterruptEnable(&nn_top_IP, 0x1);
		XNn_top_InterruptGlobalEnable(&nn_top_IP);
		xil_printf("Interrupts ready to go ...\n");
		XScuGic_Enable(&INTC_Inst, XPAR_FABRIC_NN_TOP_0_INTERRUPT_INTR);

}


int main()
{
	xil_printf("======= Starting work ========\n");
	initPeripherals();


	// timer
	XTmrCtr tmr;
	XTmrCtr_Initialize(&tmr, XPAR_AXI_TIMER_0_DEVICE_ID);


	u32 addr = 0x4c00000+0x10;

	// Write AXI master addresses to ports
	Xil_DCacheFlushRange(nn_top_IP.Ctrl_bus_BaseAddress+XNN_TOP_CTRL_BUS_ADDR_OUT_V_DATA, 8);

	Xil_DCacheFlushRange(addr,8);
	Xil_Out32(addr, 0x01234567);
	Xil_DCacheInvalidateRange(addr, 8);

	u64 inputData_address = (u64) image;
	XNn_top_Set_in_V(&nn_top_IP, (u64)image);
	Xil_DCacheInvalidateRange(nn_top_IP.Ctrl_bus_BaseAddress+XNN_TOP_CTRL_BUS_ADDR_IN_V_DATA, 8);
//	XNn_top_WriteReg(nn_top_IP.Ctrl_bus_BaseAddress, XNN_TOP_CTRL_BUS_ADDR_IN_V_DATA, (u32)(inputData_address));
//	XNn_top_WriteReg(nn_top_IP.Ctrl_bus_BaseAddress, XNN_TOP_CTRL_BUS_ADDR_IN_V_DATA + 4, 0);


	Xil_DCacheFlushRange(nn_top_IP.Ctrl_bus_BaseAddress+XNN_TOP_CTRL_BUS_ADDR_OUT_V_DATA, 8);
	u64 outputData_address = (u64) outData;
	XNn_top_Set_out_V(&nn_top_IP, (u64)outData);
	Xil_DCacheInvalidateRange(nn_top_IP.Ctrl_bus_BaseAddress+XNN_TOP_CTRL_BUS_ADDR_OUT_V_DATA, 8);
//	XNn_top_WriteReg(nn_top_IP.Ctrl_bus_BaseAddress, XNN_TOP_CTRL_BUS_ADDR_OUT_V_DATA, (u32)(outputData_address));
//	XNn_top_WriteReg(nn_top_IP.Ctrl_bus_BaseAddress, XNN_TOP_CTRL_BUS_ADDR_OUT_V_DATA + 4, 0);

	load_parameters(&nn_top_IP);
	enableInterrupt();

	Xil_DCacheFlushRange(inputData_address, SIZE_IMAGE*sizeof(unsigned long long));
	Xil_DCacheFlushRange(outputData_address, SIZE_OUT*sizeof(unsigned int));
	xil_printf("Done flushing memory");

	// enabling compute mode
	XNn_top_Set_doInit(&nn_top_IP, 0);
	// no. of test examples
	XNn_top_Set_numReps(&nn_top_IP, 1);

	// passing address of input and output buffer
	xil_printf("Sending input data\n");
	XNn_top_Set_in_V(&nn_top_IP, (u64)image);
	XNn_top_Set_out_V(&nn_top_IP, (u64)outData);

	xil_printf("Starting IP\n");

	// Start Timer
	XTmrCtr_Reset(&tmr,0);
	int tick1 = XTmrCtr_GetValue(&tmr,0), tick2;
	XTmrCtr_Start(&tmr,0);

	XNn_top_Start(&nn_top_IP);
	while(!XNn_top_IsDone(&nn_top_IP));

	// stop the timer
	XTmrCtr_Stop(&tmr,0);
	tick2 = XTmrCtr_GetValue(&tmr,0);

	// Calculate Time
	double time = (double)(tick2-tick1)/(double)XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ;
	printf("Time: %.9f ms\r\nFps: %.2f\r\n",time,1/time);


	Xil_DCacheInvalidateRange(outputData_address, SIZE_OUT*sizeof(int16_t));


	int diff=0;
	for(int i = 0; i<SIZE_OUT; i++){
		xil_printf("Hardware[%d] = %d,\t gold[%d] = %d\n", i, outData[i], i, output_gold[i]);
		if(outData[i] != output_gold[i])
			diff++;
//		xil_printf("Hardware[%d] = %d,\t gold[%d] = %d\n", 2*i+1, (u8)(outData[i]>>8), 2*i+1, (u8)(output_gold[i]>>8));
	}

	xil_printf("Total Differences: %d\n", diff);
	xil_printf("Interrupt count = %d\n", INT_count);

	return 0;
}



