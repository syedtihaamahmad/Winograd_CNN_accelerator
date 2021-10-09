#ifndef WEIGHT_HPP
#define WEIGHT_HPP


template<
	unsigned int BIT_WIDTH,
	unsigned int CH,
	unsigned int ROW,
	unsigned int COL
>
class Fixed_Weights{
ap_uint<BIT_WIDTH> weights_C1[CH][ROW][COL];

public:
	void set(unsigned int ch, unsigned int row, unsigned int col, ap_uint<64> val){
		weights_C1[ch][row][col] = val;
	}

	const ap_uint<BIT_WIDTH> * get_col(unsigned int ch, unsigned int row) const{
		return weights_C1[ch][row];

	}

};





#endif
