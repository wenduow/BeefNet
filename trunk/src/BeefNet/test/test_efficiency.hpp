#ifndef TEST_EFFICIENCY_HPP_
#define TEST_EFFICIENCY_HPP_

#include <fstream>
#include "../package.hpp"

std::ofstream result( "../../result/result_efficiency.txt", std::ios::app );

const uint32 input_num  = 8;
const uint32 output_num = 1;
const uint32 hidden_num = 10;

typedef CNN2Layer< CWeightBP<>,
                   input_num,
                   hidden_num, FXferLogSig,
                   hidden_num, FXferLogSig,
                   output_num, FXferLnr > MyNNBP;

typedef CNN2Layer< CWeightQP<>,
                   input_num,
                   hidden_num, FXferLogSig,
                   hidden_num, FXferLogSig,
                   output_num, FXferLnr > MyNNQP;

typedef CNN2Layer< CWeightRP<>,
                   input_num,
                   hidden_num, FXferLogSig,
                   hidden_num, FXferLogSig,
                   output_num, FXferLnr > MyNNRP;

typedef CNN2Layer< CWeightLM<>,
                   input_num,
                   hidden_num, FXferLogSig,
                   hidden_num, FXferLogSig,
                   output_num, FXferLnr > MyNNLM;

typedef FErrMAE MyErr;
typedef CReaderBinary<input_num>  MyInput;
typedef CReaderBinary<output_num> MyTarget;

#endif // TEST_EFFICIENCY_HPP_

