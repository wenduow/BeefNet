#ifndef TEST_TIME_HPP_
#define TEST_TIME_HPP_

#include "test_config.hpp"
#include "../package.hpp"

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

#endif // TEST_TIME_HPP_

