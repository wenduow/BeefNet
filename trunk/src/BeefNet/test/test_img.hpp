#ifndef TEST_IMG_HPP_
#define TEST_IMG_HPP_

#include <ctime>
#include <iostream>
#include "../package.hpp"

const uint32 input_num  = 8;
const uint32 output_num = 1;
const uint32 hidden_num = 10;

typedef CWeightLM<> MyWeight;
typedef CNN2Layer< MyWeight,
                   input_num,                       // input layer
                   hidden_num,  FXferLogSig,        // 1st layer
                   hidden_num,  FXferLogSig,        // 2nd layer
                   output_num,  FXferLnr > MyNN;    // output layer
typedef FErrMAE MyErr;
typedef CReaderBinary<input_num>  MyInput;
typedef CReaderBinary<output_num> MyTarget;

#endif // TEST_IMG_HPP_

