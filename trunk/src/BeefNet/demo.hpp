#ifndef DEMO_HPP_
#define DEMO_HPP_

#include "package.hpp"

std::ofstream result( "../../result/layer3_bp_fin.txt" );

// start of NN configuration

/** Input, Output and Hidden nodes */
const uint32 input_num  = 8;
const uint32 output_num = 1;
const uint32 hidden_num = 10;

/** Thread (Mapper) Number
 *  Maximum Epoch
 *  Validation Times
 *  Gradient Precision in 10 ^ x
 */
const uint32 image_num       = 4;
const uint32 max_epoch     = 500;
const uint32 valid_times   = 6;
const int32 gradient_prec  = -5;

/** Weight Type *
 *  Back-Propagation,   Quick-Propagation,  Resilient-Propagation
 *  CWeightBP,          CWeightQP,          CWeightRP
 */
typedef CWeightBP<> MyWeight;

/** NN Type
 *
 *  1-Layer,    2-Layer,    3-Layer,    Recurrent
 *  CNN1Layer,  CNN2Layer,  CNN3Layer,  CNNRecurrent
 *
 *  Transfer Function
 *  Log-Sigmoid,    Linear
 *  FXferLogSig,    FXferLnr
 */
typedef CNN3Layer< MyWeight,
                   input_num,                       // input layer
                   hidden_num,  FXferLogSig,        // 1st layer
                   hidden_num,  FXferLogSig,        // 2nd layer
                   hidden_num,  FXferLogSig,        // 3rd layer
                   output_num,  FXferLnr > MyNN;    // output layer

/** Error Function
 *
 *  MAE,        MSE,        RMSE
 *  FErrMAE,    FErrMSE,    FErrRMSE
 */
typedef FErrMAE MyErr;

// end of NN configuration

typedef CReaderBinary<input_num>  MyInput;
typedef CReaderBinary<output_num> MyTarget;

typedef CTrainer< MyNN,
                  MyInput,
                  MyTarget,
                  MyErr,
                  image_num,
                  max_epoch,
                  valid_times,
                  gradient_prec >                MyTrainer;
typedef CTester< MyNN, MyInput, MyTarget, MyErr> MyTester;
typedef CPredictor< MyNN, MyInput >              MyPredictor;

#endif // DEMO_HPP_

