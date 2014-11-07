#ifndef TEST_BP_HPP_
#define TEST_BP_HPP_

#include "test_config.hpp"
#include "../package.hpp"

const uint32 input_num  = 1;
const uint32 output_num = 1;
const uint32 learn_rate = 500;

#define CREATE_NN(hidden_num) \
    typedef CNN2Layer< CWeightBP<learn_rate>, \
                       input_num, \
                       hidden_num, FXferLogSig, \
                       hidden_num, FXferLogSig, \
                       output_num, FXferLnr > MyNN##hidden_num;

typedef FErrMAE MyErr;

CREATE_NN(1);
CREATE_NN(2);
CREATE_NN(3);
CREATE_NN(4);
CREATE_NN(5);
CREATE_NN(6);
CREATE_NN(7);
CREATE_NN(8);
CREATE_NN(9);
CREATE_NN(10);
CREATE_NN(11);
CREATE_NN(12);
CREATE_NN(13);
CREATE_NN(14);
CREATE_NN(15);
CREATE_NN(16);
CREATE_NN(17);
CREATE_NN(18);
CREATE_NN(19);
CREATE_NN(20);
CREATE_NN(21);
CREATE_NN(22);
CREATE_NN(23);
CREATE_NN(24);
CREATE_NN(25);
CREATE_NN(26);
CREATE_NN(27);
CREATE_NN(28);
CREATE_NN(29);
CREATE_NN(30);

#endif // TEST_BP_HPP_

