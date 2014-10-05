#ifndef TEST_HIDDEN_NODE_HPP_
#define TEST_HIDDEN_NODE_HPP_

#include "test_config.hpp"
#include "../package.hpp"

const uint32 input_num  = 8;
const uint32 output_num = 1;

typedef FErrMAE MyErr;
typedef CReaderBinary<input_num>  MyInput;
typedef CReaderBinary<output_num> MyTarget;

#endif // TEST_HIDDEN_NODE_HPP_

