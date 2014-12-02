#ifndef DEMO_HPP_
#define DEMO_HPP_

#include <fstream>
#include "NN/Net/nn_2_layer.hpp"
#include "NN/Weight/weight_lm.hpp"
#include "NN/Weight/param_lm.hpp"
#include "Xfer/xfer_log_sig.hpp"
#include "Xfer/xfer_lnr.hpp"
#include "Err/err_mae.hpp"
#include "Reader/reader_binary.hpp"
#include "trainer.hpp"
#include "tester.hpp"

using namespace wwd;

const uint32 pattern_num = 26304;
const uint32 input_num = 10;
const uint32 hidden_num = 10;
const uint32 output_num = 1;

const uint32 thread_num = 8;

// template < class Param >
// using MyWeight = CWeightRP<Param>;
// 
// typedef EParamRP<> MyParam;

std::ofstream result( "../../result/test_algorithm.txt", std::ios::app );

#endif // DEMO_HPP_

