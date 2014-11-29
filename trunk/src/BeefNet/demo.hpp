#ifndef DEMO_HPP_
#define DEMO_HPP_

#include <fstream>
#include "NN/Net/nn_2_layer.hpp"
#include "NN/Weight/weight_bp.hpp"
#include "NN/Weight/param_bp.hpp"
#include "Xfer/xfer_log_sig.hpp"
#include "Xfer/xfer_lnr.hpp"
#include "Err/err_mae.hpp"
#include "Reader/reader_binary.hpp"
#include "trainer.hpp"

using namespace wwd;

template < class Param >
using MyWeight = CWeightBP<Param>;

typedef EParamBP<> MyParam;

const uint32 pattern_num = 43824;
const uint32 input_num = 10;
const uint32 hidden_num = 10;
const uint32 output_num = 1;

const uint32 thread_num = 1;

std::ofstream result( "../../result/test_bp_thread.txt", std::ios::app );

#endif // DEMO_HPP_

