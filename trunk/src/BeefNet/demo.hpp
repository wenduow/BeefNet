#ifndef DEMO_HPP_
#define DEMO_HPP_

#include "NN/Net/nn_2_layer.hpp"
#include "NN/Weight/weight_bp.hpp"
#include "NN/Weight/param_bp.hpp"
#include "Xfer/xfer.hpp"
#include "Err/err.hpp"
#include "Reader/reader_binary.hpp"
#include "trainer.hpp"
#include "tester.hpp"

namespace wwd
{

const uint32 pattern_num = 26304;
const uint32 input_num = 10;
const uint32 hidden_num = 10;
const uint32 output_num = 1;

const uint32 thread_num = 8;
const bool stop_early = true;

template < class Param >
using MyWeight = CWeightBP<Param>;

typedef EParamBP<> MyParam;

} // namespace wwd

#endif // DEMO_HPP_

