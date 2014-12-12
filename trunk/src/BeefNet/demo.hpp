#ifndef DEMO_HPP_
#define DEMO_HPP_

#include "NN/Net/net.hpp"
#include "NN/Weight/weight.hpp"
#include "Xfer/xfer.hpp"
#include "Err/err.hpp"
#include "Reader/reader.hpp"
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

