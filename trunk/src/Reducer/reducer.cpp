#include <iostream>
#include <fstream>
#include <string>
#include "../BeefNet/NN/Net/net_2_layer.hpp"
#include "../BeefNet/Xfer/xfer.hpp"
#include "../BeefNet/NN/Weight/weight.hpp"

using namespace wwd;

const uint32 input_num = 10;
const uint32 hidden_num = 10;
const uint32 output_num = 1;
template < class Param >
using Weight = CWeightBP<Param>;
typedef EParamBP<> Param;

int32 main(void)
{
    CNet2Layer< input_num,
                hidden_num, FXferLogSig,
                hidden_num, FXferLogSig,
                output_num, FXferLnr,
                Weight, Param > nn;

    std::string key;

    while ( std::cin >> key )
    {
        std::cin >> nn;
    }

    nn.update();

    std::ofstream saved_nn( "../../result/nn.dat" );
    nn.save(saved_nn);
    return 0;
}

