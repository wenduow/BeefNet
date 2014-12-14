#include <iostream>
#include <sstream>
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

    std::ifstream saved_nn( "../../result/nn.dat" );
    nn.load(saved_nn);

    double input[input_num];
    for ( auto &i : input )
    {
        char delimeter;
        std::cin >> i >> delimeter;
    }

    nn.set_input(input);
    nn.forward();

    double target[output_num];
    for ( auto &i : target )
    {
        char delimeter;
        std::cin >> i >> delimeter;
    }

    nn.set_target(target);
    nn.backward();

    std::cout << "nn_image" << '\t' << nn << std::endl;
    return 0;
}

