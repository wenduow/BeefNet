#include "nn/weight/update_lm.hpp"
#include "nn/weight/param_lm.hpp"
#include "nn/net/net_2_layer.hpp"
#include "xfer/xfer.hpp"

using namespace wwd;

int32 main(void)
{
    CNet2Layer< 2,
                2, FXferLnr,
                2, FXferLnr,
                1, FXferLnr,
                CUpdateLM, CParamLM<> > nn;

    for ( uint32 i = 0; i < 2000; ++i )
    {
        for ( uint32 j = 0; j < 300; ++j )
        {
            double input[2] = { 0.001 * j, 0.002 * j };
            nn.set_input(input);
            nn.forward();

            double target[1] = { 0.003 * j };
            nn.set_target(target);
            nn.backward();
        }

        nn.update();
    }

    nn.print_weight();

    return 0;
}

