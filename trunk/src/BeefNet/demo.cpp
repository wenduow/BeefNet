#include "NN/Net/nn_2_layer.hpp"
#include "NN/Weight/weight_bp.hpp"
#include "NN/Weight/param_bp.hpp"
#include "Xfer/xfer_lnr.hpp"

using namespace wwd;

int32 main(void)
{
    CNN2Layer< 2,
               2, FXferLnr,
               2, FXferLnr,
               1, FXferLnr,
               CWeightBP, EParamBP<> > nn;

    for ( uint32 i = 0; i < 2000; ++i )
    {
        for ( uint32 j = 1; j <= 100; ++j )
        {
            double input[2] = { 0.01 * j / 3.0, 0.02 * j / 3.0 };
            double target[1] = { 0.03 * j / 3.0 };

            nn.set_input(input);
            nn.set_target(target);

            nn.forward();
            nn.backward();
        }

        nn.update();
    }

#ifdef _DEBUG
    nn.print_weight();
#endif // _DEBUG

    return 0;
}

