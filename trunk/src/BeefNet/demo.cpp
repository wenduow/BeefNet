#include "NN/Net/nn_2_layer.hpp"
#include "NN/Weight/weight_bp.hpp"
#include "NN/Weight/param_bp.hpp"
#include "Xfer/xfer_lnr.hpp"

using namespace wwd;

int32 main(void)
{
    typedef CNN2Layer< 2,
                       2, FXferLnr,
                       2, FXferLnr,
                       2, FXferLnr,
                       CWeightBP, EParamBP<> > NN;

    NN nn, nn_image_1, nn_image_2;

    for ( uint32 i = 0; i < 2000; ++i )
    {
        nn >> nn_image_1 >> nn_image_2;

        for ( uint32 j = 1; j <= 50; ++j )
        {
            double input[2] = { 0.01 * j / 4.0, 0.02 * j / 4.0 };
            double target[2] = { 0.03 * j / 4.0, 0.04 * j / 4.0 };

            nn_image_1.set_input(input);
            nn_image_1.set_target(target);

            nn_image_1.forward();
            nn_image_1.backward();
        }

        for ( uint32 j = 51; j <= 100; ++j )
        {
            double input[2] = { 0.01 * j / 4.0, 0.02 * j / 4.0 };
            double target[2] = { 0.03 * j / 4.0, 0.04 * j / 4.0 };

            nn_image_2.set_input(input);
            nn_image_2.set_target(target);

            nn_image_2.forward();
            nn_image_2.backward();
        }

        nn << nn_image_1 << nn_image_2;

        nn.update();
    }

#ifdef _DEBUG
    nn.print_weight();
#endif // _DEBUG

    return 0;
}

