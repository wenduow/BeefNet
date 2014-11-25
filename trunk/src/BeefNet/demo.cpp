#include "NN/Net/nn_2_layer.hpp"
#include "NN/Weight/weight_lm.hpp"
#include "NN/Weight/param_lm.hpp"
#include "Xfer/xfer_log_sig.hpp"
#include "Xfer/xfer_lnr.hpp"
#include "Err/err_mae.hpp"
#include "Reader/reader_binary.hpp"
#include "trainer.hpp"

using namespace wwd;

int32 main(void)
{
    srand( (uint32)time(NULL) );

    CNN2Layer< 1,
               10, FXferLogSig,
               10, FXferLogSig,
               1, FXferLnr,
               CWeightLM, EParamLM< 94, 1 > > nn;
    
    double err[1];
    CTrainer< FErrMAE, 1 > trainer;
    trainer.train<CReaderBinary>( err,
                                  nn,
                                  "../../data/train_input.dat",
                                  "../../data/train_target.dat" );

    return 0;
}

