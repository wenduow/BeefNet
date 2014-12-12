#include <fstream>
#include "demo.hpp"

using namespace wwd;

void data_transform(void);

int32 main(void)
{
    // data_transform();
    // return 0;

    srand( (uint32)time(NULL) );

    CNet2Layer< input_num,
                hidden_num, FXferLogSig,
                hidden_num, FXferLogSig,
                output_num, FXferLnr,
                MyWeight, MyParam > nn;
    
    double err[1];
    CTrainer< FErrMAE, thread_num, stop_early > trainer;

    time_t time_beg = time(NULL);
    trainer.train<CReaderBinary>( err,
                                  nn,
                                  "../../data/train_input.dat",
                                  "../../data/train_target.dat" );
    time_t time_end = time(NULL);

    result << err[0] << '\t';

    std::ofstream nn_save( "../../result/nn.dat", std::ios::binary );
    nn.save(nn_save);
    nn_save.close();

    std::ifstream nn_load( "../../result/nn.dat", std::ios::binary );
    nn.load(nn_load);
    nn_load.close();

    CTester<FErrMAE> tester;
    tester.test<CReaderBinary>( err,
                                nn,
                                "../../data/test_input.dat",
                                "../../data/test_target.dat" );

    result << err[0] << '\t' << time_end - time_beg << std::endl;

    result.close();
    return 0;
}

void data_transform(void)
{
    std::ifstream train_input_txt( "../../data/test_input.csv", std::ios::binary );
    std::ifstream train_target_txt( "../../data/test_target.csv", std::ios::binary );
    std::ofstream train_input_dat( "../../data/test_input.dat", std::ios::binary );
    std::ofstream train_target_dat( "../../data/test_target.dat", std::ios::binary );

    for ( uint32 i = 0; i < pattern_num; ++i )
    {
        for ( uint32 j = 0; j < input_num; ++j )
        {
            double val;
            char comma;
            train_input_txt >> val >> comma;
            train_input_dat.write( ( const char*)&val, sizeof(val) );
        }

        for ( uint32 j = 0; j < output_num; ++j )
        {
            double val;
            char comma;
            train_target_txt >> val >> comma;
            train_target_dat.write( ( const char*)&val, sizeof(val) );
        }
    }

    train_input_txt.close();
    train_target_txt.close();
    train_input_dat.close();
    train_target_dat.close();
}

