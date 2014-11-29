#include "demo.hpp"

void data_transform(void);

int32 main(void)
{
//     data_transform();
//     return 0;

    srand( (uint32)time(NULL) );

    CNN2Layer< input_num,
               hidden_num, FXferLogSig,
               hidden_num, FXferLogSig,
               output_num, FXferLnr,
               MyWeight, MyParam > nn;
    
    double err[1];
    CTrainer< FErrMAE, thread_num, false > trainer;

    time_t time_beg = time(NULL);
    trainer.train<CReaderBinary>( err,
                                  nn,
                                  "../../data/train_input.dat",
                                  "../../data/train_target.dat" );
    time_t time_end = time(NULL);

    result << thread_num << '\t' << time_end - time_beg << '\t' << err[0] << std::endl;

    result.close();
    return 0;
}

void data_transform(void)
{
    std::ifstream train_input_txt( "../../data/train_input.csv", std::ios::binary );
    std::ifstream train_target_txt( "../../data/train_target.csv", std::ios::binary );
    std::ofstream train_input_dat( "../../data/train_input.dat", std::ios::binary );
    std::ofstream train_target_dat( "../../data/train_target.dat", std::ios::binary );

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

