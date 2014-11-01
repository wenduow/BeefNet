#include <fstream>
#include "test_bp.hpp"

#ifdef TEST_BP

#define RUN_NN(hidden_num) \
    MyNN##hidden_num nn##hidden_num; \
    CTrainer< MyErr, false > trainer##hidden_num; \
    trainer##hidden_num.train<CReaderBinary> \
        ( err, \
          nn##hidden_num, \
          "../../data/train_input.dat", \
          "../../data/train_target.dat" ); \
    result << std::endl;

std::ofstream result( "../../result/result_bp.txt", std::ios::app );

void convert(void);

int32 main(void)
{
    // convert();

    srand( (uint32)time(NULL) );

    double err[ MyNN1::output_num ];

    RUN_NN(1);
    RUN_NN(2);
    RUN_NN(3);
    RUN_NN(4);
    RUN_NN(5);
    RUN_NN(6);
    RUN_NN(7);
    RUN_NN(8);
    RUN_NN(9);
    RUN_NN(10);
    RUN_NN(11);
    RUN_NN(12);
    RUN_NN(13);
    RUN_NN(14);
    RUN_NN(15);
    RUN_NN(16);
    RUN_NN(17);
    RUN_NN(18);
    RUN_NN(19);
    RUN_NN(20);
    RUN_NN(21);
    RUN_NN(22);
    RUN_NN(23);
    RUN_NN(24);
    RUN_NN(25);
    RUN_NN(26);
    RUN_NN(27);
    RUN_NN(28);
    RUN_NN(29);
    RUN_NN(30);

    result.close();
    return 0;
}

void convert(void)
{
    std::ifstream f_in( "../../data/train_input.txt", std::ios::binary );
    if ( !f_in.fail() )
    {
        std::ofstream f_out( "../../data/train_input.dat", std::ios::binary );
        while ( !f_in.eof() )
        {
            double val;
            f_in >> val;
            f_out.write( (const char*)&val, sizeof(val) );
        }
        f_out.close();
    }
    f_in.close();

    f_in.open( "../../data/train_target.txt", std::ios::binary );
    if ( !f_in.fail() )
    {
        std::ofstream f_out( "../../data/train_target.dat", std::ios::binary );
        while ( !f_in.eof() )
        {
            double val;
            f_in >> val;
            f_out.write( (const char*)&val, sizeof(val) );
        }
        f_out.close();
    }
    f_in.close();

    f_in.open( "../../data/test_input.txt", std::ios::binary );
    if ( !f_in.fail() )
    {
        std::ofstream f_out( "../../data/test_input.dat", std::ios::binary );
        while ( !f_in.eof() )
        {
            double val;
            f_in >> val;
            f_out.write( (const char*)&val, sizeof(val) );
        }
        f_out.close();
    }
    f_in.close();

    f_in.open( "../../data/test_target.txt", std::ios::binary );
    if ( !f_in.fail() )
    {
        std::ofstream f_out( "../../data/test_target.dat", std::ios::binary );
        while ( !f_in.eof() )
        {
            double val;
            f_in >> val;
            f_out.write( (const char*)&val, sizeof(val) );
        }
        f_out.close();
    }
    f_in.close();
}

#endif // TEST_EFFICIENCY

