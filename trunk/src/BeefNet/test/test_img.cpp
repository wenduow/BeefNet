#include "test_config.hpp"
#include "test_img.hpp"

#ifdef TEST_IMG

int32 main(void)
{
    MyNN nn;
    double err[output_num];
    time_t beg[10];
    time_t end[10];

    // test for 1 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 1 > trainer0;
    trainer0.open_input( "../../data/train_input.dat" );
    trainer0.open_target( "../../data/train_target.dat" );
    beg[0] = time(NULL);
    trainer0.train< false, false >( err, nn );
    end[0] = time(NULL);

    // test for 2 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 2 > trainer1;
    trainer1.open_input( "../../data/train_input.dat" );
    trainer1.open_target( "../../data/train_target.dat" );
    beg[1] = time(NULL);
    trainer1.train< false, false >( err, nn );
    end[1] = time(NULL);

    // test for 4 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 4 > trainer2;
    trainer2.open_input( "../../data/train_input.dat" );
    trainer2.open_target( "../../data/train_target.dat" );
    beg[2] = time(NULL);
    trainer2.train< false, false >( err, nn );
    end[2] = time(NULL);

    // test for 8 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 8 > trainer3;
    trainer3.open_input( "../../data/train_input.dat" );
    trainer3.open_target( "../../data/train_target.dat" );
    beg[3] = time(NULL);
    trainer3.train< false, false >( err, nn );
    end[3] = time(NULL);

    // test for 16 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 16 > trainer4;
    trainer4.open_input( "../../data/train_input.dat" );
    trainer4.open_target( "../../data/train_target.dat" );
    beg[4] = time(NULL);
    trainer4.train< false, false >( err, nn );
    end[4] = time(NULL);

    // test for 32 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 32 > trainer5;
    trainer5.open_input( "../../data/train_input.dat" );
    trainer5.open_target( "../../data/train_target.dat" );
    beg[5] = time(NULL);
    trainer5.train< false, false >( err, nn );
    end[5] = time(NULL);

    // test for 64 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 64 > trainer6;
    trainer6.open_input( "../../data/train_input.dat" );
    trainer6.open_target( "../../data/train_target.dat" );
    beg[6] = time(NULL);
    trainer6.train< false, false >( err, nn );
    end[6] = time(NULL);

    // test for 128 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 128 > trainer7;
    trainer7.open_input( "../../data/train_input.dat" );
    trainer7.open_target( "../../data/train_target.dat" );
    beg[7] = time(NULL);
    trainer7.train< false, false >( err, nn );
    end[7] = time(NULL);

    // test for 256 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 256 > trainer8;
    trainer8.open_input( "../../data/train_input.dat" );
    trainer8.open_target( "../../data/train_target.dat" );
    beg[8] = time(NULL);
    trainer8.train< false, false >( err, nn );
    end[8] = time(NULL);

    // test for 512 thread
    CTrainer< MyNN, MyInput, MyTarget, MyErr, 512 > trainer9;
    trainer9.open_input( "../../data/train_input.dat" );
    trainer9.open_target( "../../data/train_target.dat" );
    beg[9] = time(NULL);
    trainer9.train< false, false >( err, nn );
    end[9] = time(NULL);

    std::ofstream result( "../../result/result_img.txt", std::ios::app );

    for ( uint32 i = 0; i < 10; ++i )
    {
        result << end[i] - beg[i] << ',';
    }

    result << std::endl;

    result.close();
    return 0;
}

#endif // TEST_IMG

