#include <fstream>
#include "test_time.hpp"

#ifdef TEST_TIME

std::ofstream result( "../../result/result_time.txt", std::ios::app );

int32 main(void)
{
    srand( (uint32)time(NULL) );

    MyNNBP nn_bp;
    MyNNQP nn_qp;
    MyNNRP nn_rp;
    MyNNLM nn_lm;

    double err[output_num];
    time_t beg[4];
    time_t end[4];

    // test for bp
    CTrainer< MyNNBP, MyInput, MyTarget, MyErr > trainer_bp;
    trainer_bp.open_input( "../../data/train_input.dat" );
    trainer_bp.open_target( "../../data/train_target.dat" );
    beg[0] = time(NULL);
    trainer_bp.train<false>( err, nn_bp );
    end[0] = time(NULL);

    // test for qp
    CTrainer< MyNNQP, MyInput, MyTarget, MyErr > trainer_qp;
    trainer_qp.open_input( "../../data/train_input.dat" );
    trainer_qp.open_target( "../../data/train_target.dat" );
    beg[1] = time(NULL);
    trainer_qp.train<false>( err, nn_qp );
    end[1] = time(NULL);

    // test for rp
    CTrainer< MyNNRP, MyInput, MyTarget, MyErr > trainer_rp;
    trainer_rp.open_input( "../../data/train_input.dat" );
    trainer_rp.open_target( "../../data/train_target.dat" );
    beg[2] = time(NULL);
    trainer_rp.train<false>( err, nn_rp );
    end[2] = time(NULL);

    // test for lm
    CTrainer< MyNNLM, MyInput, MyTarget, MyErr > trainer_lm;
    trainer_lm.open_input( "../../data/train_input.dat" );
    trainer_lm.open_target( "../../data/train_target.dat" );
    beg[3] = time(NULL);
    trainer_lm.train<false>( err, nn_lm );
    end[3] = time(NULL);

    for ( uint32 i = 0; i < 4; ++i )
    {
        result << end[i] - beg[i] << ',';
    }
    result << std::endl;

    result.close();
    return 0;
}

#endif // TEST_TIME

