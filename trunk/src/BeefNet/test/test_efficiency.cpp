#include <fstream>
#include "test_efficiency.hpp"

#ifdef TEST_EFFICIENCY

std::ofstream result( "../../result/result_efficiency.txt", std::ios::app );

int32 main(void)
{
    srand( (uint32)time(NULL) );

    MyNNBP nn_bp;
    MyNNQP nn_qp;
    MyNNRP nn_rp;
    MyNNLM nn_lm;

    double err[output_num];

    // test for bp
    CTrainer< MyNNBP, MyInput, MyTarget, MyErr > trainer_bp;
    trainer_bp.open_input( "../../data/train_input.dat" );
    trainer_bp.open_target( "../../data/train_target.dat" );
    trainer_bp.train<true>( err, nn_bp );
    CTester< MyNNBP, MyInput, MyTarget, MyErr > tester_bp;
    tester_bp.open_input( "../../data/test_input.dat" );
    tester_bp.open_target( "../../data/test_target.dat" );
    tester_bp.test( err, nn_bp );
    result << std::endl;

    // test for qp
    CTrainer< MyNNQP, MyInput, MyTarget, MyErr > trainer_qp;
    trainer_qp.open_input( "../../data/train_input.dat" );
    trainer_qp.open_target( "../../data/train_target.dat" );
    trainer_qp.train<true>( err, nn_qp );
    CTester< MyNNQP, MyInput, MyTarget, MyErr > tester_qp;
    tester_qp.open_input( "../../data/test_input.dat" );
    tester_qp.open_target( "../../data/test_target.dat" );
    tester_qp.test( err, nn_qp );
    result << std::endl;

    // test for rp
    CTrainer< MyNNRP, MyInput, MyTarget, MyErr > trainer_rp;
    trainer_rp.open_input( "../../data/train_input.dat" );
    trainer_rp.open_target( "../../data/train_target.dat" );
    trainer_rp.train<true>( err, nn_rp );
    CTester< MyNNRP, MyInput, MyTarget, MyErr > tester_rp;
    tester_rp.open_input( "../../data/test_input.dat" );
    tester_rp.open_target( "../../data/test_target.dat" );
    tester_rp.test( err, nn_rp );
    result << std::endl;

    // test for lm
    CTrainer< MyNNLM, MyInput, MyTarget, MyErr > trainer_lm;
    trainer_lm.open_input( "../../data/train_input.dat" );
    trainer_lm.open_target( "../../data/train_target.dat" );
    trainer_lm.train<true>( err, nn_lm );
    CTester< MyNNLM, MyInput, MyTarget, MyErr > tester_lm;
    tester_lm.open_input( "../../data/test_input.dat" );
    tester_lm.open_target( "../../data/test_target.dat" );
    tester_lm.test( err, nn_lm );
    result << std::endl;

    result.close();
    return 0;
}

#endif // TEST_EFFICIENCY

