#include "test_efficiency.hpp"
#include "test_config.hpp"

#ifdef TEST_EFFICIENCY

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
    trainer_bp.train< true, true >( err, nn_bp );
    result << std::endl;

    // test for qp
    CTrainer< MyNNQP, MyInput, MyTarget, MyErr > trainer_qp;
    trainer_qp.open_input( "../../data/train_input.dat" );
    trainer_qp.open_target( "../../data/train_target.dat" );
    trainer_qp.train< true, true >( err, nn_qp );
    result << std::endl;

    // test for rp
    CTrainer< MyNNRP, MyInput, MyTarget, MyErr > trainer_rp;
    trainer_rp.open_input( "../../data/train_input.dat" );
    trainer_rp.open_target( "../../data/train_target.dat" );
    trainer_rp.train< true, true >( err, nn_rp );
    result << std::endl;

    // test for lm
    CTrainer< MyNNLM, MyInput, MyTarget, MyErr > trainer_lm;
    trainer_lm.open_input( "../../data/train_input.dat" );
    trainer_lm.open_target( "../../data/train_target.dat" );
    trainer_lm.train< true, true >( err, nn_lm );
    result << std::endl;

    result.close();
    return 0;
}

#endif // TEST_EFFICIENCY

