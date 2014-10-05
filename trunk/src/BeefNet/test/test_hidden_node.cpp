#include <fstream>
#include <iostream>
#include "test_hidden_node.hpp"

#ifdef TEST_HIDDEN_NODE

std::ofstream result( "../../result/result_hidden_node.txt", std::ios::app );

template < uint32 HiddenNum >
class CTestHiddenNode
{
public:

    CTestHiddenNode(void)
    {
    }

    ~CTestHiddenNode(void)
    {
    }

    typedef CNN2Layer< CWeightLM<>,
                       input_num,
                       HiddenNum, FXferLogSig,
                       HiddenNum, FXferLogSig,
                       output_num, FXferLnr > MyNN;

    static void run(void)
    {
        MyNN nn;
        double err[output_num];

        CTrainer< MyNN, MyInput, MyTarget, MyErr > trainer;
        trainer.open_input( "../../data/train_input.dat" );
        trainer.open_target( "../../data/train_target.dat" );
        trainer.train<true>( err, nn );
        for ( uint32 i = 0; i < output_num; ++i )
        {
            result << err[i] << ',';
        }

        CTester< MyNN, MyInput, MyTarget, MyErr > tester;
        tester.open_input( "../../data/test_input.dat" );
        tester.open_target( "../../data/test_target.dat" );
        tester.test( err, nn );
        for ( uint32 i = 0; i < output_num; ++i )
        {
            result << err[i] << ',';
        }

        result << std::endl;
    }
};

int32 main(void)
{
    srand( (uint32)time(NULL) );

    CWeightBP<> weight_bp;
    CWeightQP<> weight_qp;
    CWeightRP<> weight_rp;
    CWeightLM<> weight_lm;

    result << sizeof(weight_bp) << ',' << sizeof(weight_qp) << ',' << sizeof(weight_rp) << ',' << sizeof(weight_lm) << ',' << std::endl;

    CTestHiddenNode<1>::run();
    CTestHiddenNode<2>::run();
    CTestHiddenNode<3>::run();
    CTestHiddenNode<4>::run();
    CTestHiddenNode<5>::run();
    CTestHiddenNode<6>::run();
    CTestHiddenNode<7>::run();
    CTestHiddenNode<8>::run();
    CTestHiddenNode<9>::run();
    CTestHiddenNode<10>::run();
    CTestHiddenNode<11>::run();
    CTestHiddenNode<12>::run();
    CTestHiddenNode<13>::run();
    CTestHiddenNode<14>::run();
    CTestHiddenNode<15>::run();
    CTestHiddenNode<16>::run();
    CTestHiddenNode<17>::run();
    CTestHiddenNode<18>::run();
    CTestHiddenNode<19>::run();
    CTestHiddenNode<20>::run();

    result.close();
    return 0;
}

#endif // TEST_TIME

