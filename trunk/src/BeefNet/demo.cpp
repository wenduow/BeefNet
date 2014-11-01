#ifdef  DEMO

#include "test/test_config.hpp"
#include "package.hpp"

std::ofstream result( "../../result/result_qp.txt", std::ios::app );

int32 main(void)
{
    /** definition of node number
        * (input, output and hidden node number)
        */
    const uint32 input_num  = 8;
    const uint32 output_num = 1;
    const uint32 hidden_num = 10;

    /** definition of network type, err function
        *  input and target file reader type
        */
    typedef CNN2Layer
    <
        CWeightQP<>, // default parameters for LM algorithm is
                        // lambda = 10, beta = 10.
        input_num,
        hidden_num, FXferLogSig, // log-sigmoid function
        hidden_num, FXferLogSig, // log-sigmoid function
        output_num, FXferLnr     // linear function
    > NN;
    typedef FErrMAE ErrFunction; // use mean absolute error
    typedef CReaderBinary<input_num>  Input;
    typedef CReaderBinary<output_num> Target;

    srand( (uint32)time(NULL) ); // set random seed
    NN nn;                       // create network structure

    /** neural networks training */
    double train_err[output_num];
    CTrainer< NN, Input, Target, ErrFunction > trainer;
    trainer.open_input( "../../data/train_input.dat" );
    trainer.open_target( "../../data/train_target.dat" );
    trainer.train<true>( train_err, nn ); // true is stop early

    /** neural networks testing */
    double test_err[output_num];
    CTester< NN, Input, Target, ErrFunction > tester;
    tester.open_input( "../../data/test_input.dat" );
    tester.open_target( "../../data/test_target.dat" );
    tester.test( test_err, nn );
    result << std::endl;

    return 0;
}

#endif // DEMO

