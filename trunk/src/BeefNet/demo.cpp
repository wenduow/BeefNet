#define DEMO
#ifdef  DEMO

#include <ctime>
#include <iostream>
#include "demo.hpp"

int32 main(void)
{
    time_t beg = time(NULL);

    MyInput  train_input( "../../data/train_input.dat" );
    MyTarget train_target( "../../data/train_target.dat" );
    MyInput  test_input( "../../data/test_input.dat" );
    MyTarget test_target( "../../data/test_target.dat" );

    MyNN nn;
    double err[output_num];

    MyTrainer trainer;
    trainer.train< false, true >( err, nn, train_input, train_target );
    result << "training error: ";
    for ( const auto &i : err ) result << i << '\t';
    result << std::endl;

    MyTester tester;
    tester.test( err, nn, test_input, test_target );
    result << "testing error: ";
    for ( const auto &i : err ) result << i << '\t';
    result << std::endl;

    nn.save( "../../data/trained.dat" );

    time_t end = time(NULL);
    result << "running time: " << end - beg << 's' << std::endl;
    result.close();
    return 0;
}

#endif // DEMO

