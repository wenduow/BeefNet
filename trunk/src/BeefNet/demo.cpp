//#define DEMO
#ifdef  DEMO

#include <ctime>
#include <iostream>
#include "demo.hpp"

int32 main(void)
{
    time_t beg = time(NULL);

    MyNN nn;
    double err[output_num];

    MyTrainer trainer;
    trainer.open_input( "../../data/train_input.dat" );
    trainer.open_target( "../../data/train_target.dat" );

    trainer.train< false, true >( err, nn );
    result << "training error: ";
    for ( const auto &i : err ) result << i << '\t';
    result << std::endl;

    MyTester tester;
    tester.open_input( "../../data/test_input.dat" );
    tester.open_target( "../../data/test_target.dat" );

    tester.test( err, nn );
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

