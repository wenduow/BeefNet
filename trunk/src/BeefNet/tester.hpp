#ifndef TESTER_HPP_
#define TESTER_HPP_

#include "predictor.hpp"

namespace wwd
{

template < class Err >
class CTester
{
private:

    typedef CTester<Err> ThisType;

public:

    CTester(void)
    {
    }

    ~CTester(void)
    {
    }

    template < template <uint32> class Reader, class NN >
    void test( OUT double (&err)[ NN::output_num ],
               INOUT NN &nn,
               IN const char *input_path,
               IN const char *target_path ) const
    {
        Reader< NN::output_num > target(target_path);
        uint32 pattern_num   = target.get_pattern_num();
        double **predict_tmp = new double*[ NN::output_num ];
        double **target_tmp  = new double*[ NN::output_num ];

        for ( uint32 i = 0; i < NN::output_num; ++i )
        {
            predict_tmp[i] = new double[pattern_num];
            target_tmp[i]  = new double[pattern_num];
        }

        for ( uint32 i = 0; i < pattern_num; ++i )
        {
            CPredictor predictor;
            double predicted[ NN::output_num ];
            predictor.predict<Reader>( predicted, nn, input_path, i );

            for ( uint32 j = 0; j < NN::output_num; ++j )
            {
                predict_tmp[j][i] = predicted[j];
                target_tmp[j][i]  = target.get_pattern(i)[j];
            }
        }

        target.close();

        for ( uint32 i = 0; i < NN::output_num; ++i )
        {
            err[i] = m_err_fxn( predict_tmp[i], target_tmp[i], pattern_num );
        }

        for ( uint32 i = 0; i < NN::output_num; ++i )
        {
            delete[] predict_tmp[i];
            delete[] target_tmp[i];
        }
        delete[] predict_tmp;
        delete[] target_tmp;
    }

private:

    Err m_err_fxn;
};

} // namespace wwd

#endif // TESTER_HPP_

