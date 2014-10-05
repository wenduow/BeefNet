#ifndef TESTER_HPP_
#define TESTER_HPP_

#include "predictor.hpp"

namespace wwd
{

template < class NN, class InputReader, class TargetReader, class Err >
class CTester
{
private:

    typedef CTester< NN, InputReader, TargetReader, Err > ThisType;

public:

    CTester(void)
    {
    }

    ~CTester(void)
    {
        m_target.close();
    }

    template < uint32 OutputNum >
    void test( OUT double (&err)[OutputNum], INOUT NN &nn ) const
    {
        uint32 pattern_num   = m_target.get_pattern_num();
        double **predict_tmp = new double*[OutputNum];
        double **target_tmp  = new double*[OutputNum];

        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            predict_tmp[i] = new double[pattern_num];
            target_tmp[i]  = new double[pattern_num];
        }

        for ( uint32 i = 0; i < pattern_num; ++i )
        {
            double predicted[OutputNum];
            m_predictor.predict( predicted, nn, i );

            for ( uint32 j = 0; j < OutputNum; ++j )
            {
                predict_tmp[j][i] = predicted[j];
                target_tmp[j][i]  = m_target.get_pattern(i)[j];
            }
        }

        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            err[i] = m_err_fxn( predict_tmp[i], target_tmp[i], pattern_num );
        }

        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            delete[] predict_tmp[i];
            delete[] target_tmp[i];
        }
        delete[] predict_tmp;
        delete[] target_tmp;

//         for ( uint32 i = 0; i < OutputNum; ++i )
//         {
//             result << err[i] << ',';
//         }
    }

    void open_input( IN const char *path )
    {
        m_predictor.open_input(path);
    }

    void open_target( IN const char *path )
    {
        m_target.open(path);
    }

private:

    CTester( IN const ThisType &other );
    inline ThisType &operator=( IN const ThisType &other );

private:

    InputReader m_input;
    TargetReader m_target;
    CPredictor< NN, InputReader > m_predictor;
    Err m_err_fxn;
};

} // namespace wwd

#endif // TESTER_HPP_

