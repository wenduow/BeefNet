#ifndef TRAINER_HPP_
#define TRAINER_HPP_

#include <thread>
#include <functional>
#include "tester.hpp"

namespace wwd
{

template < class NN, class InputReader, class TargetReader >
void img_fxn( INOUT NN &nn,
              IN const InputReader &input,
              IN const TargetReader &target,
              IN uint32 idx_beg,
              IN uint32 idx_end )
{
    for ( uint32 i = idx_beg; i < idx_end; ++i )
    {
        nn.set_input( input.get_pattern(i) );
        nn.set_target( target.get_pattern(i) );

        nn.forward();
        nn.backward();
    }
}

template < class NN,
           class InputReader,
           class TargetReader,
           class Err,
           uint32 ImgNum = 8,
           uint32 MaxEpoch = 2000,
           uint32 ValidTimes = 6,
           int32  MinGradientChange = -6 >
class CTrainer
{
private:

    typedef CTrainer< NN,
                      InputReader,
                      TargetReader,
                      Err,
                      ImgNum,
                      MaxEpoch,
                      ValidTimes,
                      MinGradientChange > ThisType;

public:

    CTrainer(void)
        : m_valid_times(0)
        , m_gradient(DOUBLE_MAX)
        , m_err(0.0)
        , m_min_gradient_change( pow( 10.0, MinGradientChange ) )
    {
    }

    ~CTrainer(void)
    {
        m_input.close();
        m_target.close();
    }

    template < bool  StopEarly, uint32 OutputNum >
    void train( OUT double (&err)[OutputNum], INOUT NN &nn )
    {
        m_valid_times = 0;
        m_gradient    = DOUBLE_MAX;
        m_err         = 0.0;

        uint32 idx_beg[ImgNum];
        uint32 idx_end[ImgNum];
        generate_pattern_idx( idx_beg, idx_end, m_target.get_pattern_num() );

        NN          nn_img[ImgNum];
        std::thread img_thread[ImgNum];

        for ( uint32 i = 0; i < MaxEpoch; ++i )
        {
            for ( uint32 j = 0; j < ImgNum; ++j )
            {
                nn >> nn_img[j];
                img_thread[j] = std::thread
                    ( img_fxn< NN, InputReader, TargetReader >,
                      std::ref( nn_img[j] ),
                      std::cref(m_input),
                      std::cref(m_target),
                      idx_beg[j],
                      idx_end[j] );
            }

            for ( uint32 j = 0; j < ImgNum; ++j )
            {
                img_thread[j].join();
                nn << nn_img[j];
            }

            if ( stop_early<StopEarly>( nn.get_gradient_abs() ) )
            {
                break;
            }

            nn.update();
        }

        m_tester.test( err, nn );
    }

    void open_input( IN const char *path )
    {
        m_input.open(path);
        m_tester.open_input(path);
    }

    void open_target( IN const char *path )
    {
        m_target.open(path);
        m_tester.open_target(path);
    }

private:

    CTrainer( IN const ThisType &other );
    inline ThisType &operator=( IN const ThisType &other );

    void generate_pattern_idx( OUT uint32 (&idx_beg)[ImgNum],
                               OUT uint32 (&idx_end)[ImgNum],
                               IN uint32  pattern_num )
    {
        for ( uint32 i = 0; i < ImgNum; ++i )
        {
            idx_beg[i] = pattern_num / ImgNum * i;
            idx_end[i] = ( ImgNum == ( i + 1 ) )
                       ? pattern_num
                       : ( pattern_num / ImgNum * ( i + 1 ) );
        }
    }

    template < bool StopEarly >
    bool stop_early( IN double gradient )
    {
        bool choose_stop_early = StopEarly;

        if (choose_stop_early)
        {
            if ( gradient < m_min_gradient_change )
            {
                return true;
            }
            else if ( abs( gradient - m_gradient ) < m_min_gradient_change )
            {
                if ( m_valid_times >= ValidTimes )
                {
                    return true;
                }
                else
                {
                    ++m_valid_times;
                }
            }
            else
            {
                m_valid_times = 0;
            }

            m_gradient = gradient;
        }

        return false;
    }

private:

    InputReader  m_input;
    TargetReader m_target;
    CTester< NN, InputReader, TargetReader, Err > m_tester;

    uint32       m_valid_times;
    double       m_gradient;
    double       m_err;
    const double m_min_gradient_change;
};

} // namespace wwd

#endif // TRAINER_HPP_

