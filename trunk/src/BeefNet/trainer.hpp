#ifndef TRAINER_HPP_
#define TRAINER_HPP_

#include <thread>
#include <functional>
#include "tester.hpp"

namespace wwd
{

template < class NN, template <uint32> class Reader >
void image_function( INOUT NN &nn,
                     IN const Reader< NN::input_num > &reader_input,
                     IN const Reader< NN::output_num > &reader_target,
                     IN uint32 idx_beg,
                     IN uint32 idx_end )
{
    for ( uint32 i = idx_beg; i < idx_end; ++i )
    {
        nn.set_input( reader_input.get_pattern(i) );
        nn.forward();

        nn.set_target( reader_target.get_pattern(i) );
        nn.backward();
    }
}

template < class Err,
           uint32 ImageNum = 8,
           bool StopEarly = true,
           uint32 MaxEpoch = 2000,
           uint32 ValidTimes = 6,
           int32  MinGradientChange = -5 >
class CTrainer
{
private:

    typedef CTrainer< Err,
                      StopEarly,
                      ImageNum,
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
    }

    template < template <uint32> class Reader, class NN >
    void train( OUT double (&err)[ NN::output_num ],
                INOUT NN &nn,
                IN const char *input_path,
                IN const char *target_path )
    {
        m_valid_times = 0;
        m_gradient    = DOUBLE_MAX;
        m_err         = 0.0;

        Reader< NN::input_num > input(input_path);
        Reader< NN::output_num > target(target_path);

        uint32 idx_beg[ImageNum];
        uint32 idx_end[ImageNum];
        generate_pattern_idx( idx_beg, idx_end, target.get_pattern_num() );

        NN          nn_img[ImageNum];
        std::thread img_thread[ImageNum];

        for ( uint32 i = 0; i < MaxEpoch; ++i )
        {
            for ( uint32 j = 0; j < ImageNum; ++j )
            {
                nn >> nn_img[j];
                img_thread[j] = std::thread
                    ( image_function< NN, Reader >,
                      std::ref( nn_img[j] ),
                      std::cref(input),
                      std::cref(target),
                      idx_beg[j],
                      idx_end[j] );
            }

            for ( uint32 j = 0; j < ImageNum; ++j )
            {
                img_thread[j].join();
                nn << nn_img[j];
            }

//             if ( stop_early<StopEarly>( nn.get_gradient_abs() ) )
//             {
//                 break;
//             }

            nn.update();
        }

        input.close();
        target.close();

        CTester<Err> tester;
        tester.test<Reader>( err, nn, input_path, target_path );
    }

private:

    void generate_pattern_idx( OUT uint32 (&idx_beg)[ImageNum],
                               OUT uint32 (&idx_end)[ImageNum],
                               IN uint32  pattern_num )
    {
        for ( uint32 i = 0; i < ImageNum; ++i )
        {
            idx_beg[i] = pattern_num / ImageNum * i;
            idx_end[i] = ( ImageNum == ( i + 1 ) )
                       ? pattern_num
                       : ( pattern_num / ImageNum * ( i + 1 ) );
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

    CTrainer( IN const ThisType &other );
    CTrainer &operator=( IN const ThisType &other );

private:

    uint32       m_valid_times;
    double       m_gradient;
    double       m_err;
    const double m_min_gradient_change;
};

} // namespace wwd

#endif // TRAINER_HPP_

