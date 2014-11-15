#ifndef WEIGHT_BP_HPP_
#define WEIGHT_BP_HPP_

#include "weight_vector_itf.hpp"

namespace wwd
{

template < uint32 InputNum, class Param >
class CWeightBP
    : public IWeightVector<InputNum>
{
public:

    CWeightBP(void)
        : IWeightVector()
        , m_pattern_num(0)
        , m_learn_rate( (double)Param::learn_rate / 1000.0 )
    {
    }

    ~CWeightBP(void)
    {
    }

    void forward(void)
    {
        IWeightVector::forward();
    }

    void backward(void)
    {
        IWeightVector::backward();
        ++m_pattern_num;
    }

    void update(void)
    {
        for ( auto &i : m_weight )
        {
            i.update( - m_learn_rate
                      * i.get_gradient()
                      / (double)m_pattern_num );
        }

        m_pattern_num = 0;
    }

private:

    CWeightBP( IN const CWeightBP &other );
    inline CWeightBP &operator=( IN const CWeightBP &other );

private:

    uint32 m_pattern_num;
    const double m_learn_rate;
};

} // namespace wwd

#endif // WEIGHT_BP_HPP_

