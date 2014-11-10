#ifndef UPDATE_BP_HPP_
#define UPDATE_BP_HPP_

#include "weight_vector.hpp"

namespace wwd
{

template < uint32 InputNum, class Param >
class CUpdateBP
    : public CWeightVector<InputNum>
{
public:

    CUpdateBP(void)
        : m_pattern_num(0)
        , m_learn_rate( (double)Param::learn_rate / 1000.0 )
    {
    }

    ~CUpdateBP(void)
    {
    }

    inline const CUpdateBP &operator>>( OUT CUpdateBP &other ) const
    {
        other.m_pattern_num = 0;

        return *this;
    }

    inline CUpdateBP &operator>>( IN const CUpdateBP &other )
    {
        m_pattern_num += other.m_pattern_num;

        return *this;
    }

    void forward(void)
    {
        for ( auto &i : m_weight )
        {
            i.forward();
        }
    }

    void backward(void)
    {
        for ( auto &i : m_weight )
        {
            i.backward();
        }

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

    CUpdateBP( IN CUpdateBP &other );
    inline CUpdateBP &operator=( IN const CUpdateBP &other );

private:

    uint32 m_pattern_num;

    const double m_learn_rate;
};

} // namespace wwd

#endif // UPDATE_BP_HPP_

