#ifndef WEIGHT_VECTOR_HPP_
#define WEIGHT_VECTOR_HPP_

#include "../../Utility/type.hpp"

namespace wwd
{

template < uint32 InputNum,
           template <class> class WeightType,
           class Param >
class CWeightVector
{
private:

    typedef CWeightVector< InputNum, WeightType, Param > ThisType;

public:

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i] >> other.m_weight[i];
        }

        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i] << other.m_weight[i];
        }

        return *this;
    }

    CWeightVector(void)
    {
    }

    ~CWeightVector(void)
    {
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
    }

    void update(void)
    {
        for ( auto &i : m_weight )
        {
            i.update();
        }
    }

    template < class Neuron >
    void connect_input_neuron( INOUT Neuron &neuron )
    {
        for ( auto &i : m_weight )
        {
            if ( i.connect_input_node(neuron) )
            {
                neuron.connect_output_node(i);
                break;
            }
        }
    }

    inline WeightType<Param> &get_weight( IN uint32 idx )
    {
        return m_weight[idx];
    }

    double get_gradient_sum(void) const
    {
        double gradient_sum = 0.0;

        for ( const auto &i : m_weight )
        {
            gradient_sum += i.get_gradient_sum();
        }

        return gradient_sum;
    }

    uint32 get_gradient_num(void) const
    {
        uint32 gradient_num = 0;

        for ( const auto &i : m_weight )
        {
            gradient_num += i.get_gradient_num();
        }

        return gradient_num;
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        for ( const auto &i : m_weight )
        {
            i.save(stream);
        }
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        for ( auto &i : m_weight )
        {
            i.load(stream);
        }
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        for ( const auto &i : m_weight )
        {
            i.print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    WeightType<Param> m_weight[InputNum];
};

} // namespace wwd

#endif // WEIGHT_VECTOR_HPP_

