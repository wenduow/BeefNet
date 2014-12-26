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

    typedef WeightType<Param> Weight;

public:

    CWeightVector(void)
    {
        m_weight = new Weight[InputNum];
    }

    ~CWeightVector(void)
    {
        delete[] m_weight;
        m_weight = NULL;
    }

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

    void init(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].init();
        }
    }

    void forward(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].forward();
        }
    }

    void backward(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].backward();
        }
    }

    void update(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].update();
        }
    }

    template < class Neuron >
    void connect_input_neuron( INOUT Neuron &neuron )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            if ( m_weight[i].connect_input_node(neuron) )
            {
                neuron.connect_output_node( m_weight[i] );
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

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            gradient_sum += m_weight[i].get_gradient_sum();
        }

        return gradient_sum;
    }

    uint32 get_gradient_num(void) const
    {
        uint32 gradient_num = 0;

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            gradient_num += m_weight[i].get_gradient_num();
        }

        return gradient_num;
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].save(stream);
        }
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].load(stream);
        }
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    Weight *m_weight;
};

} // namespace wwd

#endif // WEIGHT_VECTOR_HPP_

