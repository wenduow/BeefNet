#ifndef LAYER_WEIGHT_HPP_
#define LAYER_WEIGHT_HPP_

#include "../weight/weight_vector.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 NeuronNum,
           template < uint32, class > class WeightType,
           class WeightParam >
class CLayerWeight
{
public:

    enum
    {
        input_num = InputNum
    };

public:

    CLayerWeight(void)
    {
    }

    ~CLayerWeight(void)
    {
    }

    void forward(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.forward();
        }
    }

    void backward(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.backward();
        }
    }

    void update(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.update();
        }
    }

    /** connect input neuron layer */
    template < class Input >
    void connect_input( INOUT Input &input )
    {
        for ( auto &i : m_weight_vector )
        {
            i.connect_input(input);
        }
    }

    inline WeightType< InputNum, WeightParam > &
        get_weight_vector( IN uint32 idx )
    {
        return m_weight_vector[idx];
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        for ( const auto &i : m_weight_vector )
        {
            i.print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    WeightType< InputNum, WeightParam > m_weight_vector[NeuronNum];
};

}

#endif // LAYER_WEIGHT_HPP_

