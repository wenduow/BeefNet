#ifndef LAYER_HIDDEN_HPP_
#define LAYER_HIDDEN_HPP_

#include "../Weight/weight_vector.hpp"
#include "../Neuron/neuron.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 NeuronNum,
           uint32 OutputNum,
           class Xfer,
           template <class> class WeightType,
           class Param >
class CLayerHidden
{
public:

    enum
    {
        hidden_num = NeuronNum
    };

private:

    typedef CLayerHidden< InputNum,
                          NeuronNum,
                          OutputNum,
                          Xfer,
                          WeightType,
                          Param > ThisType;

public:

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i] >> other.m_weight_vector[i];
        }

        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i] << other.m_weight_vector[i];
        }

        return *this;
    }

    CLayerHidden(void)
    {
        connect_inner();
    }

    ~CLayerHidden(void)
    {
    }

    void forward(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].forward();
            m_neuron[i].forward();
        }
    }

    void backward(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_neuron[i].backward();
            m_weight_vector[i].backward();
        }
    }

    void update(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.update();
        }
    }

    template < class Layer >
    void connect_input_layer( INOUT Layer &layer )
    {
        for ( uint32 i = 0; i < Layer::hidden_num; ++i )
        {
            for ( auto &j : m_weight_vector )
            {
                j.connect_input_neuron( layer.get_hidden_node(i) );
            }
        }
    }

    inline CNeuronHidden< InputNum, OutputNum, Xfer > &
        get_hidden_node( IN uint32 idx )
    {
        return m_neuron[idx];
    }

    double get_gradient_sum(void) const
    {
        double gradient_sum = 0.0;

        for ( const auto &i : m_weight_vector )
        {
            gradient_sum += i.get_gradient_sum();
        }

        return gradient_sum;
    }

    uint32 get_gradient_num(void) const
    {
        uint32 gradient_num = 0;

        for ( const auto &i : m_weight_vector )
        {
            gradient_num += i.get_gradient_num();
        }

        return gradient_num;
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        for ( const auto &i : m_weight_vector )
        {
            i.save(stream);
        }
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        for ( auto &i : m_weight_vector )
        {
            i.load(stream);
        }
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

    void connect_inner(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_neuron[i].connect_input_weight_vector( m_weight_vector[i] );
        }
    }

private:

    CWeightVector< InputNum, WeightType, Param > m_weight_vector[NeuronNum];
    CNeuronHidden< InputNum, OutputNum, Xfer > m_neuron[NeuronNum];
};

} // namespace wwd

#endif // LAYER_HIDDEN_HPP_

