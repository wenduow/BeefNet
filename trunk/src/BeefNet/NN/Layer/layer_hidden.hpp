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

    typedef CWeightVector< InputNum, WeightType, Param > WeightVector;
    typedef CNeuronHidden< InputNum, OutputNum, Xfer > Neuron;

public:

    CLayerHidden(void)
    {
        m_weight_vector = new WeightVector[NeuronNum];
        m_neuron = new Neuron[NeuronNum];

        connect_inner();
    }

    ~CLayerHidden(void)
    {
        delete[] m_weight_vector;
        m_weight_vector = NULL;
        delete[] m_neuron;
        m_neuron = NULL;
    }

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

    void init(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].init();
        }
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
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].update();
        }
    }

    template < class Layer >
    void connect_input_layer( INOUT Layer &layer )
    {
        for ( uint32 i = 0; i < Layer::hidden_num; ++i )
        {
            for ( uint32 j = 0; j < NeuronNum; ++j )
            {
                m_weight_vector[j]
                    .connect_input_neuron( layer.get_hidden_node(i) );
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

        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            gradient_sum += m_weight_vector[i].get_gradient_sum();
        }

        return gradient_sum;
    }

    uint32 get_gradient_num(void) const
    {
        uint32 gradient_num = 0;

        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            gradient_num += m_weight_vector[i].get_gradient_num();
        }

        return gradient_num;
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].save(stream);
        }
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].load(stream);
        }
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].print_weight();
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

    WeightVector *m_weight_vector;
    Neuron *m_neuron;
};

} // namespace wwd

#endif // LAYER_HIDDEN_HPP_

