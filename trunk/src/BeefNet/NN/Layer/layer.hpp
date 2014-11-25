#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "../Weight/weight_vector.hpp"
#include "../Neuron/neuron_input.hpp"
#include "../Neuron/neuron.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 NeuronNum,
           uint32 OutputNum,
           class Xfer,
           template <class> class WeightType,
           class Param >
class CLayer
{
public:

    enum
    {
        hidden_num = NeuronNum
    };

private:

    typedef CLayer< InputNum,
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

    CLayer(void)
    {
        connect_inner();
    }

    ~CLayer(void)
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

    inline CNeuron< InputNum, OutputNum, Xfer > &
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
            m_weight_vector[i].connect_output_neuron( m_neuron[i] );
        }
    }

private:

    CWeightVector< InputNum, WeightType, Param > m_weight_vector[NeuronNum];
    CNeuron< InputNum, OutputNum, Xfer > m_neuron[NeuronNum];
};

template < uint32 InputNum,
           uint32 NeuronNum,
           uint32 OutputNum,
           class Xfer,
           class Param >
class CLayer< InputNum,
              NeuronNum,
              OutputNum,
              Xfer,
              CWeightLM,
              Param >
{
public:

    enum
    {
        hidden_num = NeuronNum
    };

private:

    typedef CLayer< InputNum,
                    NeuronNum,
                    OutputNum,
                    Xfer,
                    CWeightLM,
                    Param > ThisType;

public:

    CLayer(void)
    {
        connect_inner();
    }

    ~CLayer(void)
    {
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

    void forward(void)
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_weight_vector[i].forward();
            m_neuron[i].forward();
        }
    }

    void backward( IN double err )
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            m_neuron[i].backward();
            m_weight_vector[i].backward(err);
        }
    }

    void update(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.update();
        }
    }

    void revert(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.revert();
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

    inline CNeuron< InputNum, OutputNum, Xfer > &
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
            m_weight_vector[i].connect_output_neuron( m_neuron[i] );
        }
    }

private:

    CWeightVector< InputNum, CWeightLM, Param > m_weight_vector[NeuronNum];
    CNeuron< InputNum, OutputNum, Xfer > m_neuron[NeuronNum];
};

} // namespace wwd

#endif // LAYER_HPP_

