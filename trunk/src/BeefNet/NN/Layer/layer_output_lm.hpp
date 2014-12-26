#ifndef LAYER_OUTPUT_LM_HPP_
#define LAYER_OUTPUT_LM_HPP_

#include "layer_output.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 OutputNum,
           class Xfer,
           class Param >
class CLayerOutput< InputNum,
                    OutputNum,
                    Xfer,
                    CWeightLM,
                    Param >
{
private:

    typedef CLayerOutput< InputNum,
                          OutputNum,
                          Xfer,
                          CWeightLM,
                          Param > ThisType;
    
    typedef CWeightVector< InputNum, CWeightLM, Param > WeightVector;
    typedef CNeuronHidden< InputNum, 1, Xfer > Output;
    typedef CNeuronTarget Target;

public:

    CLayerOutput(void)
    {
        connect_inner();
    }

    ~CLayerOutput(void)
    {
    }

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_weight_vector[i] >> other.m_weight_vector[i];
        }

        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_weight_vector[i] << other.m_weight_vector[i];
        }

        return *this;
    }

    void init(void)
    {
        for ( auto &i : m_weight_vector )
        {
            i.init();
        }
    }

    void forward(void)
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_weight_vector[i].forward();
            m_output[i].forward();
        }
    }

    void backward( IN double err, IN uint32 output_idx )
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_target[i].set_output_value( ( i == output_idx ) ? 1.0 : 0.0 );
            m_output[i].backward();
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

    void set_target( IN const double *target )
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_target[i].set_target( target[i] );
        }
    }

    double get_error( IN uint32 idx )
    {
        m_target[idx].backward();

        return m_target[idx].get_output_value();
    }

    void get_output( OUT double (&output)[OutputNum] ) const
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            output[i] = m_output[i].IPathForward::get_output_value();
        }
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
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            m_output[i].connect_input_weight_vector( m_weight_vector[i] );
            m_target[i].connect_input_neuron( m_output[i] );
        }
    }

private:

    WeightVector m_weight_vector[OutputNum];
    Output m_output[OutputNum];
    Target m_target[OutputNum];
};

} // namespace wwd

#endif // LAYER_OUTPUT_LM_HPP_

