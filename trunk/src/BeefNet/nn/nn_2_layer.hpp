#ifndef NN_2_LAYER_HPP_
#define NN_2_LAYER_HPP_

#include "nn_itf.hpp"

namespace wwd
{

template < class Weight,
           uint32 InputNum,
           uint32 NeuronNum0, class Xfer0,
           uint32 NeuronNum1, class Xfer1,
           uint32 OutputNum,  class XferOutput >
class CNN2Layer
    : public INN
{
private:

    typedef CNN2Layer< Weight,
                       InputNum,
                       NeuronNum0, Xfer0,
                       NeuronNum1, Xfer1,
                       OutputNum,  XferOutput > ThisType;

public:

    CNN2Layer(void)
        : INN()
    {
        connect();
    }

    ~CNN2Layer(void)
    {
    }

    // TODO: descriptize these overloaded operators
    const ThisType &operator>>( OUT ThisType &other ) const
    {
        map_to(other);
        return *this;
    }

    ThisType &operator<<( INOUT ThisType &other )
    {
        reduce_from(other);
        return *this;
    }

    void set_input( IN const double *input )
    {
        INN::set_input( m_input, input );
    }

    void set_target( IN const double *target )
    {
        INN::set_target( m_target, target );
    }

    template < uint32 OutputNum >
    void get_output( OUT double (&output)[OutputNum] ) const
    {
        INN::get_output( output, m_output );
    }

    void forward(void)
    {
        INN::forward(m_input);
        INN::forward( m_bias_0,
                      m_weight_bias_0,      m_weight_neuron_0,
                      m_neuron_0 );
        INN::forward( m_bias_1,
                      m_weight_bias_1,      m_weight_neuron_1,
                      m_neuron_1 );
        INN::forward( m_bias_output,
                      m_weight_bias_output, m_weight_output,
                      m_output );
    }

    void backward(void)
    {
        INN::backward(m_target);
        INN::backward( m_weight_bias_output, m_weight_output,   m_output );
        INN::backward( m_weight_bias_1,      m_weight_neuron_1, m_neuron_1 );
        INN::backward( m_weight_bias_0,      m_weight_neuron_0, m_neuron_0 );
    }

    void update(void)
    {
        INN::update( m_weight_bias_0,      m_weight_neuron_0 );
        INN::update( m_weight_bias_1,      m_weight_neuron_1 );
        INN::update( m_weight_bias_output, m_weight_output );
    }

    double get_gradient_abs(void) const
    {
        return ( INN::get_gradient_abs_sum( m_weight_bias_0,
                                            m_weight_neuron_0 )
               + INN::get_gradient_abs_sum( m_weight_bias_1,
                                            m_weight_neuron_1 )
               + INN::get_gradient_abs_sum( m_weight_bias_output,
                                            m_weight_output ) )
             / (double)( INN::get_weight_num( m_weight_bias_0,
                                              m_weight_neuron_0 )
                       + INN::get_weight_num( m_weight_bias_1,
                                              m_weight_neuron_1 )
                       + INN::get_weight_num( m_weight_bias_output,
                                              m_weight_output ) );
    }

#if ( defined _DEBUG || defined PRINT_WEIGHT )

    void print(void) const
    {
        INN::print_input(m_input);
        INN::print( m_weight_bias_0,      m_weight_neuron_0, m_neuron_0 );
        INN::print( m_weight_bias_1,      m_weight_neuron_1, m_neuron_1 );
        INN::print( m_weight_bias_output, m_weight_output,   m_output );
        INN::print_target(m_target);
    }

#endif // _DEBUG || PRINT_WEIGHT

    void save( IN const char *path ) const
    {
        INN::save( *this, path );
    }

    void load( IN const char *path )
    {
        INN::load( *this, path );
    }

private:

    CNN2Layer( IN const ThisType &other );
    inline ThisType &operator=( IN const ThisType &other );

    void connect(void)
    {
        INN::connect( m_bias_0,      m_weight_bias_0,
                      m_input,       m_weight_neuron_0,
                      m_neuron_0 );
        INN::connect( m_bias_1,      m_weight_bias_1,
                      m_neuron_0,    m_weight_neuron_1,
                      m_neuron_1 );
        INN::connect( m_bias_output, m_weight_bias_output,
                      m_neuron_1,    m_weight_output,
                      m_output );
        INN::connect( m_output, m_target );
    }

    void map_to( OUT ThisType &other ) const
    {
        INN::map_to( other.m_weight_bias_0,      m_weight_bias_0,
                     other.m_weight_neuron_0,    m_weight_neuron_0 );
        INN::map_to( other.m_weight_bias_1,      m_weight_bias_1,
                     other.m_weight_neuron_1,    m_weight_neuron_1 );
        INN::map_to( other.m_weight_bias_output, m_weight_bias_output,
                     other.m_weight_output,      m_weight_output );
    }

    void reduce_from( IN const ThisType &other )
    {
        INN::reduce_from( m_weight_bias_0,      other.m_weight_bias_0,
                          m_weight_neuron_0,    other.m_weight_neuron_0 );
        INN::reduce_from( m_weight_bias_1,      other.m_weight_bias_1,
                          m_weight_neuron_1,    other.m_weight_neuron_1 );
        INN::reduce_from( m_weight_bias_output, other.m_weight_bias_output,
                          m_weight_output,      other.m_weight_output );
    }

private:

    CInput<NeuronNum0> m_input[InputNum];

    CInput<NeuronNum0> m_bias_0;
    Weight m_weight_bias_0[NeuronNum0];
    Weight m_weight_neuron_0[NeuronNum0][InputNum];
    CNeuron< InputNum + 1, NeuronNum1, Xfer0 > m_neuron_0[NeuronNum0];

    CInput<NeuronNum1> m_bias_1;
    Weight m_weight_bias_1[NeuronNum1];
    Weight m_weight_neuron_1[NeuronNum1][NeuronNum0];
    CNeuron< NeuronNum0 + 1, OutputNum, Xfer1 > m_neuron_1[NeuronNum1];

    CInput<OutputNum> m_bias_output;
    Weight m_weight_bias_output[OutputNum];
    Weight m_weight_output[OutputNum][NeuronNum1];
    CNeuron< NeuronNum1 + 1, 1, XferOutput > m_output[OutputNum];

    CTarget m_target[OutputNum];
};

} // namespace wwd

#endif // NN_2_LAYER_HPP_

