#ifndef NN_3_LAYER_HPP_
#define NN_3_LAYER_HPP_

#include "nn_itf.hpp"

namespace wwd
{

template < class Weight,
           uint32 InputNum,
           uint32 HiddenNum0, class Xfer0,
           uint32 HiddenNum1, class Xfer1,
           uint32 HiddenNum2, class Xfer2,
           uint32 OutputNum,  class XferOutput >
class CNN3Layer
    : public INN
{
public:

    enum
    {
        input_num    = InputNum,
        hidden_num_0 = HiddenNum0,
        hidden_num_1 = HiddenNum1,
        hidden_num_2 = HiddenNum2,
        output_num   = OutputNum
    };

private:

    typedef CNN3Layer< Weight,
                       input_num,
                       hidden_num_0, Xfer0,
                       hidden_num_1, Xfer1,
                       hidden_num_2, Xfer2,
                       output_num,   XferOutput > ThisType;

public:

    CNN3Layer(void)
        : INN()
    {
        connect();
    }

    ~CNN3Layer(void)
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

    void get_output( OUT double (&output)[output_num] ) const
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
        INN::forward( m_bias_2,
                      m_weight_bias_2,      m_weight_neuron_2,
                      m_neuron_2 );
        INN::forward( m_bias_output,
                      m_weight_bias_output, m_weight_output,
                      m_output );
    }

    void backward(void)
    {
        INN::backward(m_target);
        INN::backward( m_weight_bias_output, m_weight_output,   m_output );
        INN::backward( m_weight_bias_2,      m_weight_neuron_2, m_neuron_2 );
        INN::backward( m_weight_bias_1,      m_weight_neuron_1, m_neuron_1 );
        INN::backward( m_weight_bias_0,      m_weight_neuron_0, m_neuron_0 );
    }

    void update(void)
    {
        INN::update( m_weight_bias_0,      m_weight_neuron_0 );
        INN::update( m_weight_bias_1,      m_weight_neuron_1 );
        INN::update( m_weight_bias_2,      m_weight_neuron_2 );
        INN::update( m_weight_bias_output, m_weight_output );
    }

    double get_gradient_abs(void) const
    {
        return ( INN::get_gradient_abs_sum( m_weight_bias_0,
                                            m_weight_neuron_0 )
               + INN::get_gradient_abs_sum( m_weight_bias_1,
                                            m_weight_neuron_1 )
               + INN::get_gradient_abs_sum( m_weight_bias_2,
                                            m_weight_neuron_2 )
               + INN::get_gradient_abs_sum( m_weight_bias_output,
                                            m_weight_output ) )
             / (double)( INN::get_weight_num( m_weight_bias_0,
                                              m_weight_neuron_0 )
                       + INN::get_weight_num( m_weight_bias_1,
                                              m_weight_neuron_1 )
                       + INN::get_weight_num( m_weight_bias_2,
                                              m_weight_neuron_2 )
                       + INN::get_weight_num( m_weight_bias_output,
                                              m_weight_output ) );
    }

#if ( defined _DEBUG || defined PRINT_WEIGHT )

    void print(void) const
    {
        INN::print_input(m_input);
        INN::print( m_weight_bias_0,      m_weight_neuron_0, m_neuron_0 );
        INN::print( m_weight_bias_1,      m_weight_neuron_1, m_neuron_1 );
        INN::print( m_weight_bias_2,      m_weight_neuron_2, m_neuron_2 );
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

    CNN3Layer( IN const ThisType &other );
    inline ThisType &operator=( IN const ThisType &other );

    void connect(void)
    {
        INN::connect( m_bias_0,      m_weight_bias_0,
                      m_input,       m_weight_neuron_0,
                      m_neuron_0 );
        INN::connect( m_bias_1,      m_weight_bias_1,
                      m_neuron_0,    m_weight_neuron_1,
                      m_neuron_1 );
        INN::connect( m_bias_2,      m_weight_bias_2,
                      m_neuron_1,    m_weight_neuron_2,
                      m_neuron_2 );
        INN::connect( m_bias_output, m_weight_bias_output,
                      m_neuron_2,    m_weight_output,
                      m_output );
        INN::connect( m_output, m_target );
    }

    void map_to( OUT ThisType &other ) const
    {
        INN::map_to( other.m_weight_bias_0,      m_weight_bias_0,      
                     other.m_weight_neuron_0,    m_weight_neuron_0 );
        INN::map_to( other.m_weight_bias_1,      m_weight_bias_1,      
                     other.m_weight_neuron_1,    m_weight_neuron_1 );
        INN::map_to( other.m_weight_bias_2,      m_weight_bias_2,      
                     other.m_weight_neuron_2,    m_weight_neuron_2 );
        INN::map_to( other.m_weight_bias_output, m_weight_bias_output,
                     other.m_weight_output,      m_weight_output );
    }

    void reduce_from( IN const ThisType &other )
    {
        INN::reduce_from( m_weight_bias_0,      other.m_weight_bias_0,
                          m_weight_neuron_0,    other.m_weight_neuron_0 );
        INN::reduce_from( m_weight_bias_1,      other.m_weight_bias_1,
                          m_weight_neuron_1,    other.m_weight_neuron_1 );
        INN::reduce_from( m_weight_bias_2,      other.m_weight_bias_2,
                          m_weight_neuron_2,    other.m_weight_neuron_2 );
        INN::reduce_from( m_weight_bias_output, other.m_weight_bias_output,
                          m_weight_output,      other.m_weight_output );
    }

private:

    CInput<hidden_num_0> m_input[input_num];

    CInput<hidden_num_0> m_bias_0;
    Weight m_weight_bias_0[hidden_num_0];
    Weight m_weight_neuron_0[hidden_num_0][input_num];
    CNeuron< input_num + 1, hidden_num_1, Xfer0 > m_neuron_0[hidden_num_0];

    CInput<hidden_num_1> m_bias_1;
    Weight m_weight_bias_1[hidden_num_1];
    Weight m_weight_neuron_1[hidden_num_1][hidden_num_0];
    CNeuron< hidden_num_0 + 1, hidden_num_2, Xfer1 > m_neuron_1[hidden_num_1];

    CInput<hidden_num_2> m_bias_2;
    Weight m_weight_bias_2[hidden_num_2];
    Weight m_weight_neuron_2[hidden_num_2][hidden_num_1];
    CNeuron< hidden_num_1 + 1, output_num, Xfer2 > m_neuron_2[hidden_num_2];

    CInput<output_num> m_bias_output;
    Weight m_weight_bias_output[output_num];
    Weight m_weight_output[output_num][hidden_num_2];
    CNeuron< hidden_num_2 + 1, 1, XferOutput > m_output[output_num];

    CTarget m_target[output_num];
};

} // namespace wwd

#endif // NN_3_LAYER_HPP_

