#ifndef NN_1_LAYER_HPP_
#define NN_1_LAYER_HPP_

#include "nn_itf.hpp"

namespace wwd
{

template < class Weight,
           uint32 InputNum,
           uint32 HiddenNum, class Xfer,
           uint32 OutputNum, class XferOutput >
class CNN1Layer
    : public INN
{
public:

    enum
    {
        input_num  = InputNum,
        hidden_num = HiddenNum,
        output_num = OutputNum
    };

private:

    typedef CNN1Layer< Weight,
                       input_num,
                       hidden_num, Xfer,
                       output_num, XferOutput > ThisType;

public:

    CNN1Layer(void)
        : INN()
    {
        connect();
    }

    ~CNN1Layer(void)
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
        INN::forward( m_bias,
                      m_weight_bias,        m_weight_neuron,
                      m_neuron );
        INN::forward( m_bias_output,
                      m_weight_bias_output, m_weight_output,
                      m_output );
    }

    void backward(void)
    {
        INN::backward(m_target);
        INN::backward( m_weight_bias_output, m_weight_output, m_output );
        INN::backward( m_weight_bias,        m_weight_neuron, m_neuron );
    }

    void update(void)
    {
        INN::update( m_weight_bias,        m_weight_neuron );
        INN::update( m_weight_bias_output, m_weight_output );
    }

    double get_gradient_abs(void) const
    {
        return ( INN::get_gradient_abs_sum( m_weight_bias,
                                            m_weight_neuron )
               + INN::get_gradient_abs_sum( m_weight_bias_output,
                                            m_weight_output ) )
             / (double)( INN::get_weight_num( m_weight_bias,
                                              m_weight_neuron )
                       + INN::get_weight_num( m_weight_bias_output,
                                              m_weight_output ) );
    }

#if ( defined _DEBUG || defined PRINT_WEIGHT )

    void print(void) const
    {
        INN::print_input(m_input);
        INN::print( m_weight_bias,        m_weight_neuron, m_neuron );
        INN::print( m_weight_bias_output, m_weight_output, m_output );
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

    CNN1Layer( IN const ThisType &other );
    inline ThisType &operator=( IN const ThisType &other );

    void connect(void)
    {
        INN::connect( m_bias,        m_weight_bias,
                      m_input,       m_weight_neuron,
                      m_neuron );
        INN::connect( m_bias_output, m_weight_bias_output,
                      m_neuron,      m_weight_output,
                      m_output );
        INN::connect( m_output, m_target );
    }

    void map_to( OUT ThisType &other ) const
    {
        INN::map_to( other.m_weight_bias,        m_weight_bias,
                     other.m_weight_neuron,      m_weight_neuron );
        INN::map_to( other.m_weight_bias_output, m_weight_bias_output,
                     other.m_weight_output,      m_weight_output );
    }

    void reduce_from( INOUT ThisType &other )
    {
        INN::reduce_from( m_weight_bias,        other.m_weight_bias,
                          m_weight_neuron,      other.m_weight_neuron );
        INN::reduce_from( m_weight_bias_output, other.m_weight_bias_output,
                          m_weight_output,      other.m_weight_output );
    }

private:

    CInput<hidden_num> m_input[input_num];

    CInput<hidden_num> m_bias;
    Weight m_weight_bias[hidden_num];
    Weight m_weight_neuron[hidden_num][input_num];
    CNeuron< input_num + 1, output_num, Xfer > m_neuron[hidden_num];

    CInput<output_num> m_bias_output;
    Weight m_weight_bias_output[output_num];
    Weight m_weight_output[output_num][hidden_num];
    CNeuron< hidden_num + 1, 1, XferOutput > m_output[output_num];

    CTarget m_target[output_num];
};

} // namespace wwd

#endif // NN_1_LAYER_HPP_

