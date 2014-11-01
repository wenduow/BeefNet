#ifndef NN_RECURRENT_HPP_
#define NN_RECURRENT_HPP_

#include "nn_itf.hpp"

namespace wwd
{

template < class Weight,
           uint32 InputNum,
           uint32 ForwardNum,  class XferForward,
           uint32 BackwardNum, class XferBackward,
           uint32 OutputNum,   class XferOutput >
class CNNRecurrent
    : public INN
{
public:

    enum
    {
        input_num    = InputNum,
        forward_num  = ForwardNum,
        backward_num = BackwardNum,
        output_num   = OutputNum
    };

private:

    typedef CNNRecurrent< Weight,
                          input_num,
                          forward_num,  XferForward,
                          backward_num, XferBackward ,
                          output_num,   XferOutput > ThisType;

public:

    CNNRecurrent(void)
        : INN()
    {
        connect();
    }

    ~CNNRecurrent(void)
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
        INN::forward( m_bias_forward,
                      m_weight_bias_forward,  m_weight_forward,
                      m_forward );
        INN::forward( m_bias_backward,
                      m_weight_bias_backward, m_weight_backward,
                      m_backward );
        INN::forward(m_weight_feedback);
        INN::forward( m_bias_output,
                      m_weight_bias_output,   m_weight_output,
                      m_output );
    }

    void backward(void)
    {
        INN::backward(m_target);
        INN::backward( m_weight_bias_output,   m_weight_output, m_output );
        INN::backward(m_weight_feedback);
        INN::backward( m_weight_bias_backward, m_weight_backward, m_backward );
        INN::backward( m_weight_bias_forward,  m_weight_forward, m_forward );
    }

    void update(void)
    {
        INN::update( m_weight_bias_forward,  m_weight_forward );
        INN::update( m_weight_bias_backward, m_weight_backward );
        INN::update(m_weight_feedback);
        INN::update( m_weight_bias_output,   m_weight_output );
    }

    double get_gradient_abs(void) const
    {
        return ( INN::get_gradient_abs_sum( m_weight_bias_forward,
                                            m_weight_forward )
               + INN::get_gradient_abs_sum( m_weight_bias_backward,
                                            m_weight_backward )
               + INN::get_gradient_abs_sum(m_weight_feedback)
               + INN::get_gradient_abs_sum( m_weight_bias_output,
                                            m_weight_output ) )
             / (double)( INN::get_weight_num( m_weight_bias_forward,
                                              m_weight_forward )
                       + INN::get_weight_num( m_weight_bias_backward,
                                              m_weight_backward )
                       + INN::get_weight_num(m_weight_feedback)
                       + INN::get_weight_num( m_weight_bias_output,
                                              m_weight_output ) );
    }

#if ( defined _DEBUG || defined PRINT_WEIGHT )

    void print(void) const
    {
        INN::print_input(m_input);
        INN::print( m_weight_bias_forward,  m_weight_forward, m_forward );
        INN::print( m_weight_bias_backward, m_weight_backward, m_backward );
        INN::print( m_weight_feedback );
        INN::print( m_weight_bias_output,   m_weight_output, m_output );
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

    CNNRecurrent( IN const ThisType &other );
    inline ThisType &operator=( IN const ThisType &other );

    void connect(void)
    {
        INN::connect( m_bias_forward,  m_weight_bias_forward,
                      m_input,         m_weight_forward,
                      m_forward );
        INN::connect( m_bias_backward, m_weight_bias_backward,
                      m_forward,       m_weight_backward,
                      m_backward );
        INN::connect( m_backward,      m_weight_feedback, m_forward );
        INN::connect( m_bias_output,   m_weight_bias_output,
                      m_forward,       m_weight_output,
                      m_output );
        INN::connect( m_output, m_target );
    }

    void map_to( OUT ThisType &other ) const
    {
        INN::map_to( other.m_weight_bias_forward,  m_weight_bias_forward,
                     other.m_weight_forward,       m_weight_forward );
        INN::map_to( other.m_weight_bias_backward, m_weight_bias_backward,
                     other.m_weight_backward,      m_weight_backward );
        INN::map_to( other.m_weight_feedback,      m_weight_feedback );
        INN::map_to( other.m_weight_bias_output,   m_weight_bias_output,
                     other.m_weight_output,        m_weight_output );
    }

    void reduce_from( INOUT ThisType &other )
    {
        INN::reduce_from( m_weight_bias_forward,  other.m_weight_bias_forward,
                          m_weight_forward,       other.m_weight_forward );
        INN::reduce_from( m_weight_bias_backward, other.m_weight_bias_backward,
                          m_weight_backward,      other.m_weight_backward );
        INN::reduce_from( m_weight_feedback,      other.m_weight_feedback );
        INN::reduce_from( m_weight_bias_output,   other.m_weight_bias_output,
                          m_weight_output,        other.m_weight_output );
    }

private:

    CInput<forward_num> m_input[input_num];

    CInput<forward_num> m_bias_forward;
    Weight m_weight_bias_forward[forward_num];
    Weight m_weight_forward[forward_num][input_num];
    CNeuron< input_num + backward_num + 1,
             output_num + backward_num,
             XferForward > m_forward[forward_num];

    CInput<output_num> m_bias_output;
    Weight m_weight_bias_output[output_num];
    Weight m_weight_output[output_num][forward_num];
    CNeuron< forward_num + 1, 1, XferOutput > m_output[output_num];

    CTarget m_target[output_num];

    CInput<backward_num> m_bias_backward;
    Weight m_weight_bias_backward[backward_num];
    Weight m_weight_backward[backward_num][forward_num];
    CNeuron< forward_num + 1, forward_num, XferOutput > m_backward[backward_num];

    Weight m_weight_feedback[forward_num][backward_num];
};

} // namespace wwd

#endif // NN_RECURRENT_HPP_

