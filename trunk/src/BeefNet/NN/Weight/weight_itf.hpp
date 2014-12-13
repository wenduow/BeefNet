#ifndef WEIGHT_ITF_HPP_
#define WEIGHT_ITF_HPP_

#include <cstdlib>
#include "../../Utility/node_input_itf.hpp"
#include "../../Utility/node_output_itf.hpp"

namespace wwd
{

class IWeight
    : public INodeInput<1>
    , public INodeOutput<1>
{
public:

    inline const IWeight &operator>>( OUT IWeight &other ) const
    {
        other.m_weight = m_weight;
        other.m_gradient_sum = 0.0;
        other.m_pattern_num = 0;

        return *this;
    }

    inline IWeight &operator<<( IN const IWeight &other )
    {
        m_gradient_sum += other.m_gradient_sum;
        m_pattern_num += other.m_pattern_num;

        return *this;
    }

    friend std::istream &operator>>( INOUT std::istream &is,
                                     OUT IWeight &rhs )
    {
        double gradient;
        uint32 pattern;
        char delimeter;
        is >> gradient >> delimeter >> pattern >> delimeter;
        
        rhs.m_gradient_sum += gradient;
        rhs.m_pattern_num += pattern;

        return is;
    }

    friend std::ostream &operator<<( INOUT std::ostream &os,
                                     IN const IWeight &rhs )
    {
        os << rhs.m_gradient_sum << '#' << rhs.m_pattern_num << '#';

        return os;
    }

    inline void forward(void)
    {
        IPathForward::m_input_val = m_input_node[0]->get_output_value();
        IPathForward::m_output_val = m_weight * IPathForward::m_input_val;
    }

    inline void backward(void)
    {
        IPathBackward::m_input_val = m_output_node[0]->get_output_value();
        IPathBackward::m_output_val = m_weight * IPathBackward::m_input_val;

        // gradient = dE / dWi = - Sum( delta * Xi ),
        // where Sum is through all input samples.
        // IForward::m_input_val is the f'(net) * sum(delta) from next layer.
        m_gradient_sum -= IPathBackward::m_input_val
                        * IPathForward::m_input_val;

        ++m_pattern_num;
    }

    inline void update( IN double delta_weight )
    {
        m_weight += delta_weight;
        m_gradient_sum = 0.0;
        m_pattern_num = 0;
    }

    inline double get_gradient_sum(void) const
    {
        return m_gradient_sum;
    }

    inline uint32 get_gradient_num(void) const
    {
        return m_pattern_num;
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        stream.write( (const char*)&m_weight, sizeof(m_weight) );
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        stream.read( (char*)&m_weight, sizeof(m_weight) );
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        std::cout << m_weight << '\t';
    }
#endif // _DEBUG

protected:

    IWeight(void)
        : INodeInput()
        , INodeOutput()
        , m_weight( (double)rand() / (double)RAND_MAX * 1.4 - 0.7 )
        , m_gradient_sum(0.0)
        , m_pattern_num(0)
    {
    }

    ~IWeight(void)
    {
    }

protected:

    double m_weight;
    double m_gradient_sum;
    uint32 m_pattern_num;
};

} // namespace wwd

#endif // WEIGHT_ITF_HPP_

