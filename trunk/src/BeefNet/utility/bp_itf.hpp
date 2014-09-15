#ifndef BP_ITF_HPP_
#define BP_ITF_HPP_

#include "type.hpp"

namespace wwd
{

class IBP
{
public:

    inline double get_backward_input(void) const
    {
        return m_backward_input;
    }

    inline double get_backward_val(void) const
    {
        return m_backward_val;
    }

    inline double get_backward_output(void) const
    {
        return m_backward_output;
    }

protected:

    IBP( IN double val = 0.0 )
        : m_backward_input(0.0)
        , m_backward_val(val)
        , m_backward_output(0.0)
    {
    }

    ~IBP(void)
    {
    }

protected:

    double m_backward_input;
    double m_backward_val;
    double m_backward_output;
};

} // namespace wwd

#endif // BP_ITF_HPP_

