#ifndef FP_ITF_HPP_
#define FP_ITF_HPP_

#include "type.hpp"

namespace wwd
{

class IFP
{
public:

    inline double get_forward_input(void) const
    {
        return m_forward_input;
    }

    inline double get_forward_val(void) const
    {
        return m_forward_val;
    }

    inline double get_forward_output(void) const
    {
        return m_forward_output;
    }

protected:

    IFP( IN double val = 0.0 )
        : m_forward_input(0.0)
        , m_forward_val(val)
        , m_forward_output(0.0)
    {
    }

    ~IFP(void)
    {
    }

protected:

    double m_forward_input;
    double m_forward_val;
    double m_forward_output;
};

} // namespace wwd

#endif // FP_ITF_HPP_

