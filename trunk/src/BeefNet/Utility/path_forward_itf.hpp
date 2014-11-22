#ifndef PATH_FORWARD_ITF_HPP_
#define PATH_FORWARD_ITF_HPP_

#include "type.hpp"

namespace wwd
{

class IPathForward
{
public:

    inline void set_input_value( IN double val )
    {
        m_input_val = val;
    }

    inline double get_input_value(void) const
    {
        return m_input_val;
    }

    inline void set_output_value( IN double val )
    {
        m_output_val = val;
    }

    inline double get_output_value(void) const
    {
        return m_output_val;
    }

protected:

    IPathForward(void)
        : m_input_val(0.0)
        , m_output_val(0.0)
    {
    }

    ~IPathForward(void)
    {
    }

protected:

    double m_input_val;
    double m_output_val;
};

} // namespace wwd

#endif // PATH_FORWARD_ITF_HPP_

