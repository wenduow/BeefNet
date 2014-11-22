#ifndef PATH_BACKWARD_HPP_
#define PATH_BACKWARD_HPP_

#include "type.hpp"

namespace wwd
{

class IPathBackward
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

    IPathBackward(void)
        : m_input_val(0.0)
        , m_output_val(0.0)
    {
    }

    ~IPathBackward(void)
    {
    }

protected:

    double m_input_val;
    double m_output_val;
};

} // namespace wwd

#endif // PATH_BACKWARD_HPP_

