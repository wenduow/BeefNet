#ifndef FORWARD_ITF_HPP_
#define FORWARD_ITF_HPP_

#include "type.hpp"

namespace wwd
{

class IForward
{
public:

    inline double get_input_val(void) const
    {
        return m_input_val;
    }

    inline double get_output_val(void) const
    {
        return m_output_val;
    }

protected:

    IForward(void)
        : m_input_val(0.0)
        , m_output_val(0.0)
    {
    }

    ~IForward(void)
    {
    }

protected:

    double m_input_val;
    double m_output_val;
};

} // namespace wwd

#endif // FORWARD_ITF_HPP_

