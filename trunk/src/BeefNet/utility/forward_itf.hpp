#ifndef FORWARD_ITF_HPP_
#define FORWARD_ITF_HPP_

#include "type.hpp"

namespace wwd
{

class IForward
{
public:

    inline double get_input(void) const
    {
        return m_input;
    }

    inline double get_output(void) const
    {
        return m_output;
    }

protected:

    IForward(void)
        : m_input(0.0)
        , m_output(0.0)
    {
    }

    ~IForward(void)
    {
    }

protected:

    double m_input;
    double m_output;
};

} // namespace wwd

#endif // FORWARD_ITF_HPP_

