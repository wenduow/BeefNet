#ifndef BACKWARD_ITF_HPP_
#define BACKWARD_ITF_HPP_

#include "type.hpp"

namespace wwd
{

class IBackward
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

    IBackward(void)
        : m_input(0.0)
        , m_output(0.0)
    {
    }

    ~IBackward(void)
    {
    }

protected:

    double m_input;
    double m_output;
};

} // namespace wwd

#endif // BACKWARD_ITF_HPP_

