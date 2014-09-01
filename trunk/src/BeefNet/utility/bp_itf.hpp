#ifndef BP_ITF_HPP_
#define FP_ITF_HPP_

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

    inline double get_backward_output(void) const
    {
        return m_backward_output;
    }

protected:

    IBP(void)
        : m_backward_input(0.0)
        , m_backward_output(0.0)
    {
    }

    ~IBP(void)
    {
    }

protected:

    double m_backward_input;
    double m_backward_output;
};

} // namespace wwd

#endif // FP_ITF_HPP_

