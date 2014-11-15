#ifndef BACKWARD_ITF_HPP_
#define BACKWARD_ITF_HPP_

#include "type.hpp"

namespace wwd
{

class IBackward
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

    IBackward(void)
        : m_input_val(0.0)
        , m_output_val(0.0)
    {
    }

    ~IBackward(void)
    {
    }

protected:

    double m_input_val;
    double m_output_val;
};

} // namespace wwd

#endif // BACKWARD_ITF_HPP_

