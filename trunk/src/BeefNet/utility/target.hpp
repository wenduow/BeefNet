#ifndef TARGET_HPP_
#define TARGET_HPP_

#include "output_itf.hpp"

namespace wwd
{

class CTarget
    : public IOutput<1>
{
private:

    typedef CTarget ThisType;

public:

    CTarget(void)
        : IOutput<1>()
    {
    }

    ~CTarget(void)
    {
    }

    inline ThisType &operator=( IN double target )
    {
        m_backward_input = target;
        return *this;
    }

    inline void backward(void)
    {
        m_backward_output = m_backward_input
                          - m_input[0]->get_forward_output();
    }
};

} // namespace wwd

#endif // TARGET_HPP_

