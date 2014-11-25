#ifndef PARAM_LM_HPP_
#define PARAM_LM_HPP_

#include "../../Utility/type.hpp"

namespace wwd
{

template < uint32 PatternNum,
           uint32 OutputNum,
           uint32 Lambda = 10,
           uint32 Beta = 10 >
class EParamLM
{
private:

    EParamLM(void);
    ~EParamLM(void);

public:

    enum
    {
        pattern_num = PatternNum,
        output_num = OutputNum
    };

    static double lambda;
    static const double beta;
};

template < uint32 PatternNum,
           uint32 OutputNum,
           uint32 Lambda /** = 10 */,
           uint32 Beta /** = 10 */ >
double EParamLM< PatternNum, OutputNum, Lambda, Beta >::lambda
    = (double)Lambda / 1000.0;

template < uint32 PatternNum,
           uint32 OutputNum,
           uint32 Lambda /** = 10 */,
           uint32 Beta /** = 10 */ >
const double EParamLM< PatternNum, OutputNum, Lambda, Beta >::beta
    = (double)Beta;

} // namespace wwd

#endif // PARAM_LM_HPP_

