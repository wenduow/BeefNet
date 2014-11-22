#ifndef PARAM_RP_HPP_
#define PARAM_RP_HPP_

#include "../../Utility/type.hpp"

namespace wwd
{

template < uint32 FactDec = 500, uint32 FactInc = 1200, uint32 UpdateInit = 100 >
class EParamRP
{
private:

    EParamRP(void);
    ~EParamRP(void);

public:

    static const double fact_dec;
    static const double fact_inc;
    static const double update_init;
};

template < uint32 FactDec /** = 50 */,
           uint32 FactInc /** = 120 */,
           uint32 UpdateInit /** = 100 */ >
const double EParamRP< FactDec, FactInc, UpdateInit >::fact_dec
    = (double)FactDec / 1000.0;

template < uint32 FactDec /** = 50 */,
           uint32 FactInc /** = 120 */,
           uint32 UpdateInit /** = 100 */ >
const double EParamRP< FactDec, FactInc, UpdateInit >::fact_inc
    = (double)FactInc / 1000.0;

template < uint32 FactDec /** = 50 */,
           uint32 FactInc /** = 120 */,
           uint32 UpdateInit /** = 100 */ >
const double EParamRP< FactDec, FactInc, UpdateInit >::update_init
    = (double)UpdateInit / 1000.0;

} // namespace wwd

#endif // PARAM_RP_HPP_

