#include <fstream>
#include <iostream>
#include "type.hpp"

using namespace wwd;

int32 main( IN uint32 argc, IN const char *argv[] )
{
    if ( 4 != argc )
    {
        std::cerr << "Invalid parameter!" << std::endl;
        return -1;
    }

    std::ifstream config_in( argv[1], std::ios::binary );
    if ( config_in.fail() )
    {
        std::cerr << "Unable to open demo.hpp" << std::endl;
        return -1;
    }

    config_in.seekg( 0, std::ios::end );
    size_t len = config_in.tellg();
    config_in.seekg( 0, std::ios::beg );

    char *buf_config = new char[ len + 1 ];
    memset( buf_config, 0, len + 1 );
    config_in.read( buf_config, len );
    std::string str_config( buf_config, len );
    delete[] buf_config;
    buf_config = NULL;
    config_in.close();

    size_t pos_beg = str_config.find( argv[2], 0 ) + strlen( argv[2] ) + 3;
    size_t pos_end = str_config.find( ';', pos_beg + 1 );
    str_config.erase( pos_beg, pos_end - pos_beg );
    str_config.insert( pos_beg, argv[3] );

    std::ofstream config_out( argv[1], std::ios::binary );
    if ( config_out.fail() )
    {
        std::cerr << "Unable to open demo.hpp" << std::endl;
        return -1;
    }

    config_out.write( str_config.c_str(), str_config.length() );
    config_out.close();
    return 0;
}

