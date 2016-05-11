add_executable(parametric_dmp_lwr
    parametric_dmp_lwr.cpp
    )

set_target_properties(parametric_dmp_lwr 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )

target_link_libraries ( parametric_dmp_lwr ${LIBS} )


add_executable(parametric_dmp_lwpr
    parametric_dmp_lwpr.cpp
    )

set_target_properties(parametric_dmp_lwpr 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )

target_link_libraries ( parametric_dmp_lwpr ${LIBS} )






