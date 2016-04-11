add_executable(fixed_se_dmp
    fixed_se_dmp.cpp
    )

set_target_properties(fixed_se_dmp 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )

target_link_libraries ( fixed_se_dmp ${LIBS} )

add_executable(parametric_se_dmp
    parametric_se_dmp.cpp
    )

set_target_properties(parametric_se_dmp 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )

target_link_libraries ( parametric_se_dmp ${LIBS} )


