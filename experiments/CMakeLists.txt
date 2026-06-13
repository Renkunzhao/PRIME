file(GLOB _CONTACT_ID_EXPERIMENT_MODULES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_module ${_CONTACT_ID_EXPERIMENT_MODULES})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_module}
     AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_module}/CMakeLists.txt)
    add_subdirectory(${_module})
  endif()
endforeach()
