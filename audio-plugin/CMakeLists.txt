cmake_minimum_required(VERSION 3.15)

project(Regulator VERSION 1.0.0)

add_subdirectory(${JUCE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/juce)

juce_add_plugin(Regulator
  COMPANY_NAME "tadmn"
  PLUGIN_MANUFACTURER_CODE Tadn
  PLUGIN_CODE Rglt
  FORMATS VST3
  PRODUCT_NAME "Regulator")

target_sources(Regulator PRIVATE
  PluginProcessor.cpp
  PluginProcessor.h
  PluginEditor.cpp
  PluginEditor.h)

target_compile_definitions(Regulator PUBLIC JUCE_VST3_CAN_REPLACE_VST2=0)

# Automatically copy to user plugin folder to speed up development process
set(VST3_INSTALL_DIR "~/Library/Audio/Plug-Ins/VST3")
add_custom_command(TARGET Regulator POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "$<TARGET_FILE_DIR:Regulator>/VST3"
  "${VST3_INSTALL_DIR}"
  COMMENT "Copying VST3 to ${VST3_INSTALL_DIR}")