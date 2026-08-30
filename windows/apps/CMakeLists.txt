add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
if(WIN32)
  # D5: the Windows GUI is Ninfer.exe; the CLI keeps a distinct name on the case-insensitive FS.
  set_target_properties(ninfer PROPERTIES OUTPUT_NAME ninfer-cli)
endif()
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/apps
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/apps
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

if(WIN32)
  # Native Win32 GUI, double-clickable as Ninfer.exe (the CLI is ninfer-cli.exe).
  add_executable(ninfer-gui gui/main.cpp)
  set_target_properties(ninfer-gui PROPERTIES OUTPUT_NAME Ninfer WIN32_EXECUTABLE ON)
  # D3/D4/D5: header-only nlohmann for the /health JSON; winhttp for the
  # self-contained local HTTP GET (the only new system library).
  target_include_directories(ninfer-gui PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
  target_link_libraries(ninfer-gui PRIVATE winhttp)
endif()
