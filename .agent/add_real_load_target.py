import io

path = "core/tests/CMakeLists.txt"
with io.open(path, "r", encoding="utf-8") as fh:
    text = fh.read()

anchor = '    RUNTIME_OUTPUT_DIRECTORY "${FN_PROJ_ROOT}/build/tests")\nendif()'
block = anchor + '\n\n' + '''# P12 S1 (b) / P1V gate driver: real-geometry .ninner load + T==1 decode gates.
if(EXISTS "${FN_PROJ_ROOT}/tools/flash_next_dev/real_load_driver.cpp")
  add_executable(flash_next_real_load
    "${FN_PROJ_ROOT}/tools/flash_next_dev/real_load_driver.cpp")
  target_link_libraries(flash_next_real_load PRIVATE ninfer_engine ninfer_core)
  target_include_directories(flash_next_real_load PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_flash_next/export)
  target_compile_definitions(flash_next_real_load PRIVATE
    NINFER_SOURCE_DIR="${FN_PROJ_ROOT}")
  if(TARGET CUDA::cudart)
    target_link_libraries(flash_next_real_load PRIVATE CUDA::cudart)
  endif()
  set_target_properties(flash_next_real_load PROPERTIES
    OUTPUT_NAME flash_next_real_load
    RUNTIME_OUTPUT_DIRECTORY "${FN_PROJ_ROOT}/build/tests")
endif()'''

if "flash_next_real_load" in text:
    print("ALREADY PRESENT")
elif text.count(anchor) != 1:
    print("ANCHOR COUNT:", text.count(anchor))
else:
    text = text.replace(anchor, block)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print("INSERTED")
