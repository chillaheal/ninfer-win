@echo off
rem P10 build (Tokenizer, template, serve mini). VsDevCmd x64 env + pinned cmake/ninja.
rem Pattern: p9_build.bat. Builds the engine (all flash_next runtime sources +
rem engine.cpp + registry.cpp wiring) and the serve (relinked against the current
rem engine so the mini flash_next route is live). Also builds the product schema
rem test targets so the P10 "schema tests still green" check is one build. The
rem P10 serve-smoke test (flash_next_p10_serve_smoke_pytest) is a python ctest
rem entry registered in tests/CMakeLists.txt (no separate build target).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin;C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cmake --build "C:\Users\Micke\Documents\Ninfer\ninfer-win\build" --target ninfer_engine ninfer-serve ninfer_openai_schema_test ninfer_anthropic_schema_test ninfer_responses_schema_test ninfer_tool_call_parser_test ninfer_request_log_test ninfer_response_store_test -j8
