#define UNICODE 1
#define _UNICODE 1
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <winhttp.h>
#include <string>
#include <vector>

#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "comctl32.lib")

namespace {
constexpr wchar_t kWindowClass[]  = L"NinferGui";
constexpr wchar_t kWindowTitle[]  = L"Wallawalla Launcher";

void set_status(HWND /*hwnd*/, const wchar_t* /*text*/) {
    // Filled in by Task 6 (a status STATIC control). No-op for now.
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_COMMAND:
        if (LOWORD(wParam) == IDCANCEL) { ::PostQuitMessage(0); }
        return 0;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hwnd, msg, wParam, lParam);
}

void create_scaffold(HWND hwnd) {
    // Filled in by Task 6 (Core controls). No-op for now.
    (void)hwnd;
}
} // namespace

int WINAPI wWinMain(HINSTANCE hinst, HINSTANCE, LPWSTR, int nCmdShow) {
    INITCOMMONCONTROLSEX icc { sizeof(icc), ICC_STANDARD_CLASSES };
    ::InitCommonControlsEx(&icc);

    WNDCLASSEXW wc { sizeof(wc) };
    wc.lpfnWndProc   = WndProc;
    wc.hInstance      = hinst;
    wc.hCursor        = ::LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground  = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName  = kWindowClass;
    ::RegisterClassExW(&wc);

    HWND hwnd = ::CreateWindowExW(0, kWindowClass, kWindowTitle, WS_OVERLAPPEDWINDOW,
                                  CW_USEDEFAULT, CW_USEDEFAULT, 760, 940,
                                  nullptr, nullptr, hinst, nullptr);
    if (!hwnd) { return 1; }
    create_scaffold(hwnd);
    ::ShowWindow(hwnd, nCmdShow);
    ::UpdateWindow(hwnd);

    MSG msg;
    for (;;) {
        if (::GetMessageW(&msg, nullptr, 0, 0) > 0) {
            ::TranslateMessage(&msg);
            ::DispatchMessageW(&msg);
        } else {
            break;
        }
    }
    return static_cast<int>(msg.wParam);
}
