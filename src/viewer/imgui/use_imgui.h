#ifndef USE_IMGUI_H_
#define USE_IMGUI_H_

#include <array>

class GLFWwindow;


class UseImgui {
public:
    using RGBAColor = std::array<float, 4>;

    void init();
    void show_demo();
    void show_implot_demo();
    void shutdown();
private:
    GLFWwindow* window;
    const char* glsl_version;

    // Our state
    bool show_demo_window {true};
    bool show_implot_demo_window { false };
    bool show_another_window {false};
    float f = 0.0f;
    int counter = 0;
    RGBAColor clear_color {0.45f, 0.55f, 0.60f, 1.00f};
};



#endif  // USE_IMGUI_H_
