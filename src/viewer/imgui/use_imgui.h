#ifndef USE_IMGUI_H_
#define USE_IMGUI_H_

#include <array>

class GLFWwindow;


class UseImgui {
public:
    using RGBAColor = std::array<float, 4>;
    UseImgui() = default;

    void init();
    void show_window();
    void shutdown();
private:
    GLFWwindow* window;
    const char* gls_version;

    // Our state
    bool show_demo_window {false};
    //bool show_implot_demo_window { false };
    bool show_another_window {false};
    // Dummy parameters for the demo
    float f;
    int counter = 0;
    RGBAColor clear_color {0.45f, 0.55f, 0.60f, 1.00f};

    // The window (frame) I'm playing with
    bool show_my_implot {true};

    int x_min = -15;
    int x_max = 15;

    static constexpr std::size_t data_size = 5000;
    // This should be roughly proportional to the size
    // of the range of $F$ when rescaled so that the view
    // is a square (we want to take a nice, controlled portion
    // of the screen)
    float amplitude = 15;
    // Should be scaled linearly along with the amplitude
    float noise {1.5f};

    // Controls whether the noise slider is locked or unlocked
    bool do_noise{true};

};



#endif  // USE_IMGUI_H_
