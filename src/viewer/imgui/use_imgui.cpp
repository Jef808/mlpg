#include "use_imgui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "implot.h"
#include "implot_internal.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>


#ifndef PI
#define PI 3.14159265358979323846
#endif


void UseImgui::init() {
    // Setup window
    if (!glfwInit()) {
        std::cerr << "Failed to initialize glfw" << std::endl;
        return;
    }

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create window" << std::endl;
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    // ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void UseImgui::show_window() {

    // Objects to help with random queries with global lifetime
    std::random_device rd;
    std::mt19937 eng{ rd() };

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to
        // use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or
        // clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main
        // application, or clear/overwrite your copy of the keyboard data. Generally you may always pass all
        // inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse
        // its code to learn more about Dear ImGui!).
        if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

        if (show_my_implot)
        {
            // The data, recomputed every loop so we get ghostly graphs with the code noise.
            // Really amazing way to vizually depicts uncertainty!
            int bar_data[11] = {0, 1, 2, 5, 5, 6, 6, 7, 8, 9, 10};
            float x_data[data_size];
            float y_data[data_size];

            {
                // Start with the ordinary cosine from -2pi to 2pi then rescale
                double step = (x_max - x_min) / static_cast<double>( data_size );
                x_data[0] = x_min;
                y_data[0] = 0;
                const double scale = 4 * PI / (x_max - x_min);
                const double trans = - (x_max - x_min) / 2.0;
                const double scale_inv = (x_max - x_min) / (4 * PI);

                for (size_t i = 1; i < data_size - 1; ++i)
                {
                    x_data[i] = x_data[i-1] + step;
                    y_data[i] = scale_inv * std::cos( scale * (x_data[i] + trans) );
                }
            }

            float last_amplitude = amplitude;
            float max_amplitude = 80.0f;
            float max_noise = 40.0f;

            ImGui::Begin("My plots");
            // Start/stop the noise with a checkbox
            ImGui::Checkbox(do_noise ? "Remove noise":  "Add noise", &do_noise);
            // Modify the amplitude with a slider

            ImGui::SliderFloat("Control the amplitude with this slider", &amplitude, 2.0 * (noise), 5.0 * (x_max - x_min), "%.2f", ImGuiSliderFlags_AlwaysClamp);
            if (do_noise)
            {
                // NOTE: Try the more restrictive flags too
                ImGui::SliderFloat("Noise Control", &noise, 0.0f, (), "%.3f", ImGuiSliderFlags_AlwaysClamp & ImGuiSliderFlags_Logarithmic);//ImGuiSliderFlags_AlwaysClamp);

                std::normal_distribution<float> dist(0, noise);
                for (float & y : y_data) {
                    y += dist(eng);
                }
            }

            if (amplitude != last_amplitude)
            {
                double rescaling = amplitude / last_amplitude;
                for (float & y : y_data) {
                    y *= rescaling;
                }
                last_amplitude = amplitude;
            }

            ImGui::Text("Amplitude: %.2f", amplitude);
            ImGui::Text("Noise mean: %.2f, Noise standard deviation: %.2f: ", 0.0f, noise);
            ImGui::Text("Number of data points: %lu", data_size);

            if (ImPlot::BeginPlot("Noisy Cosine"))
            {
                //ImPlot::PlotBars("My Bar Plot", bar_data, 11);
                ImPlot::PlotScatter("My Scatter Plot", x_data, y_data, data_size);

                // etc...
                ImPlot::EndPlot();
            }

            ImGui::End();
        }

        // if (show_implot_demo_window)
        //     UseImgui::show_implot_demo();

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named
        // window.
        {
            ImGui::Begin("Hello, world!");  // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");  // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window",
                            &show_demo_window);  // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color);  // Edit 3 floats representing a color

            if (ImGui::Button("Button"))  // Buttons return true when clicked (most widgets return true when
                                          // edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window) {
            ImGui::Begin("Another Window",
                         &show_another_window);  // Pass a pointer to our bool variable (the window will have
                                                 // a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me")) show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color[0] * clear_color[3], clear_color[1] * clear_color[3],
                     clear_color[2] * clear_color[3], clear_color[3]);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
}

void UseImgui::shutdown() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::CreateContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
