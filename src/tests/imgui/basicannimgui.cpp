#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <chrono>
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "implot.h"
#include "implot_internal.h"

#include <iostream>
#include <numeric>
#include <thread>

#include <Eigen/Core>

#include "ann/Config.h"
#include "ann/TemplateAnn.h"
#include "data/filesystem.h"
#include "data/load_csv.h"
#include "data/manip.h"


using FT = float;
using Index = Eigen::Index;
using Matrix = Eigen::MatrixXf;

using namespace simple;

// utility structure for plotting
struct PlotBuffer {
  int MaxSize;
  int Offset;
  ImVector<ImVec2> Data;
  PlotBuffer(int max_size)
    : MaxSize { max_size }, Offset{ 0 }
  {
        Data.reserve(MaxSize);
  }
  void AddPoint(float x, float y) {
    if (Data.size() < MaxSize)
      Data.push_back(ImVec2(x,y));
    else {
      Data[Offset] = ImVec2(x,y);
      Offset =  (Offset + 1) % MaxSize;
    }
  }
  void Erase() {
    if (Data.size() > 0) {
      Data.shrink(0);
      Offset  = 0;
    }
  }
};


class PushDataCb {
  public:
    explicit PushDataCb(PlotBuffer& ref) : buffer{ ref } {}
    template<typename T>
    void operator()(T d) { buffer.AddPoint(counter++, static_cast<float>(d)); }
  private:
    float counter = 0;
    PlotBuffer& buffer;
};

class OnExitCb {
  public:
    explicit OnExitCb(bool& ref) : flag{ ref } {}
    void notify() { flag = !flag; }
  private:
    bool& flag;
};


static bool p_open = true;


int main(int argc, char *argv[]) {

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  auto window = glfwCreateWindow(1200, 900, "Example", nullptr, nullptr);
  if (!window) {
    std::cerr << "Error creating glfw window" << std::endl;
    return EXIT_FAILURE;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  if (!gladLoaderLoadGL()) {
    std::cerr << "Error initializing glad";
    return EXIT_FAILURE;
  }

  // Resolve symlinks and get actual path to the data
  std::error_code ec;
  auto [train_fp, test_fp] = Data::get_train_test_fp("data/mnist", ec);
  if (ec) {
    std::cerr << "Failed to resolve the path to the data directory: " << ec.message() << std::endl;
    return EXIT_FAILURE;
  }

  // Set up network's configuration
  size_t n_epochs = 35;

  Config config {
    .InputSize = 784,
    .OutputSize = 10,
    .HiddenLayers = { 32 },

    .n_data = 18000,
    .batch_size = 32,

    .LearningRate = 0.1,
    .L2RegCoeff = 2.0,
  };

  // Load data with checks
  std::vector<FT> data_x;
  std::vector<FT> data_y;
  data_x.reserve(config.n_data * config.InputSize);
  data_y.reserve(config.n_data * config.OutputSize);
  auto err = Data::load_csv(train_fp, data_x, data_y,
                            config.OutputSize, config.n_data);
  if (err) {
    std::cerr << "Failed to load csv files: " << err->what() << std::endl;
    return EXIT_FAILURE;
  }
  if ((data_x.size() != config.n_data * config.InputSize)
      || (data_y.size() != config.n_data * config.OutputSize)) {
    std::cerr << "Incorrect size of data collected: "
              << " data_x.size() = " << data_x.size()
              << " and data_y.size() = " << data_y.size() << std::endl;
    return EXIT_FAILURE;
  }

  // Normalize the data
  std::transform(data_x.begin(), data_x.end(), data_x.begin(), [](auto x) { return x / 255.0f; });

  // Helpful quantities we will to use (Make n_data a multiple of batch_size)
  // Data is divided as training ratio * n_batch batches for training,
  // then what's left is split equally into validation and testing data.
  size_t batch_size = config.batch_size;
  ptrdiff_t n_batch = std::floor(config.n_data / batch_size);
  ptrdiff_t n_batch_train = std::ceil(static_cast<double>(n_batch) * config.training_ratio);
  ptrdiff_t n_batch_validation = std::ceil(static_cast<double>(n_batch - n_batch_train) / 2);
  ptrdiff_t n_batch_test = n_batch - n_batch_train - n_batch_validation;

  // Callbacks we pass to the network's Train method
  bool training_done = false;
  int n_batch_processed = 0;
  PlotBuffer LossDataBuffer { static_cast<int>(n_batch_train * n_epochs) };
  PlotBuffer AccuracyDataBuffer { static_cast<int>(n_batch_train * n_epochs) };

  // Start training in new thread
  std::thread in_training([&]{
    ANN<float> NN;
    NN.setup( config );
    NN.Train(n_epochs,
             data_x.data(), data_y.data(), n_batch_train,
             PushDataCb{ LossDataBuffer }, PushDataCb{ AccuracyDataBuffer },
             OnExitCb{ training_done });
  });
  in_training.detach();

  // Initialize ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450 core");
  ImGui::StyleColorsClassic();

  // Flag controlling visibility of the plot frames
  bool p_open = n_batch_processed > 0;

  double x_min = 0.0;
  double x_max = n_batch_processed;
  double y_min_loss = 0.0;
  double y_max_loss = 2.0;
  double y_min_accuracy = 0.0 - 0.2;
  double y_max_accuracy = 1.0 + 0.2;

  int size_accuracy = 0;
  int size_loss = 0;
  float b = 0;
  float history = 100;

  ImPlotAxisFlags x_flags = ImPlotAxisFlags_None;
  ImPlotAxisFlags y_flags_loss = ImPlotAxisFlags_LogScale;

  // Main display loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Freeze the sizes since it is controlled in another thread
    size_loss = LossDataBuffer.Data.size();
    size_accuracy = AccuracyDataBuffer.Data.size();

    if (size_loss > 0 && size_accuracy > 0)
    {
      if (ImGui::Begin("Monitoring network", &p_open))
      {
        if (not training_done) {
          ImGui::Text("Training in progress");
        } else {
          ImGui::Text("Training Completed");
        }
        if (ImPlot::BeginSubplots("##Training Data", 2, 1, ImVec2(800, 800), ImPlotSubplotFlags_LinkAllX))
        {
          if (ImPlot::BeginPlot("Losses"))
          {
            ImPlot::SetupAxis(ImAxis_X1, "Batches trained");
            ImPlot::SetupAxis(ImAxis_Y1, "Loss function values");
            //ImPlot::SetupAxes("Batches trained", "Loss function values", x_flags, y_flags_loss);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, n_batch_train * n_epochs);
            ImPlot::SetupAxisLimits(ImAxis_Y1, y_min_loss, y_max_loss);//, ImPlotCond_Always);
            ImPlot::PlotLine("Average cost per batch processed", &LossDataBuffer.Data[0].x, &LossDataBuffer.Data[0].y, size_loss, LossDataBuffer.Offset, 2 * sizeof(float));
            ImPlot::EndPlot();
          }

          if (ImPlot::BeginPlot("Accuracy"))
          {
            ImPlot::SetupAxis(ImAxis_Y1, "Predictions accuracy");
            ImPlot::SetupAxisLimits(ImAxis_Y1, y_min_accuracy, y_max_accuracy);
            ImPlot::PlotLine("Average accuracy per batch processed", &AccuracyDataBuffer.Data[0].x, &AccuracyDataBuffer.Data[0].y, size_accuracy, AccuracyDataBuffer.Offset, 2 * sizeof(float));
            ImPlot::EndPlot();
          }

          ImPlot::EndSubplots();
        }

        ImGui::End();
      }
    }

    // Render ImGui data
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glClear(GL_COLOR_BUFFER_BIT);

    // Not sure why this is needed, from the example in imgui-glad-glm package
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Shutdown viewer
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
