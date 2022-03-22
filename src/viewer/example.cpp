#include "use_imgui.h"

int main() {
    UseImgui viewer;
    viewer.init();
    viewer.show_demo();
    viewer.shutdown();
}
