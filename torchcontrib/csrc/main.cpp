#include <torch/torch.h>

void show() {
    std::cout << "asdf" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("show", &show);
}
