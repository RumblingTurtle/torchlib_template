#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[]) {
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("../resources/module.jit");
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({49}));
    std::cout << module.forward(inputs).toTensor() << std::endl;
    std::cout << "ok\n";
}