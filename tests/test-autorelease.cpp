// ref: upstream-reference

#include <cstdio>
#include <string>
#include <thread>

#include "monowire.h"
#include "get-model.h"

// This creates a new context inside a pthread and then tries to exit cleanly.
int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);

    std::thread([&model_path]() {
        monowire_backend_init();
        auto * model = monowire_model_load_from_file(model_path, monowire_model_default_params());
        auto * ctx = monowire_init_from_model(model, monowire_context_default_params());
        monowire_free(ctx);
        monowire_model_free(model);
        monowire_backend_free();
    }).join();

    return 0;
}
