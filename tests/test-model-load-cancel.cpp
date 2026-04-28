#include "monowire.h"
#include "get-model.h"

#include <cstdlib>

int main(int argc, char *argv[] ) {
    auto * model_path = get_model_or_exit(argc, argv);
    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    monowire_backend_init();
    auto params = monowire_model_params{};
    params.use_mmap = false;
    params.progress_callback = [](float progress, void * ctx){
        (void) ctx;
        return progress > 0.50;
    };
    auto * model = monowire_model_load_from_file(model_path, params);
    monowire_backend_free();
    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
}
