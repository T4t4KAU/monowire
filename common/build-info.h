#pragma once

int monowire_build_number(void);

const char * monowire_commit(void);
const char * monowire_compiler(void);

const char * monowire_build_target(void);
const char * monowire_build_info(void);

void monowire_print_build_info(void);
