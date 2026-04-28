#include "monowire-memory.h"

monowire_memory_status monowire_memory_status_combine(monowire_memory_status s0, monowire_memory_status s1) {
    bool has_update = false;

    switch (s0) {
    case MONOWIRE_MEMORY_STATUS_SUCCESS: {
        has_update = true;
        break;
    }
    case MONOWIRE_MEMORY_STATUS_NO_UPDATE: {
        break;
    }
    case MONOWIRE_MEMORY_STATUS_FAILED_PREPARE:
    case MONOWIRE_MEMORY_STATUS_FAILED_COMPUTE: {
        return s0;
    }
    }

    switch (s1) {
    case MONOWIRE_MEMORY_STATUS_SUCCESS: {
        has_update = true;
        break;
    }
    case MONOWIRE_MEMORY_STATUS_NO_UPDATE: {
        break;
    }
    case MONOWIRE_MEMORY_STATUS_FAILED_PREPARE:
    case MONOWIRE_MEMORY_STATUS_FAILED_COMPUTE: {
        return s1;
    }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? MONOWIRE_MEMORY_STATUS_SUCCESS : MONOWIRE_MEMORY_STATUS_NO_UPDATE;
}

bool monowire_memory_status_is_fail(monowire_memory_status status) {
    switch (status) {
    case MONOWIRE_MEMORY_STATUS_SUCCESS:
    case MONOWIRE_MEMORY_STATUS_NO_UPDATE: {
        return false;
    }
    case MONOWIRE_MEMORY_STATUS_FAILED_PREPARE:
    case MONOWIRE_MEMORY_STATUS_FAILED_COMPUTE: {
        return true;
    }
    }

    return false;
}
