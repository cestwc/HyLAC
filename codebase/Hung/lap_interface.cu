#include <cuda.h>
#include "LAP/Hung_lap.cuh"  // Your LAP template class and GLOBAL_HANDLE

typedef unsigned int uint;

extern "C" {

// Abstract pointer to LAP object
typedef void* LAPHandle;

// Create a new LAP instance
LAPHandle create_lap(uint *h_costs, int user_n, int dev) {
    return new LAP<uint>(h_costs, user_n, dev);
}

// Solve the LAP and return the assignment as a host array
void solve_lap_with_result(LAPHandle handle, int *assignment_out, int user_n) {
    LAP<uint> *lap = reinterpret_cast<LAP<uint> *>(handle);
    lap->solve();

    // Use passed-in user_n instead of lap->get_size()
    for (int r = 0; r < user_n; ++r) {
        assignment_out[r] = lap->gh.column_of_star_at_row[r];
    }
}

// Free the LAP object
void destroy_lap(LAPHandle handle) {
    LAP<uint> *lap = reinterpret_cast<LAP<uint> *>(handle);
    delete lap;
}

}
