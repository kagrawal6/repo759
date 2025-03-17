#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <omp.h>

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Size of the board

// --------------------------------------------------------------
// Function to calculate acceleration due to gravity in parallel
// with OpenMP. Mirrors the Python getAcc() implementation.
// --------------------------------------------------------------
void getAcc(const double pos[][3], const double mass[], double acc[][3], int N) {
    // Reset accelerations to 0
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        acc[i][0] = 0.0;
        acc[i][1] = 0.0;
        acc[i][2] = 0.0;
    }

    // Calculate gravitational interactions
    // We parallelize over i; each thread accumulates the total acceleration
    // on particle i from all other j particles.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0;  // local accumulators
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = pos[j][0] - pos[i][0];
                double dy = pos[j][1] - pos[i][1];
                double dz = pos[j][2] - pos[i][2];

                double r2 = dx*dx + dy*dy + dz*dz + softening*softening;
                double inv_r3 = 1.0 / (std::pow(r2, 1.5));

                ax += G * dx * inv_r3 * mass[j];
                ay += G * dy * inv_r3 * mass[j];
                az += G * dz * inv_r3 * mass[j];
            }
        }
        acc[i][0] = ax;
        acc[i][1] = ay;
        acc[i][2] = az;
    }
}

// (Optional) For debugging: save positions to CSV
void savePositionsToCSV(const double pos[][3], int N, int step, const std::string &filename) {
    std::ofstream file;
    // Open the file in append mode
    file.open(filename, std::ios_base::app);
    if (file.is_open()) {
        file << step << ",[";
        for (int i = 0; i < N; i++) {
            if (i != N - 1)
                file << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << "],";
            else
                file << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << "]";
        }
        file << "]\n";  // Newline for separation between steps
        file.close();
    } else {
        std::cerr << "Unable to open file!" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------
    // 1) Parse command-line arguments
    //    Expect three args: N, tEnd, numThreads
    // -----------------------------------------------------------------
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <num_particles> <simulation_end_time> <num_threads>\n";
        return 1;
    }

    int    N       = std::stoi(argv[1]);    // number of particles
    double tEnd    = std::stod(argv[2]);    // end time
    int    threads = std::stoi(argv[3]);    // number of OMP threads

    // Set number of threads for OpenMP
    omp_set_num_threads(threads);

    // -----------------------------------------------------------------
    // 2) Start timer using omp_get_wtime()
    // -----------------------------------------------------------------
    double wtime_start = omp_get_wtime();

    // (Optional) We could clear or append to a CSV file
    std::string filename = "positions_omp_static.csv";
      // Clear the file before starting simulation (optional)
      std::ofstream file;
      file.open(filename, std::ofstream::out | std::ofstream::trunc);
      file.close();

    // 3) Allocate arrays dynamically
    double* mass    = new double[N];
    double(*pos)[3] = new double[N][3];
    double(*vel)[3] = new double[N][3];
    double(*acc)[3] = new double[N][3];

    // -----------------------------------------------------------------
    // 4) Generate Initial Conditions
    // -----------------------------------------------------------------
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // Initialize random masses, positions, velocities
    for (int i = 0; i < N; i++) {
        mass[i]   = uniform_dist(generator);
        pos[i][0] = normal_dist(generator);
        pos[i][1] = normal_dist(generator);
        pos[i][2] = normal_dist(generator);

        vel[i][0] = normal_dist(generator);
        vel[i][1] = normal_dist(generator);
        vel[i][2] = normal_dist(generator);
    }

    // Convert to Center-of-Mass frame
    double velCM[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;
    for (int i = 0; i < N; i++) {
        velCM[0] += vel[i][0] * mass[i];
        velCM[1] += vel[i][1] * mass[i];
        velCM[2] += vel[i][2] * mass[i];
        totalMass += mass[i];
    }
    velCM[0] /= totalMass;
    velCM[1] /= totalMass;
    velCM[2] /= totalMass;

    for (int i = 0; i < N; i++) {
        vel[i][0] -= velCM[0];
        vel[i][1] -= velCM[1];
        vel[i][2] -= velCM[2];
    }

    // 5) Initial Acceleration
    getAcc(pos, mass, acc, N);

    // Number of timesteps
    int Nt = int(tEnd / dt);
    double t = 0.0; // current simulation time

    // -----------------------------------------------------------------
    // 6) Main Simulation Loop (Leapfrog Integration)
    // -----------------------------------------------------------------
    for (int step = 0; step < Nt; step++) {
        // (1) Half Kick
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * (dt / 2.0);
            vel[i][1] += acc[i][1] * (dt / 2.0);
            vel[i][2] += acc[i][2] * (dt / 2.0);
        }

        // (2) Drift
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;
        }

        // (3) Ensure board limits
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            if (pos[i][0] > board_size) pos[i][0] = board_size;
            else if (pos[i][0] < -board_size) pos[i][0] = -board_size;

            if (pos[i][1] > board_size) pos[i][1] = board_size;
            else if (pos[i][1] < -board_size) pos[i][1] = -board_size;

            if (pos[i][2] > board_size) pos[i][2] = board_size;
            else if (pos[i][2] < -board_size) pos[i][2] = -board_size;
        }

        // (4) Recompute accelerations
        getAcc(pos, mass, acc, N);

        // (5) Half Kick
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * (dt / 2.0);
            vel[i][1] += acc[i][1] * (dt / 2.0);
            vel[i][2] += acc[i][2] * (dt / 2.0);
        }

        // (6) Advance time
        t += dt;

        // (Optional) Save positions
        // savePositionsToCSV(pos, N, step, filename);
    }

    // -----------------------------------------------------------------
    // 7) Cleanup + End Timer
    // -----------------------------------------------------------------
    delete[] mass;
    delete[] pos;
    delete[] vel;
    delete[] acc;

    double wtime_end = omp_get_wtime();
    double elapsed = (wtime_end - wtime_start) * 1000.0; // convert seconds to ms
    std::cout << elapsed << " \n";

    return 0;
}
