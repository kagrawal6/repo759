#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Size of the board

// --------------------------------------------------------------
// Function to calculate acceleration due to gravity
// Mirrors the Python getAcc() implementation
// --------------------------------------------------------------
void getAcc(const double pos[][3], const double mass[], double acc[][3], int N) {
    // Reset accelerations to 0
    for (int i = 0; i < N; i++) {
        acc[i][0] = 0.0;
        acc[i][1] = 0.0;
        acc[i][2] = 0.0;
    }

    // Calculate gravitational interactions
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = pos[j][0] - pos[i][0];
                double dy = pos[j][1] - pos[i][1];
                double dz = pos[j][2] - pos[i][2];

                // Add softening^2 to avoid singularities when particles get very close
                double r2 = dx*dx + dy*dy + dz*dz + softening*softening;
                // Inverse distance cubed
                double inv_r3 = 1.0 / (std::pow(r2, 1.5));

                // Accumulate acceleration contribution from particle j
                acc[i][0] += G * dx * inv_r3 * mass[j];
                acc[i][1] += G * dy * inv_r3 * mass[j];
                acc[i][2] += G * dz * inv_r3 * mass[j];
            }
        }
    }
}

// --------------------------------------------------------------
// For debugging: save positions to a CSV file
// Comment out if measuring performance
// --------------------------------------------------------------
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

int main(int argc, char *argv[]) {
    // Measure wall-clock time
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    // Check if correct number of arguments are provided
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_particles> <simulation_end_time>" << std::endl;
        return 1;
    }

    // Read N and tEnd from command line
    int N = std::stoi(argv[1]);      // Number of particles
    double tEnd = std::stod(argv[2]); // Time at which simulation ends

    // File to save positions (optional)
    std::string filename = "positions.csv";
    
    // Clear the file before starting simulation (optional)
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::trunc);
    file.close();

    // Allocate dynamic arrays based on N
    double* mass = new double[N];
    double(*pos)[3] = new double[N][3];
    double(*vel)[3] = new double[N][3];
    double(*acc)[3] = new double[N][3];

    // Create a random number engine
    std::mt19937 generator(std::random_device{}());

    // Create random distributions
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // Simulation parameters
    double t = 0.0;

    // Set initial masses and random positions/velocities
    for (int i = 0; i < N; i++) {
        mass[i] = uniform_dist(generator);

        pos[i][0] = normal_dist(generator);
        pos[i][1] = normal_dist(generator);
        pos[i][2] = normal_dist(generator);

        vel[i][0] = normal_dist(generator);
        vel[i][1] = normal_dist(generator);
        vel[i][2] = normal_dist(generator);
    }

    // --------------------------------------------------------------
    // Convert to Center-of-Mass frame
    // --------------------------------------------------------------
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

    // --------------------------------------------------------------
    // Compute initial accelerations
    // --------------------------------------------------------------
    getAcc(pos, mass, acc, N);

     // Number of timesteps
     int Nt = int(tEnd / dt);

    // --------------------------------------------------------------
    // Main simulation loop (Leapfrog integration)
    // --------------------------------------------------------------
    for (int step = 0; step < Nt; step++) {
        // (1) Half Kick
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * (dt / 2.0);
            vel[i][1] += acc[i][1] * (dt / 2.0);
            vel[i][2] += acc[i][2] * (dt / 2.0);
        }

        // (2) Drift
        for (int i = 0; i < N; i++) {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;
        }

        // (3) Ensure particles stay within the board limits
        for (int i = 0; i < N; i++) {
            if (pos[i][0] > board_size)      pos[i][0] = board_size;
            else if (pos[i][0] < -board_size) pos[i][0] = -board_size;

            if (pos[i][1] > board_size)      pos[i][1] = board_size;
            else if (pos[i][1] < -board_size) pos[i][1] = -board_size;

            if (pos[i][2] > board_size)      pos[i][2] = board_size;
            else if (pos[i][2] < -board_size) pos[i][2] = -board_size;
        }

        // (4) Recompute accelerations with updated positions
        getAcc(pos, mass, acc, N);

        // (5) Half Kick
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * (dt / 2.0);
            vel[i][1] += acc[i][1] * (dt / 2.0);
            vel[i][2] += acc[i][2] * (dt / 2.0);
        }

        // (6) Update simulation time
        t += dt;

        // Optional: save positions to CSV at each step
        savePositionsToCSV(pos, N, step, filename);
    }

    // --------------------------------------------------------------
    // Clean up dynamically allocated memory
    // --------------------------------------------------------------
    delete[] mass;
    delete[] pos;
    delete[] vel;
    delete[] acc;

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    std::cout << "time: " << duration_sec.count() << "ms\n";

    return 0;
}
