#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <array>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#ifdef _OPENMP
#include <omp.h>
#endif

// Add these helper macros for vectorization hints
#if defined(__GNUC__) || defined(__clang__)
#define VECTORIZE_HINT __attribute__((vector))
#elif defined(_MSC_VER)
#define VECTORIZE_HINT __pragma(vector_always)
#else
#define VECTORIZE_HINT
#endif

namespace py = pybind11;

class SimulationCore {
public:
    SimulationCore(
        const py::array_t<double>& phi_H,
        const py::array_t<double>& phi_D,
        const py::array_t<double>& phi_N,
        const py::array_t<double>& nutrient,
        const py::array_t<double>& n_H,
        const py::array_t<double>& n_D,
        double dx,
        double dt,
        const py::dict& params
    );

    // Add a new method for getting the current state as numpy arrays
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
    get_state() const;

    // Change step_rk4 to return void since we don't need the arrays every step
    void step_rk4();

    // Add compute_divergence declaration
    std::vector<double> compute_divergence(
        const std::vector<double>& field_x,
        const std::vector<double>& field_y,
        const std::vector<double>& field_z
    );

private:
    // Add thread count parameter
    int m_num_threads;
    
    // Use standard vectors instead of aligned vectors
    std::vector<double> m_phi_H;
    std::vector<double> m_phi_D;
    std::vector<double> m_phi_N;
    std::vector<double> m_nutrient;
    std::vector<double> m_n_H;
    std::vector<double> m_n_D;

    // Grid parameters
    std::array<size_t, 3> m_shape;
    double m_dx;
    double m_dt;

    // Model parameters
    double m_lambda_H;
    double m_lambda_D;
    double m_mu_H;
    double m_mu_D;
    double m_mu_N;
    double m_p_H;
    double m_p_D;
    double m_gamma;
    double m_epsilon;
    double m_M;

    // Helper methods
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
    compute_derivatives(
        const std::vector<double>& phi_H,
        const std::vector<double>& phi_D,
        const std::vector<double>& phi_N,
        const std::vector<double>& nutrient
    );

    // Convert between numpy arrays and std::vector
    py::array_t<double> vector_to_numpy(const std::vector<double>& vec) const;
    std::vector<double> numpy_to_vector(const py::array_t<double>& arr) const;

    // Add these helper methods
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    compute_cell_sources(
        const std::vector<double>& phi_H,
        const std::vector<double>& phi_D,
        const std::vector<double>& phi_N,
        const std::vector<double>& nutrient
    );

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    compute_cell_dynamics(
        const std::vector<double>& phi_H,
        const std::vector<double>& phi_D,
        const std::vector<double>& phi_N,
        const std::vector<double>& nutrient
    );

    std::vector<double>
    compute_nutrient_diffusion(
        const std::vector<double>& phi_H,
        const std::vector<double>& phi_D,
        const std::vector<double>& phi_N,
        const std::vector<double>& nutrient
    );

    // Helper functions for numerical operations
    std::vector<double> laplacian(const std::vector<double>& field);
    std::vector<double> gradient_x(const std::vector<double>& field);
    std::vector<double> gradient_y(const std::vector<double>& field);
    std::vector<double> gradient_z(const std::vector<double>& field);
    size_t idx(size_t i, size_t j, size_t k) const {
        return i * m_shape[1] * m_shape[2] + j * m_shape[2] + k;
    }

    // Add these methods for pressure solver
    std::vector<double> compute_pressure(
        const std::vector<double>& phi_H,
        const std::vector<double>& phi_D,
        const std::vector<double>& phi_N,
        const std::vector<double>& nutrient
    );


    // Add these methods for adhesion energy derivative
    std::vector<double> compute_adhesion_energy_derivative(
        const std::vector<double>& phi_T,
        const std::vector<double>& laplace_phi_T
    );

    // Add these methods for Laplacian matrix
    Eigen::SparseMatrix<double> build_laplacian_matrix();

    // Cache the Laplacian matrix and solver
    Eigen::SparseMatrix<double> m_laplacian_matrix;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> m_solver;
    bool m_solver_initialized = false;

    // Velocity and mass flux methods
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    compute_solid_velocity(
        const std::vector<double>& pressure,
        const std::vector<double>& phi_H,
        const std::vector<double>& phi_D,
        const std::vector<double>& phi_N,
        const std::vector<double>& energy_deriv
    );

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    compute_mass_flux(
        const std::vector<double>& v_cell,
        const std::vector<double>& phi_D,
        const std::vector<double>& phi_N,
        const std::vector<double>& energy_deriv
    );
};
