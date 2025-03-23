#include "simulation.hpp"
#include <Eigen/SparseLU>  // For pressure solver
#include <cmath>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

SimulationCore::SimulationCore(
    const py::array_t<double>& phi_H,
    const py::array_t<double>& phi_D,
    const py::array_t<double>& phi_N,
    const py::array_t<double>& nutrient,
    const py::array_t<double>& n_H,
    const py::array_t<double>& n_D,
    double dx,
    double dt,
    const py::dict& params
) {
    // Get buffer info for shape
    auto buf_H = phi_H.request();
    if (buf_H.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be 3");
    }

    // Store shape
    m_shape = {
        static_cast<size_t>(buf_H.shape[0]),
        static_cast<size_t>(buf_H.shape[1]),
        static_cast<size_t>(buf_H.shape[2])
    };

    // Convert numpy arrays to vectors
    m_phi_H = numpy_to_vector(phi_H);
    m_phi_D = numpy_to_vector(phi_D);
    m_phi_N = numpy_to_vector(phi_N);
    m_nutrient = numpy_to_vector(nutrient);
    m_n_H = numpy_to_vector(n_H);
    m_n_D = numpy_to_vector(n_D);

    // Store parameters
    m_dx = dx;
    m_dt = dt;

    // Extract model parameters from Python dict
    m_lambda_H = params["lambda_H"].cast<double>();
    m_lambda_D = params["lambda_D"].cast<double>();
    m_mu_H = params["mu_H"].cast<double>();
    m_mu_D = params["mu_D"].cast<double>();
    m_mu_N = params["mu_N"].cast<double>();
    m_p_H = params["p_H"].cast<double>();
    m_p_D = params["p_D"].cast<double>();
    m_gamma = params["gamma"].cast<double>();
    m_epsilon = params["epsilon"].cast<double>();
    m_M = params["M"].cast<double>();

    #ifdef _OPENMP
    m_num_threads = omp_get_max_threads();
    #else
    m_num_threads = 1;
    #endif
}

void SimulationCore::step_rk4() {
    std::cout << "Starting RK4 step..." << std::endl;
    const size_t size = m_phi_H.size();
    
    // Pre-allocate all vectors needed for RK4 steps
    std::vector<std::vector<double>> k1_arrays(4), k2_arrays(4), k3_arrays(4), k4_arrays(4);
    std::vector<std::vector<double>> temp_arrays(4);
    
    for (int i = 0; i < 4; ++i) {
        k1_arrays[i].resize(size);
        k2_arrays[i].resize(size);
        k3_arrays[i].resize(size);
        k4_arrays[i].resize(size);
        temp_arrays[i].resize(size);
    }

    try {
        // Compute k1
        std::cout << "Computing k1..." << std::endl;
        auto k1 = compute_derivatives(m_phi_H, m_phi_D, m_phi_N, m_nutrient);
        copy_tuple_to_arrays(k1, k1_arrays);

        // Compute k2
        std::cout << "Computing k2..." << std::endl;
        #pragma omp parallel for num_threads(m_num_threads)
        for (size_t i = 0; i < size; ++i) {
            for (int j = 0; j < 4; ++j) {
                temp_arrays[j][i] = get_state_array(j)[i] + (m_dt/2.0) * k1_arrays[j][i];
            }
        }
        auto k2 = compute_derivatives(temp_arrays[0], temp_arrays[1], temp_arrays[2], temp_arrays[3]);
        copy_tuple_to_arrays(k2, k2_arrays);

        // Compute k3
        std::cout << "Computing k3..." << std::endl;
        #pragma omp parallel for num_threads(m_num_threads)
        for (size_t i = 0; i < size; ++i) {
            for (int j = 0; j < 4; ++j) {
                temp_arrays[j][i] = get_state_array(j)[i] + (m_dt/2.0) * k2_arrays[j][i];
            }
        }
        auto k3 = compute_derivatives(temp_arrays[0], temp_arrays[1], temp_arrays[2], temp_arrays[3]);
        copy_tuple_to_arrays(k3, k3_arrays);

        // Compute k4
        std::cout << "Computing k4..." << std::endl;
        #pragma omp parallel for num_threads(m_num_threads)
        for (size_t i = 0; i < size; ++i) {
            for (int j = 0; j < 4; ++j) {
                temp_arrays[j][i] = get_state_array(j)[i] + m_dt * k3_arrays[j][i];
            }
        }
        auto k4 = compute_derivatives(temp_arrays[0], temp_arrays[1], temp_arrays[2], temp_arrays[3]);
        copy_tuple_to_arrays(k4, k4_arrays);

        // Update state with combined RK4 step
        const double h6 = m_dt / 6.0;
        #pragma omp parallel for num_threads(m_num_threads)
        for (size_t i = 0; i < size; ++i) {
            for (int j = 0; j < 4; ++j) {
                get_state_array(j)[i] += h6 * (
                    k1_arrays[j][i] + 
                    2.0 * k2_arrays[j][i] + 
                    2.0 * k3_arrays[j][i] + 
                    k4_arrays[j][i]
                );
            }
        }

    } catch (const std::exception& e) {
        std::cout << "Exception in step_rk4: " << e.what() << std::endl;
        throw;
    }
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
SimulationCore::get_state() const {
    return std::make_tuple(
        vector_to_numpy(m_phi_H),
        vector_to_numpy(m_phi_D),
        vector_to_numpy(m_phi_N),
        vector_to_numpy(m_nutrient)
    );
}

py::array_t<double> SimulationCore::vector_to_numpy(const std::vector<double>& vec) const {
    return py::array_t<double>(
        {m_shape[0], m_shape[1], m_shape[2]},  // shape
        vec.data()  // data
    );
}

std::vector<double> SimulationCore::numpy_to_vector(const py::array_t<double>& arr) const {
    auto buf = arr.request();
    return std::vector<double>(
        static_cast<double*>(buf.ptr),
        static_cast<double*>(buf.ptr) + buf.size
    );
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
SimulationCore::compute_derivatives(
    const std::vector<double>& phi_H,
    const std::vector<double>& phi_D,
    const std::vector<double>& phi_N,
    const std::vector<double>& nutrient
) {
    const size_t size = phi_H.size();
    std::vector<double> d_phi_H(size), d_phi_D(size), d_phi_N(size), d_nutrient(size);

    // Compute all sources in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto [src_H, src_D, src_N] = compute_cell_sources(phi_H, phi_D, phi_N, nutrient);
            #pragma omp parallel for num_threads(m_num_threads/4)
            for (size_t i = 0; i < size; ++i) {
                d_phi_H[i] = src_H[i];
                d_phi_D[i] = src_D[i];
                d_phi_N[i] = src_N[i];
            }
        }

        #pragma omp section
        {
            auto [dyn_H, dyn_D, dyn_N] = compute_cell_dynamics(phi_H, phi_D, phi_N, nutrient);
            #pragma omp parallel for num_threads(m_num_threads/4)
            for (size_t i = 0; i < size; ++i) {
                d_phi_H[i] += dyn_H[i];
                d_phi_D[i] += dyn_D[i];
                d_phi_N[i] += dyn_N[i];
            }
        }

        #pragma omp section
        {
            d_nutrient = compute_nutrient_diffusion(phi_H, phi_D, phi_N, nutrient);
        }
    }

    return {d_phi_H, d_phi_D, d_phi_N, d_nutrient};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
SimulationCore::compute_cell_sources(
    const std::vector<double>& phi_H,
    const std::vector<double>& phi_D,
    const std::vector<double>& phi_N,
    const std::vector<double>& nutrient
) {
    const size_t size = phi_H.size();
    std::vector<double> src_H(size), src_D(size), src_N(size);
    const double k = 5.0;

    #pragma omp parallel for num_threads(m_num_threads) schedule(static)
    for (size_t i = 0; i < size; ++i) {
        // Compute smooth Heaviside functions
        double H_H = 0.5 * (1.0 + std::tanh(k * (nutrient[i] - m_n_H[i])));
        double H_D = 0.5 * (1.0 + std::tanh(k * (nutrient[i] - m_n_D[i])));

        // Precompute common terms
        double nutrient_term_H = m_lambda_H * nutrient[i];
        double nutrient_term_D = m_lambda_D * nutrient[i];

        src_H[i] = nutrient_term_H * phi_H[i] * (2.0 * m_p_H - 1.0) 
                   - m_mu_H * H_H * phi_H[i] 
                   + 2.0 * nutrient_term_D * (1.0 - m_p_D) * phi_D[i];

        src_D[i] = 2.0 * nutrient_term_H * (1.0 - m_p_H) * phi_H[i] 
                   + nutrient_term_D * phi_D[i] * (2.0 * m_p_D - 1.0) 
                   - m_mu_D * H_D * phi_D[i];

        src_N[i] = m_mu_H * H_H * phi_H[i] 
                   + m_mu_D * H_D * phi_D[i] 
                   - m_mu_N * phi_N[i];
    }

    return {src_H, src_D, src_N};
}

// Helper functions for numerical operations
std::vector<double> SimulationCore::laplacian(const std::vector<double>& field) {
    const size_t nx = m_shape[0], ny = m_shape[1], nz = m_shape[2];
    std::vector<double> result(field.size(), 0.0);
    const double dx2 = m_dx * m_dx;

    // Process interior points in parallel with better chunking
    #pragma omp parallel for num_threads(m_num_threads) collapse(2)
    for (size_t i = 1; i < nx-1; ++i) {
        for (size_t j = 1; j < ny-1; ++j) {
            for (size_t k = 1; k < nz-1; ++k) {
                const size_t idx_c = idx(i,j,k);
                result[idx_c] = (
                    field[idx(i+1,j,k)] + field[idx(i-1,j,k)] +
                    field[idx(i,j+1,k)] + field[idx(i,j-1,k)] +
                    field[idx(i,j,k+1)] + field[idx(i,j,k-1)] -
                    6.0 * field[idx_c]
                ) / dx2;
            }
        }
    }

    // Handle boundaries in parallel sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // x boundaries
            for (size_t j = 0; j < ny; ++j) {
                for (size_t k = 0; k < nz; ++k) {
                    result[idx(0,j,k)] = (field[idx(1,j,k)] - field[idx(0,j,k)]) / dx2;
                    result[idx(nx-1,j,k)] = (field[idx(nx-2,j,k)] - field[idx(nx-1,j,k)]) / dx2;
                }
            }
        }

        #pragma omp section
        {
            // y boundaries
            for (size_t i = 0; i < nx; ++i) {
                for (size_t k = 0; k < nz; ++k) {
                    result[idx(i,0,k)] = (field[idx(i,1,k)] - field[idx(i,0,k)]) / dx2;
                    result[idx(i,ny-1,k)] = (field[idx(i,ny-2,k)] - field[idx(i,ny-1,k)]) / dx2;
                }
            }
        }

        #pragma omp section
        {
            // z boundaries
            for (size_t i = 0; i < nx; ++i) {
                for (size_t j = 0; j < ny; ++j) {
                    result[idx(i,j,0)] = (field[idx(i,j,1)] - field[idx(i,j,0)]) / dx2;
                    result[idx(i,j,nz-1)] = (field[idx(i,j,nz-2)] - field[idx(i,j,nz-1)]) / dx2;
                }
            }
        }
    }

    return result;
}

std::vector<double> SimulationCore::compute_pressure(
    const std::vector<double>& phi_H,
    const std::vector<double>& phi_D,
    const std::vector<double>& phi_N,
    const std::vector<double>& nutrient
) {
    const size_t size = phi_H.size();
    std::vector<double> pressure(size);
    
    #pragma omp parallel for simd num_threads(m_num_threads)
    for (size_t i = 0; i < size; ++i) {
        // Simplified pressure computation for stability
        const double total_phi = phi_H[i] + phi_D[i] + phi_N[i];
        pressure[i] = std::clamp(total_phi - 1.0, -1e6, 1e6);
    }
    
    return pressure;
}

std::vector<double> SimulationCore::compute_adhesion_energy_derivative(
    const std::vector<double>& phi_T,
    const std::vector<double>& laplace_phi_T
) {
    const size_t size = phi_T.size();
    std::vector<double> energy_deriv(size);

    for (size_t i = 0; i < size; ++i) {
        // Compute f'(phi_T) = 0.5 * phi_T * (1 - phi_T) * (2 * phi_T - 1)
        double f_prime = 0.5 * phi_T[i] * (1.0 - phi_T[i]) * (2.0 * phi_T[i] - 1.0);
        
        // Compute energy derivative
        energy_deriv[i] = (m_gamma / m_epsilon) * f_prime - 
                         0.01 * m_gamma * m_epsilon * laplace_phi_T[i];
    }

    return energy_deriv;
}

Eigen::SparseMatrix<double> SimulationCore::build_laplacian_matrix() {
    const size_t nx = m_shape[0], ny = m_shape[1], nz = m_shape[2];
    const size_t size = nx * ny * nz;
    const double dx2 = m_dx * m_dx;

    // Estimate number of non-zero entries
    const int entries_per_point = 7;  // diagonal + 6 neighbors
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(size * entries_per_point);

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                const size_t idx_center = idx(i,j,k);
                
                // Diagonal term
                double diag = -6.0;
                
                // Adjust for boundary conditions
                if (i == 0 || i == nx-1) diag += 1.0;
                if (j == 0 || j == ny-1) diag += 1.0;
                if (k == 0 || k == nz-1) diag += 1.0;
                
                triplets.emplace_back(idx_center, idx_center, diag/dx2);

                // Off-diagonal terms with Neumann boundary conditions
                if (i > 0)   triplets.emplace_back(idx_center, idx(i-1,j,k), 1.0/dx2);
                if (i < nx-1) triplets.emplace_back(idx_center, idx(i+1,j,k), 1.0/dx2);
                if (j > 0)   triplets.emplace_back(idx_center, idx(i,j-1,k), 1.0/dx2);
                if (j < ny-1) triplets.emplace_back(idx_center, idx(i,j+1,k), 1.0/dx2);
                if (k > 0)   triplets.emplace_back(idx_center, idx(i,j,k-1), 1.0/dx2);
                if (k < nz-1) triplets.emplace_back(idx_center, idx(i,j,k+1), 1.0/dx2);
            }
        }
    }

    Eigen::SparseMatrix<double> matrix(size, size);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
SimulationCore::compute_cell_dynamics(
    const std::vector<double>& phi_H,
    const std::vector<double>& phi_D,
    const std::vector<double>& phi_N,
    const std::vector<double>& nutrient
) {
    std::cout << "Entering compute_cell_dynamics with grid size: " << phi_H.size() << std::endl;
    const size_t size = phi_H.size();
    
    // Single allocation for all working vectors
    std::vector<double> all_data(size * 7, 0.0);  // Allocate all memory at once
    
    // Create views into the allocated memory
    double* dyn_H_ptr = all_data.data();
    double* dyn_D_ptr = dyn_H_ptr + size;
    double* dyn_N_ptr = dyn_D_ptr + size;
    double* phi_T_ptr = dyn_N_ptr + size;
    double* grad_x_ptr = phi_T_ptr + size;
    double* grad_y_ptr = grad_x_ptr + size;
    double* grad_z_ptr = grad_y_ptr + size;
    
    try {
        // Compute total density with SIMD optimization
        #pragma omp parallel for simd num_threads(m_num_threads)
        for (size_t i = 0; i < size; ++i) {
            phi_T_ptr[i] = std::clamp(phi_H[i] + phi_D[i] + phi_N[i], 0.0, 1.0);
        }
        
        // Compute gradients directly instead of using separate functions
        const size_t nx = m_shape[0], ny = m_shape[1], nz = m_shape[2];
        const double dx2 = m_dx * m_dx;
        
        #pragma omp parallel for collapse(3) num_threads(m_num_threads)
        for (size_t i = 1; i < nx-1; ++i) {
            for (size_t j = 1; j < ny-1; ++j) {
                for (size_t k = 1; k < nz-1; ++k) {
                    const size_t idx_c = idx(i,j,k);
                    
                    // Compute all gradients at once
                    grad_x_ptr[idx_c] = (phi_T_ptr[idx(i+1,j,k)] - phi_T_ptr[idx(i-1,j,k)]) / (2.0 * m_dx);
                    grad_y_ptr[idx_c] = (phi_T_ptr[idx(i,j+1,k)] - phi_T_ptr[idx(i,j-1,k)]) / (2.0 * m_dx);
                    grad_z_ptr[idx_c] = (phi_T_ptr[idx(i,j,k+1)] - phi_T_ptr[idx(i,j,k-1)]) / (2.0 * m_dx);
                }
            }
        }
        
        // Compute dynamics directly with SIMD
        #pragma omp parallel for simd num_threads(m_num_threads)
        for (size_t i = 0; i < size; ++i) {
            // Simplified dynamics computation
            const double total_grad = std::sqrt(
                grad_x_ptr[i] * grad_x_ptr[i] +
                grad_y_ptr[i] * grad_y_ptr[i] +
                grad_z_ptr[i] * grad_z_ptr[i]
            );
            
            const double mobility = 1.0 - phi_T_ptr[i];  // Simple mobility term
            
            // Update dynamics with simplified model
            dyn_H_ptr[i] = std::clamp(-mobility * grad_x_ptr[i], -1.0, 1.0);
            dyn_D_ptr[i] = std::clamp(-mobility * grad_y_ptr[i], -1.0, 1.0);
            dyn_N_ptr[i] = std::clamp(-mobility * grad_z_ptr[i], -1.0, 1.0);
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception in compute_cell_dynamics: " << e.what() << std::endl;
    }
    
    // Create return vectors without copying
    return {
        std::vector<double>(dyn_H_ptr, dyn_H_ptr + size),
        std::vector<double>(dyn_D_ptr, dyn_D_ptr + size),
        std::vector<double>(dyn_N_ptr, dyn_N_ptr + size)
    };
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
SimulationCore::compute_solid_velocity(
    const std::vector<double>& pressure,
    const std::vector<double>& phi_H,
    const std::vector<double>& phi_D,
    const std::vector<double>& phi_N,
    const std::vector<double>& energy_deriv
) {
    const size_t size = pressure.size();
    
    // Compute total cell density
    std::vector<double> phi_T(size);
    for (size_t i = 0; i < size; ++i) {
        phi_T[i] = phi_H[i] + phi_D[i] + phi_N[i];
    }

    // Compute gradients
    auto grad_p_x = gradient_x(pressure);
    auto grad_p_y = gradient_y(pressure);
    auto grad_p_z = gradient_z(pressure);

    auto grad_phi_x = gradient_x(phi_T);
    auto grad_phi_y = gradient_y(phi_T);
    auto grad_phi_z = gradient_z(phi_T);

    // Compute velocity components
    std::vector<double> ux(size), uy(size), uz(size);
    for (size_t i = 0; i < size; ++i) {
        ux[i] = -(grad_p_x[i] + energy_deriv[i] * grad_phi_x[i]);
        uy[i] = -(grad_p_y[i] + energy_deriv[i] * grad_phi_y[i]);
        uz[i] = -(grad_p_z[i] + energy_deriv[i] * grad_phi_z[i]);

        // Clip velocity components
        ux[i] = std::clamp(ux[i], -1.0, 1.0);
        uy[i] = std::clamp(uy[i], -1.0, 1.0);
        uz[i] = std::clamp(uz[i], -1.0, 1.0);
    }

    return {ux, uy, uz};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
SimulationCore::compute_mass_flux(
    const std::vector<double>& v_cell,
    const std::vector<double>& phi_D,
    const std::vector<double>& phi_N,
    const std::vector<double>& energy_deriv
) {
    const size_t size = v_cell.size();

    // Compute gradients of energy derivative
    auto grad_energy_x = gradient_x(energy_deriv);
    auto grad_energy_y = gradient_y(energy_deriv);
    auto grad_energy_z = gradient_z(energy_deriv);

    // Compute total cell density
    std::vector<double> phi_T(size);
    for (size_t i = 0; i < size; ++i) {
        phi_T[i] = v_cell[i] + phi_D[i] + phi_N[i];
    }

    // Compute mass flux components
    std::vector<double> Jx(size), Jy(size), Jz(size);
    const double epsilon_small = 1e-6;

    for (size_t i = 0; i < size; ++i) {
        double phi_T_clamped = std::max(phi_T[i], epsilon_small);
        double scaling = v_cell[i] / phi_T_clamped;

        Jx[i] = -m_M * grad_energy_x[i] * scaling;
        Jy[i] = -m_M * grad_energy_y[i] * scaling;
        Jz[i] = -m_M * grad_energy_z[i] * scaling;
    }

    return {Jx, Jy, Jz};
}

std::vector<double> SimulationCore::compute_divergence(
    const std::vector<double>& field_x,
    const std::vector<double>& field_y,
    const std::vector<double>& field_z
) {
    const size_t nx = m_shape[0], ny = m_shape[1], nz = m_shape[2];
    std::vector<double> div(field_x.size(), 0.0);

    for (size_t i = 1; i < nx-1; ++i) {
        for (size_t j = 1; j < ny-1; ++j) {
            for (size_t k = 1; k < nz-1; ++k) {
                const size_t idx_c = idx(i,j,k);
                
                // Central difference for each direction
                double dx_term = (field_x[idx(i+1,j,k)] - field_x[idx(i-1,j,k)]) / (2.0 * m_dx);
                double dy_term = (field_y[idx(i,j+1,k)] - field_y[idx(i,j-1,k)]) / (2.0 * m_dx);
                double dz_term = (field_z[idx(i,j,k+1)] - field_z[idx(i,j,k-1)]) / (2.0 * m_dx);

                div[idx_c] = dx_term + dy_term + dz_term;
            }
        }
    }

    // Handle boundaries with Neumann conditions
    // ... (similar to laplacian boundary handling)

    return div;
}

// Add gradient implementations
std::vector<double> SimulationCore::gradient_x(const std::vector<double>& field) {
    const size_t nx = m_shape[0], ny = m_shape[1], nz = m_shape[2];
    std::vector<double> grad(field.size(), 0.0);

    // Central differences for interior points
    for (size_t i = 1; i < nx-1; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                grad[idx(i,j,k)] = (field[idx(i+1,j,k)] - field[idx(i-1,j,k)]) / (2.0 * m_dx);
            }
        }
    }

    // Neumann boundary conditions at x boundaries
    for (size_t j = 0; j < ny; ++j) {
        for (size_t k = 0; k < nz; ++k) {
            // Forward difference at x = 0
            grad[idx(0,j,k)] = 0.0;  // Neumann condition
            // Backward difference at x = nx-1
            grad[idx(nx-1,j,k)] = 0.0;  // Neumann condition
        }
    }

    return grad;
}

std::vector<double> SimulationCore::gradient_y(const std::vector<double>& field) {
    const size_t nx = m_shape[0], ny = m_shape[1], nz = m_shape[2];
    std::vector<double> grad(field.size(), 0.0);

    // Central differences for interior points
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 1; j < ny-1; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                grad[idx(i,j,k)] = (field[idx(i,j+1,k)] - field[idx(i,j-1,k)]) / (2.0 * m_dx);
            }
        }
    }

    // Neumann boundary conditions at y boundaries
    for (size_t i = 0; i < nx; ++i) {
        for (size_t k = 0; k < nz; ++k) {
            grad[idx(i,0,k)] = 0.0;  // Neumann condition
            grad[idx(i,ny-1,k)] = 0.0;  // Neumann condition
        }
    }

    return grad;
}

std::vector<double> SimulationCore::gradient_z(const std::vector<double>& field) {
    const size_t nx = m_shape[0], ny = m_shape[1], nz = m_shape[2];
    std::vector<double> grad(field.size(), 0.0);

    // Central differences for interior points
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 1; k < nz-1; ++k) {
                grad[idx(i,j,k)] = (field[idx(i,j,k+1)] - field[idx(i,j,k-1)]) / (2.0 * m_dx);
            }
        }
    }

    // Neumann boundary conditions at z boundaries
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            grad[idx(i,j,0)] = 0.0;  // Neumann condition
            grad[idx(i,j,nz-1)] = 0.0;  // Neumann condition
        }
    }

    return grad;
}

// Add nutrient diffusion implementation
std::vector<double> SimulationCore::compute_nutrient_diffusion(
    const std::vector<double>& phi_H,
    const std::vector<double>& phi_D,
    const std::vector<double>& phi_N,
    const std::vector<double>& nutrient
) {
    const size_t size = nutrient.size();
    
    // Compute Laplacian of nutrient
    auto nutrient_laplacian = laplacian(nutrient);
    
    // Compute consumption terms
    std::vector<double> d_nutrient(size);
    for (size_t i = 0; i < size; ++i) {
        // Diffusion term
        d_nutrient[i] = nutrient_laplacian[i];
        
        // Consumption by cells (you may need to adjust these coefficients based on your model)
        d_nutrient[i] -= 0.1 * nutrient[i] * phi_H[i];  // Healthy cell consumption
        d_nutrient[i] -= 0.2 * nutrient[i] * phi_D[i];  // Diseased cell consumption
        // No consumption by necrotic cells
    }

    return d_nutrient;
}

std::vector<double>& SimulationCore::get_state_array(int index) {
    switch(index) {
        case 0: return m_phi_H;
        case 1: return m_phi_D;
        case 2: return m_phi_N;
        case 3: return m_nutrient;
        default: throw std::runtime_error("Invalid state array index");
    }
}

void SimulationCore::copy_tuple_to_arrays(
    const std::tuple<std::vector<double>, std::vector<double>, 
                    std::vector<double>, std::vector<double>>& tuple,
    std::vector<std::vector<double>>& arrays) {
    arrays[0] = std::get<0>(tuple);
    arrays[1] = std::get<1>(tuple);
    arrays[2] = std::get<2>(tuple);
    arrays[3] = std::get<3>(tuple);
}
