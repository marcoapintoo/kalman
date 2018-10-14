from libcpp cimport bool


cdef extern from "SSM/StateSpaceModels" namespace "SSM::TimeInvariant":
    #
    ctypedef long int_t
    ctypedef double double_t
    ctypedef long long index_t
    #
    cdef void* _new_matrix2d()
    cdef void* _new_matrix3d()
    cdef void* _new_ssm_parameters()
    cdef void* _new_ssm_estimated()
    cdef void* _new_kalman_filter()
    cdef void* _new_kalman_smoother()
    cdef void* _new_expectation_maximization_estimator()
    cdef void* _new_pso_heuristic_estimator_particle()
    cdef void* _new_pso_heuristic_estimator()
    cdef void* _new_lse_heuristic_estimator_particle()
    cdef void* _new_lse_heuristic_estimator()
    cdef void* _new_em_heuristic_estimator_particle()
    cdef void* _new_em_heuristic_estimator()
    #
    cdef void _del_matrix2d(void* p)
    cdef void _del_matrix3d(void* p)
    cdef void _del_ssm_parameters(void* p)
    cdef void _del_ssm_estimated(void* p)
    cdef void _del_kalman_filter(void* p)
    cdef void _del_kalman_smoother(void* p)
    cdef void _del_expectation_maximization_estimator(void* p)
    cdef void _del_pso_heuristic_estimator_particle(void* p)
    cdef void _del_pso_heuristic_estimator(void* p)
    cdef void _del_lse_heuristic_estimator_particle(void* p)
    cdef void _del_lse_heuristic_estimator(void* p)
    cdef void _del_em_heuristic_estimator_particle(void* p)
    cdef void _del_em_heuristic_estimator(void* p)
    #
    cdef void* _create_matrix2d_from(double* data, index_t n_rows, index_t n_cols)
    #
    cdef void _fill_array_from2d(double* data, void* matrix)
    #
    cdef void _fill_array_from3d(double* data, void* matrix)
    #
    cdef void* _kalman_filter(
        index_t obs_dim, index_t lat_dim, index_t T,
        double_t* Y, double_t* F, double_t* H, 
        double_t* Q, double_t* R, 
        double_t* X0, double_t* P0)
    #
    cdef void _kalman_filter_results(
        double* Xp, double* Pp, double* Yp, 
        double* Xf, double* Pf, double* Yf,
        void* kf)
    #
    cdef void _kalman_filter_parameters(
        index_t* obs_dim, index_t* lat_dim, index_t* T,
        double* F, double* H,  double* Q,
        double* R, double* X0, double* P0,
        void* kalman_filter)
    #
    cdef void* _kalman_smoother(
        index_t obs_dim, index_t lat_dim, index_t T,
        double_t* Y, double_t* F, double_t* H, 
        double_t* Q, double_t* R, 
        double_t* X0, double_t* P0)
    #
    cdef void _kalman_smoother_results(
        double* Xp, double* Pp, double* Yp, 
        double* Xf, double* Pf, double* Yf,
        double* Xs, double* Ps, double* Ys,
        void* kalman_smoother)
    #
    cdef void _kalman_smoother_parameters(
        index_t* obs_dim, index_t* lat_dim, index_t* T,
        double* F, double* H,  double* Q,
        double* R, double* X0, double* P0,
        void* kalman_smoother)
    #
    cdef void* _estimate_using_em(
            const char* estimates,
            index_t obs_dim, index_t lat_dim, index_t T,
            double_t* Y,
            double_t* F, double_t* H, 
            double_t* Q, double_t* R, 
            double_t* X0, double_t* P0,
            index_t min_iterations,
            index_t max_iterations, 
            double_t min_improvement,
            double_t* loglikelihood_record)
    #
    cdef void* _estimate_using_pso(
            const char* estimates,
            index_t obs_dim, index_t lat_dim, index_t T,
            double_t* Y,
            double_t* F, double_t* H, 
            double_t* Q, double_t* R, 
            double_t* X0, double_t* P0,
            index_t min_iterations,
            index_t max_iterations, 
            double_t min_improvement,
            index_t sample_size,
            index_t population_size,
            double_t* loglikelihood_record,
            double_t penalty_low_variance_Q,
            double_t penalty_low_variance_R,
            double_t penalty_low_variance_P0,
            double_t penalty_low_std_mean_ratio,
            double_t penalty_inestable_system,
            double_t penalty_mse,
            double_t penalty_roughness_X,
            double_t penalty_roughness_Y)
    #
    cdef void* _estimate_using_lse_pso(
            const char* estimates,
            index_t obs_dim, index_t lat_dim, index_t T,
            double_t* Y,
            double_t* F, double_t* H, 
            double_t* Q, double_t* R, 
            double_t* X0, double_t* P0,
            index_t min_iterations,
            index_t max_iterations, 
            double_t min_improvement,
            index_t sample_size,
            index_t population_size,
            double_t* loglikelihood_record,
            double_t penalty_low_variance_Q,
            double_t penalty_low_variance_R,
            double_t penalty_low_variance_P0,
            double_t penalty_low_std_mean_ratio,
            double_t penalty_inestable_system,
            double_t penalty_mse,
            double_t penalty_roughness_X,
            double_t penalty_roughness_Y)
    #
    cdef void* _estimate_using_em_pso(
            const char* estimates,
            index_t obs_dim, index_t lat_dim, index_t T,
            double_t* Y,
            double_t* F, double_t* H, 
            double_t* Q, double_t* R, 
            double_t* X0, double_t* P0,
            index_t min_iterations,
            index_t max_iterations, 
            double_t min_improvement,
            index_t sample_size,
            index_t population_size,
            double_t* loglikelihood_record,
            double_t penalty_low_variance_Q,
            double_t penalty_low_variance_R,
            double_t penalty_low_variance_P0,
            double_t penalty_low_std_mean_ratio,
            double_t penalty_inestable_system,
            double_t penalty_mse,
            double_t penalty_roughness_X,
            double_t penalty_roughness_Y)
    #
    cdef void* _estimate_ssm(
            const char* type_of_estimator,
            const char* estimates,
            index_t obs_dim, index_t lat_dim, index_t T,
            double_t* Y,
            double_t* F, double_t* H, 
            double_t* Q, double_t* R, 
            double_t* X0, double_t* P0,
            index_t min_iterations,
            index_t max_iterations, 
            double_t min_improvement,
            index_t sample_size,
            index_t population_size,
            double_t* loglikelihood_record,
            double_t penalty_low_variance_Q,
            double_t penalty_low_variance_R,
            double_t penalty_low_variance_P0,
            double_t penalty_low_std_mean_ratio,
            double_t penalty_inestable_system,
            double_t penalty_mse,
            double_t penalty_roughness_X,
            double_t penalty_roughness_Y)
    #

import numpy as np
cimport numpy as np

ctypedef np.double_t dtype_t
#ctypedef np.ndarray[dtype_t, ndim=1] matrix1d_t
#ctypedef np.ndarray[dtype_t, ndim=2] matrix2d_t
#ctypedef np.ndarray[dtype_t, ndim=3] matrix3d_t
ctypedef dtype_t[:] matrix1d_t
ctypedef dtype_t[:, :] matrix2d_t
ctypedef dtype_t[:, :, :] matrix3d_t

cpdef _test_module():
    print("Test successfully installed!")

cdef unicode tounicode(char* s):
    return s.decode('UTF-8', 'strict')

cdef double_t* tovector3(matrix3d_t X):
    if X is None:
        return <double_t*>0
    return <double_t*>&X[0, 0, 0]

cdef double_t* tovector2(matrix2d_t X):
    if X is None:
        return <double_t*>0
    return <double_t*>&X[0, 0]

cdef double_t* tovector1(matrix1d_t X):
    if X is None:
        return <double_t*>0
    return <double_t*>&X[0]


cpdef estimate_ssm(str type_of_estimator,
    str estimates,
    matrix2d_t Y,
    index_t obs_dim=-1,
    index_t lat_dim=-1,
    index_t T=-1,
    matrix2d_t F=None, matrix2d_t H=None, 
    matrix2d_t Q=None, matrix2d_t R=None, 
    matrix2d_t X0=None, matrix2d_t P0=None,
    index_t min_iterations=1,
    index_t max_iterations=20, 
    double_t min_improvement=0.01,
    index_t sample_size=30,
    index_t population_size=50,
    double_t penalty_low_variance_Q=0.5,
    double_t penalty_low_variance_R=0.5,
    double_t penalty_low_variance_P0=0.5,
    double_t penalty_low_std_mean_ratio=0.5,
    double_t penalty_inestable_system=10.0,
    double_t penalty_mse=1e-5,
    double_t penalty_roughness_X=0.5,
    double_t penalty_roughness_Y=0.5,
    int max_length_loglikelihood=1000,
    bool return_details=False):
    type_of_estimator1 = type_of_estimator.encode('UTF-8')
    estimates1 = estimates.encode('UTF-8')
    cdef const char* _type_of_estimator = type_of_estimator1
    cdef const char* _estimates = estimates1
    cdef str dtype = "f8"
    cdef matrix1d_t loglikelihood_record = np.zeros(max_length_loglikelihood, dtype=dtype)
    cdef matrix2d_t Xp = np.zeros((lat_dim, T), dtype=dtype)
    cdef matrix3d_t Pp = np.zeros((lat_dim, lat_dim, T), dtype=dtype)
    cdef matrix2d_t Yp = np.zeros((obs_dim, T), dtype=dtype)
    cdef matrix2d_t Xf = np.zeros((lat_dim, T), dtype=dtype)
    cdef matrix3d_t Pf = np.zeros((lat_dim, lat_dim, T), dtype=dtype)
    cdef matrix2d_t Yf = np.zeros((obs_dim, T), dtype=dtype)
    cdef matrix2d_t Xs = np.zeros((lat_dim, T), dtype=dtype)
    cdef matrix3d_t Ps = np.zeros((lat_dim, lat_dim, T), dtype=dtype)
    cdef matrix2d_t Ys = np.zeros((obs_dim, T), dtype=dtype)

    cdef void* kalman_smoother = _estimate_ssm(
            _type_of_estimator,
            _estimates,
            obs_dim, lat_dim, T,
            tovector2(Y),
            tovector2(F), tovector2(H), 
            tovector2(Q), tovector2(R), 
            tovector2(X0), tovector2(P0),
            min_iterations,
            max_iterations, 
            min_improvement,
            sample_size,
            population_size,
            tovector1(loglikelihood_record),
            penalty_low_variance_Q,
            penalty_low_variance_R,
            penalty_low_variance_P0,
            penalty_low_std_mean_ratio,
            penalty_inestable_system,
            penalty_mse,
            penalty_roughness_X,
            penalty_roughness_Y)
    
    
    _kalman_smoother_results(
        tovector2(Xp), tovector3(Pp), tovector2(Yp),
        tovector2(Xf), tovector3(Pf), tovector2(Yf),
        tovector2(Xs), tovector3(Ps), tovector2(Ys),
        kalman_smoother)
    
    if return_details:
        return (
            Xp, Pp, Yp,
            Xf, Pf, Yf,
            Xs, Ps, Ys,
            loglikelihood_record
        )
    return (Xs, Ys)
