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
            double_t penalty_roughness_Y,
            double_t random_spread)
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
            double_t penalty_roughness_Y,
            double_t random_spread)
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
            double_t penalty_roughness_Y,
            double_t random_spread)
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
            double_t penalty_roughness_Y,
            double_t random_spread) except +RuntimeError
    #
    cdef void _performance_of_parameters(
                double_t* loglikelihood, 
                double_t* low_std_to_mean_penalty, 
                double_t* low_variance_Q_penalty, 
                double_t* low_variance_R_penalty, 
                double_t* low_variance_P0_penalty, 
                double_t* system_inestability_penalty, 
                double_t* mean_squared_error_penalty, 
                double_t* roughness_X_penalty, 
                double_t* roughness_Y_penalty, 
                index_t obs_dim, index_t lat_dim, index_t T,
                double_t* Y,
                double_t* F, double_t* H, 
                double_t* Q, double_t* R, 
                double_t* X0, double_t* P0) except +RuntimeError
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

cdef index_t* tovector1_i(index_t[:] X):
    if X is None:
        return <index_t*>0
    return <index_t*>&X[0]


cdef _estimate_ssm_p(str type_of_estimator,
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
    double_t random_spread=0.5,
    int max_length_loglikelihood=1000,
    bool return_details=False):

    type_of_estimator1 = type_of_estimator.encode('UTF-8')
    estimates1 = estimates.encode('UTF-8')
    cdef const char* _type_of_estimator = type_of_estimator1
    cdef const char* _estimates = estimates1
    cdef str dtype = "f8"
    cdef matrix1d_t loglikelihood_record = np.zeros(max_length_loglikelihood, dtype=dtype)
    #
    cdef matrix2d_t Xp = np.zeros((lat_dim, T), dtype=dtype)
    cdef matrix3d_t Pp = np.zeros((lat_dim, lat_dim, T), dtype=dtype)
    cdef matrix2d_t Yp = np.zeros((obs_dim, T), dtype=dtype)
    cdef matrix2d_t Xf = np.zeros((lat_dim, T), dtype=dtype)
    cdef matrix3d_t Pf = np.zeros((lat_dim, lat_dim, T), dtype=dtype)
    cdef matrix2d_t Yf = np.zeros((obs_dim, T), dtype=dtype)
    cdef matrix2d_t Xs = np.zeros((lat_dim, T), dtype=dtype)
    cdef matrix3d_t Ps = np.zeros((lat_dim, lat_dim, T), dtype=dtype)
    cdef matrix2d_t Ys = np.zeros((obs_dim, T), dtype=dtype)
    #
    cdef index_t[:] obs_dim_new = np.zeros((1), dtype="i8")
    cdef index_t[:] lat_dim_new = np.zeros((1), dtype="i8")
    cdef index_t[:] T_new = np.zeros((1), dtype="i8")
    cdef matrix2d_t F_new  = np.zeros((lat_dim, lat_dim), dtype=dtype)
    cdef matrix2d_t H_new  = np.zeros((obs_dim, lat_dim), dtype=dtype)
    cdef matrix2d_t Q_new  = np.zeros((lat_dim, lat_dim), dtype=dtype)
    cdef matrix2d_t R_new  = np.zeros((obs_dim, obs_dim), dtype=dtype)
    cdef matrix2d_t X0_new = np.zeros((lat_dim, 1), dtype=dtype)
    cdef matrix2d_t P0_new = np.zeros((lat_dim, lat_dim), dtype=dtype)
    #

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
            penalty_roughness_Y,
            random_spread)
    
    
    _kalman_smoother_results(
        tovector2(Xp), tovector3(Pp), tovector2(Yp),
        tovector2(Xf), tovector3(Pf), tovector2(Yf),
        tovector2(Xs), tovector3(Ps), tovector2(Ys),
        kalman_smoother)
    
    _kalman_smoother_parameters(
        tovector1_i(obs_dim_new), tovector1_i(lat_dim_new), tovector1_i(T_new),
        tovector2(F_new), tovector2(H_new), tovector2(Q_new),
        tovector2(R_new), tovector2(X0_new), tovector2(P0_new),
        kalman_smoother)
    
    if return_details:
        return (
            np.array(F_new), np.array(H_new), np.array(Q_new),
            np.array(R_new), np.array(X0_new), np.array(P0_new),
            #
            np.array(Xp), np.array(Pp), np.array(Yp),
            np.array(Xf), np.array(Pf), np.array(Yf),
            np.array(Xs), np.array(Ps), np.array(Ys),
            np.array(loglikelihood_record)
        )
    return (np.array(Xs), np.array(Ys))

cpdef estimate(str type_of_estimator,
    str estimates,
    np.ndarray Y,
    index_t obs_dim=-1,
    index_t lat_dim=-1,
    index_t T=-1,
    np.ndarray F=None, np.ndarray H=None, 
    np.ndarray Q=None, np.ndarray R=None, 
    np.ndarray X0=None, np.ndarray P0=None,
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
    double_t random_spread=0.5,
    int max_length_loglikelihood=1000,
    bool return_details=False):
    Y = np.ascontiguousarray(Y).astype("f8").copy()
    F = np.ascontiguousarray(F).astype("f8").copy() if F is not None else np.ones((lat_dim, lat_dim))
    H = np.ascontiguousarray(H).astype("f8").copy() if H is not None else np.ones((obs_dim, lat_dim))
    Q = np.ascontiguousarray(Q).astype("f8").copy() if Q is not None else 0.1 * np.eye(lat_dim, lat_dim)
    R = np.ascontiguousarray(R).astype("f8").copy() if R is not None else 0.1 * np.eye(obs_dim, obs_dim)
    X0 = np.ascontiguousarray(X0).astype("f8").copy() if X0 is not None else np.ones((lat_dim, 1))
    P0 = np.ascontiguousarray(P0).astype("f8").copy() if P0 is not None else 0.1 * np.eye(lat_dim, lat_dim)
    return _estimate_ssm_p(
        type_of_estimator, estimates, Y,
        obs_dim, lat_dim, T,
        F,
        H,
        Q,
        R,
        X0,
        P0,
        min_iterations,
        max_iterations, 
        min_improvement,
        sample_size,
        population_size,
        penalty_low_variance_Q,
        penalty_low_variance_R,
        penalty_low_variance_P0,
        penalty_low_std_mean_ratio,
        penalty_inestable_system,
        penalty_mse,
        penalty_roughness_X,
        penalty_roughness_Y,
        random_spread,
        max_length_loglikelihood,
        return_details)


cpdef tuple performance_of_parameters(
                        np.ndarray Y,
                        index_t obs_dim,
                        index_t lat_dim,
                        index_t T,
                        np.ndarray F, np.ndarray H,
                        np.ndarray Q, np.ndarray R,
                        np.ndarray X0, np.ndarray P0
):
    cdef str dtype = "f8"
    cdef matrix2d_t _Y = np.ascontiguousarray(np.array(Y, dtype=dtype)).copy() #Fix np issue with C-type arrays
    cdef matrix2d_t _F = np.ascontiguousarray(np.array(F, dtype=dtype)).copy()
    cdef matrix2d_t _H = np.ascontiguousarray(np.array(H, dtype=dtype)).copy()
    cdef matrix2d_t _Q = np.ascontiguousarray(np.array(Q, dtype=dtype)).copy()
    cdef matrix2d_t _R = np.ascontiguousarray(np.array(R, dtype=dtype)).copy()
    cdef matrix2d_t _X0 = np.ascontiguousarray(np.array(X0, dtype=dtype)).copy()
    cdef matrix2d_t _P0 = np.ascontiguousarray(np.array(P0, dtype=dtype)).copy()
    cdef matrix1d_t loglikelihood = np.zeros(1, dtype=dtype)
    cdef matrix1d_t low_std_to_mean_penalty = np.zeros(1, dtype=dtype)
    cdef matrix1d_t low_variance_Q_penalty = np.zeros(1, dtype=dtype)
    cdef matrix1d_t low_variance_R_penalty = np.zeros(1, dtype=dtype)
    cdef matrix1d_t low_variance_P0_penalty = np.zeros(1, dtype=dtype)
    cdef matrix1d_t system_inestability_penalty = np.zeros(1, dtype=dtype)
    cdef matrix1d_t mean_squared_error_penalty = np.zeros(1, dtype=dtype)
    cdef matrix1d_t roughness_X_penalty = np.zeros(1, dtype=dtype)
    cdef matrix1d_t roughness_Y_penalty = np.zeros(1, dtype=dtype)
    _performance_of_parameters(
                tovector1(loglikelihood), 
                tovector1(low_std_to_mean_penalty), 
                tovector1(low_variance_Q_penalty), 
                tovector1(low_variance_R_penalty), 
                tovector1(low_variance_P0_penalty), 
                tovector1(system_inestability_penalty), 
                tovector1(mean_squared_error_penalty), 
                tovector1(roughness_X_penalty), 
                tovector1(roughness_Y_penalty), 
                obs_dim, lat_dim, T,
                tovector2(_Y),
                tovector2(_F), tovector2(_H), 
                tovector2(_Q), tovector2(_R), 
                tovector2(_X0), tovector2(_P0) )
    return (
        loglikelihood[0], 
        low_std_to_mean_penalty[0], 
        low_variance_Q_penalty[0], 
        low_variance_R_penalty[0], 
        low_variance_P0_penalty[0], 
        system_inestability_penalty[0], 
        mean_squared_error_penalty[0], 
        roughness_X_penalty[0], 
        roughness_Y_penalty[0]
    )