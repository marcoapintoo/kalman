#pragma once
#include <iostream>
#include <typeinfo>
#include <string>
#include <sstream>
#include <ctime>
#include <cstdio>
#include <armadillo>

using namespace std;
using namespace arma;

///////////////////////////////////////////////////////////////////////////
///  Macros definition
///////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
    #define ASSERT(Expr, Msg) \
        __assert(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
    #define ASSERT(Expr, Msg) ;
#endif

void __assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if(!expr){
        std::cerr << "Assert failed:\t" << msg << "\n"
                  << "Expected:\t" << expr_str << "\n"
                  << "Source:\t\t" << file << ", line " << line << "\n";
        std::abort();
    }
}

// Simple snippet for translation from Python prototype XD
#define for_range(i, i0, i1)  for(index_t i = i0; i < i1; i++)

///////////////////////////////////////////////////////////////////////////
///  Simple test framework
///////////////////////////////////////////////////////////////////////////
#define EXECUTE_TEST(name) \
    {\
    clock_t begin = clock(); \
    try{\
        std::cerr << "[[TEST CASE " << #name << "]]" << std::endl; \
        test_##name(); \
        std::cerr << "[[OK: "; \
    }catch(...){ \
        std::cerr << "[[FAILED: "; \
    }\
    clock_t end = clock(); \
    std::cerr.precision(3);\
    std::cerr << (double(end - begin) / CLOCKS_PER_SEC);\
    std::cerr << " secs.]]" << std::endl; \
    std::cerr << std::endl; \
    }
//std::cerr << "///////////////////////////////////////////////////////////////////////////" << std::endl; \


namespace SSM::TimeInvariant {
    ///////////////////////////////////////////////////////////////////////////
    ///  Basic typedef
    ///////////////////////////////////////////////////////////////////////////
    typedef long int_t;
    typedef double double_t;
    typedef unsigned long long index_t;
    typedef Col<double_t> matrix1d_t;
    typedef Mat<double_t> matrix2d_t;
    typedef Cube<double_t> matrix3d_t;

    static matrix1d_t empty_matrix1d;
    static matrix2d_t empty_matrix2d;
    static matrix3d_t empty_matrix3d;

    struct BaseKalmanSmoother{
        virtual double_t loglikelihood(){return 0.0;}
        matrix2d_t empty;
        virtual matrix2d_t& X0(){return empty;}
        virtual matrix2d_t& P0(){return empty;}
        virtual matrix2d_t& Q(){return empty;}
        virtual matrix2d_t& R(){return empty;}
        virtual matrix2d_t& F(){return empty;}
        virtual matrix2d_t& H(){return empty;}
        virtual matrix2d_t& Xs(){return empty;}
        virtual matrix2d_t& Ys(){return empty;}
    };

    ///////////////////////////////////////////////////////////////////////////
    ///  Math cross-platform functions
    ///////////////////////////////////////////////////////////////////////////

    inline bool is_none(matrix1d_t& X){
        return addressof(X) == addressof(empty_matrix1d);
    }

    inline bool is_none(matrix2d_t& X){
        return addressof(X) == addressof(empty_matrix2d);
    }

    inline bool is_none(matrix3d_t& X){
        return addressof(X) == addressof(empty_matrix3d);
    }

    inline auto size(matrix1d_t& X){
        return arma::size((const matrix1d_t&) X);
    }
    inline auto size(matrix2d_t& X){
        return arma::size((const matrix2d_t&) X);
    }
    inline auto size(matrix3d_t& X){
        return arma::size((const matrix3d_t&) X);
    }

    inline index_t _nrows(const matrix2d_t& X){
        return X.n_rows;
    }
    
    inline index_t _ncols(const matrix2d_t& X){
        return X.n_cols;
    }
    
    inline index_t _nrows(const matrix3d_t& X){
        return X.n_rows;
    }
    
    inline index_t _ncols(const matrix3d_t& X){
        return X.n_cols;
    }
    
    inline index_t _nslices(const matrix3d_t& X){
        return X.n_slices;
    }

    inline matrix2d_t _inv(const matrix2d_t& X){
        return pinv(X);
    }

    inline matrix2d_t _create_noised_values(index_t L, index_t M){
        return randn(L, M);
    }

    inline matrix2d_t _create_noised_ones(index_t L, index_t M, double factor=0.5){
        return ones<matrix2d_t>(L, M) + factor * randn(L, M);
    }

    inline matrix2d_t _create_noised_zeros(index_t L, index_t M, double factor=0.5){
        return zeros<matrix2d_t>(L, M) + factor * randn(L, M);
    }

    inline matrix2d_t _create_noised_diag(index_t L, index_t M, double factor=0.5){
        return eye<matrix2d_t>(L, M) + factor * randn(L, M);
    }

    void test_create_noised_ones(){
        matrix2d_t m = _create_noised_ones(1, 1);
        ASSERT(abs(m(0, 0) - 1) < 0.1, "");
        m = _create_noised_ones(10, 10);
        for_range(i, 0, 10){
            for_range(j, 0, 10){
                ASSERT(abs(m(i, j) - 1) < 0.18, "");
            }
        }
    }

    void test_create_noised_zeros(){
        matrix2d_t m = _create_noised_zeros(1, 1);
        ASSERT(abs(m(0, 0) - 0) < 0.18, "");
        m = _create_noised_zeros(10, 10);
        for_range(i, 0, 10){
            for_range(j, 0, 10){
                ASSERT(abs(m(i, j) - 0) < 0.18, "");
            }
        }
    }

    void test_create_noised_diag(){
        matrix2d_t m = _create_noised_diag(1, 1);
        ASSERT(abs(m(0, 0) - 1) < 0.18, "");
        m = _create_noised_diag(10, 10);
        for_range(i, 0, 10){
            for_range(j, 0, 10){
                if(i == j){
                    ASSERT(abs(m(i, j) - 1) < 0.18, "");
                }else{
                    ASSERT(abs(m(i, j) - 0) < 0.18, "");
                }
            }
        }
    }

    inline matrix2d_t _sum_slices(const matrix3d_t& X){
        return sum(X, 2);
    }

    void test_sum_slices(){
        matrix3d_t X = zeros<matrix3d_t>(2, 2, 4);
        X.slice(0) = {{0, 1}, {2, 3}};
        X.slice(1) = {{0, 0}, {0, 1}};
        X.slice(2) = {{1, 1}, {0, -1}};
        X.slice(3) = {{2, 0}, {2, 0}};
        matrix2d_t Y = _sum_slices(X);
        ASSERT(Y(0, 0) == 3, "");
        ASSERT(Y(0, 1) == 2, "");
        ASSERT(Y(1, 0) == 4, "");
        ASSERT(Y(1, 1) == 2, "");
        X = zeros<matrix3d_t>(1, 1, 3);
        X(0, 0, 0) = 1;
        X(0, 0, 1) = 1;
        X(0, 0, 2) = 2;
        Y = _sum_slices(X);
        ASSERT(Y(0, 0) == 4, "");
    }

    inline matrix2d_t _sum_cols(const matrix2d_t& X){
        return sum(X, 1);
    }

    void test_sum_cols(){
        matrix2d_t X = {{0, 1, 2, 3}, {4, 5, 6, 7}};
        matrix2d_t Y = _sum_cols(X);
        ASSERT(_ncols(Y) == 1, "");
        ASSERT(_ncols(Y) == 2, "");
        ASSERT(Y(0, 0) == 6, "");
        ASSERT(Y(1, 0) == 22, "");
    }

    inline void _set_diag_values_positive(matrix2d_t& X){
        X.diag() = abs(X.diag());
    }

    inline void _subsample(/*out*/ index_t& i0, /*out*/ matrix2d_t& Ysampled, const matrix2d_t& Y, index_t sample_size){
        if(sample_size >= _ncols(Y)){
            i0 = 0;
            Ysampled = matrix2d_t(Y);
        }
        i0 = randi<index_t>(distr_param(0, _ncols(Y) - sample_size - 1));
        Ysampled = matrix2d_t(Y.cols(i0, i0 + sample_size));
    }

    inline matrix2d_t _dot(initializer_list<matrix2d_t> vars){
        matrix2d_t p;
        int i = 0;
        for(matrix2d_t v: vars){
            if(i == 0){
                p = v;
            }else{
                p = p * v;
            }
            i++;
        }
    }

    inline matrix2d_t _dot(const matrix2d_t& v1){
        return v1;
    }

    inline matrix2d_t _dot(const matrix2d_t& v1, const matrix2d_t& v2){
        return v1 * v2;
    }

    inline matrix2d_t _dot(const matrix2d_t& v1, const matrix2d_t& v2, const matrix2d_t& v3){
        return v1 * v2 * v3;
    }

    inline matrix2d_t _dot(const matrix2d_t& v1, const matrix2d_t& v2, const matrix2d_t& v3, const matrix2d_t& v4){
        return v1 * v2 * v3 * v4;
    }

    void test_dot(){
        matrix2d_t A = {{1, 0}, {0, 1}}; 
        matrix2d_t B = colvec({1, 10});
        matrix2d_t C = rowvec({1, 10});
        matrix2d_t AB = _dot(A, B);
        matrix2d_t ABC = _dot(A, B, C);
        ASSERT(accu(AB) == 1, "");
        ASSERT(_ncols(AB) + _nrows(AB) == 3, "");
        ASSERT(accu(ABC) == (5 + 10 + 50 + 100), "");
    }

    void test_inv(){
        matrix2d_t X = {{5, 10}, {3, 6}};
        matrix2d_t X1 = _dot(_dot(X, _inv(X)), X);
        matrix2d_t X2 = _dot(X, _dot(_inv(X), X));
        for_range(i, 0, _nrows(X)){
            for_range(j, 0, _ncols(X)){
                ASSERT(abs(X1(i, j) - X(i, j)) < 0.1, "");
            }
        }
        X = {{5, 10, 2}, {3, 6, 10}};
        X1 = _dot(_dot(X, _inv(X)), X);
        X2 = _dot(X, _dot(_inv(X), X));
        for_range(i, 0, _nrows(X)){
            for_range(j, 0, _ncols(X)){
                ASSERT(abs(X1(i, j) - X(i, j)) < 0.1, "");
            }
        }
    }

    inline matrix2d_t _t(const matrix2d_t& X){
        return X.t();
    }

    inline matrix2d_t _row(const matrix2d_t& X, index_t k){
        return X.row(k);
    }

    void test_row(){
        matrix2d_t X = {{0, 1}, {1, 2}, {3, 4}};
        matrix2d_t Y = _row(X, 0);
        ASSERT(_ncols(Y) == 2, "");
        ASSERT(_nrows(Y) == 1, "");
    }

    inline matrix2d_t _col(matrix2d_t& X, index_t k){
        return X.col(k);
    }

    void test_col(){
        matrix2d_t X = {{0, 1}, {1, 2}, {3, 4}};
        matrix2d_t Y = _col(X, 0);
        ASSERT(_ncols(Y) == 1, "");
        ASSERT(_nrows(Y) == 3, "");
    }

    inline matrix2d_t _slice(matrix3d_t& X, index_t k){
        return X.slice(k);
    }

    void test_slice(){
        matrix3d_t X(3, 2, 1);
        X.slice(0) = {{0, 1}, {1, 2}, {3, 4}};
        matrix2d_t Y = _slice(X, 0);
        ASSERT(_ncols(Y) == 2, "");
        ASSERT(_nrows(Y) == 3, "");
    }

    inline void _set_row(matrix2d_t& X, index_t k, const matrix2d_t& v){
        X.row(k) = v;
    }

    inline void _set_col(matrix2d_t& X, index_t k, const matrix2d_t& v){
        X.col(k) = v;
    }
    
    inline void _set_slice(matrix3d_t& X, index_t k, const matrix2d_t& v){
        X.slice(k) = v;
    }
    
    inline matrix2d_t _one_matrix(index_t L, index_t M){
        return ones<matrix2d_t>(L, M);
    }
    
    inline matrix2d_t _zero_matrix(index_t L, index_t M){
        return zeros<matrix2d_t>(L, M);
    }
    
    inline matrix3d_t _zero_cube(index_t L, index_t M, index_t N){
        return zeros<matrix3d_t>(L, M, N);
    }
    
    inline matrix2d_t _diag_matrix(index_t L, index_t M){
        return eye<matrix2d_t>(L, M);
    }
    
    inline void _no_finite_to_zero(matrix2d_t& A){
        A.elem(find_nonfinite(A)).zeros();
    }
    
    inline matrix3d_t _head_slices(matrix3d_t& X){
        return matrix3d_t(X.head_slices(_nslices(X) - 1));
    }

    inline matrix3d_t _tail_slices(matrix3d_t& X){
        return matrix3d_t(X.tail_slices(_nslices(X) - 1));
    }

    inline matrix2d_t _head_cols(matrix2d_t& X){
        return matrix2d_t(X.head_cols(_ncols(X) - 1));
    }

    inline matrix2d_t _tail_cols(matrix2d_t& X){
        return matrix2d_t(X.tail_cols(_ncols(X) - 1));
    }


    ///////////////////////////////////////////////////////////////////////////
    ///  Stats helpers
    ///////////////////////////////////////////////////////////////////////////

    
    inline double _mvn_probability(const matrix2d_t& x, const matrix2d_t& mean, const matrix2d_t& cov){
        return exp(-0.5 * as_scalar(_dot(_t(x - mean), _inv(cov), (x - mean)))) / sqrt(2 * datum::pi * det(cov));
    }

    inline double _mvn_logprobability(const matrix2d_t& x, const matrix2d_t& mean, const matrix2d_t& cov){
        return (-0.5 * as_scalar(_dot(_t(x - mean), _inv(cov), (x - mean))) - 0.5 * log(2 * datum::pi) - 0.5 * log(det(cov)));
    }
    
    void test_mvn_probability(){
        matrix1d_t xs = linspace<matrix1d_t>(-5, 5, 101);
        double_t dxs = xs[1] - xs[0];
        double_t cdf = 0;
        for(double_t x: xs){
            matrix2d_t A = colvec({x});
            matrix2d_t B = colvec({0});
            matrix2d_t C = colvec({1});
            cdf += _mvn_probability(A, B, C) * dxs;
        }
        ASSERT(abs(cdf - 1) < 1e-3, "Bad integration!");
        xs = linspace<matrix1d_t>(-5, 5, 41);
        dxs = xs[1] - xs[0];
        dxs = (dxs * dxs* 0.4);
        cdf = 0;
        for(double_t x: xs){
            for(double_t y: xs){
                matrix2d_t A = colvec({{x}, {y}});
                matrix2d_t B = colvec({{0}, {0}});
                matrix2d_t C = {{1, 0}, {0, 1}};
                cdf += _mvn_probability(A, B, C) * dxs;
            }
        }
        ASSERT(abs(cdf - 1) < 1e-2, "Bad integration!");
    }
    
    void test_mvn_logprobability(){
        matrix1d_t xs = linspace<matrix1d_t>(-5, 5, 101);
        double_t dxs = xs[1] - xs[0];
        double_t cdf = 0;
        for(double_t x: xs){
            matrix2d_t A = colvec({x});
            matrix2d_t B = colvec({0});
            matrix2d_t C = colvec({1});
            cdf += exp(_mvn_logprobability(A, B, C)) * dxs;
        }
        ASSERT(abs(cdf - 1) < 1e-3, "Bad integration!");
        xs = linspace<matrix1d_t>(-5, 5, 41);
        dxs = xs[1] - xs[0];
        dxs = (dxs * dxs* 0.4);
        cdf = 0;
        for(double_t x: xs){
            for(double_t y: xs){
                matrix2d_t A = colvec({{x}, {y}});
                matrix2d_t B = colvec({{0}, {0}});
                matrix2d_t C = {{1, 0}, {0, 1}};
                cdf += exp(_mvn_logprobability(A, B, C)) * dxs;
            }
        }
        ASSERT(abs(cdf - 1) < 1e-2, "Bad integration!");
    }

    inline matrix2d_t _mvn_sample(matrix2d_t& mean, matrix2d_t& cov){
        return mvnrnd(mean, cov);
    }

    inline matrix2d_t _covariance_matrix_estimation(matrix2d_t& X){// # unbiased estimation
        matrix2d_t Q = -1 * _dot(_sum_cols(X), _t(_sum_cols(X))) / _ncols(X);
        for_range(t, 0, _ncols(X)){
            Q += _dot(_col(X, t), _t(_col(X, t)));
        }
        Q /= (_ncols(X) - 1);
        return Q;
    }

    void test_covariance_matrix_estimation(){
        matrix2d_t X = {{1, -2, 4}, {1, 5, 3}};
        matrix2d_t Y = _covariance_matrix_estimation(X);
        ASSERT(_ncols(Y) == 2, "");
        ASSERT(_nrows(Y) == 2, "");
        ASSERT(abs(Y(0, 0) -  9) < 0.01, "");
        ASSERT(abs(Y(0, 1) - -3) < 0.01, "");
        ASSERT(abs(Y(1, 0) - -3) < 0.01, "");
        ASSERT(abs(Y(1, 1) -  4) < 0.01, "");
        
        X = _zero_matrix(1, 4);
        X(0, 0) = 0;
        X(0, 1) = 1;
        X(0, 2) = 2;
        X(0, 3) = 3;
        Y = _covariance_matrix_estimation(X);
        ASSERT(_ncols(Y) == 1, "");
        ASSERT(_nrows(Y) == 1, "");
        ASSERT(abs(Y(0, 0) - 1.666) < 0.01, "");
    }

    ///////////////////////////////////////////////////////////////////////////
    ///  Roughness measurement
    ///////////////////////////////////////////////////////////////////////////
    
    inline double_t _measure_roughness_proposed(matrix1d_t& y0, index_t M=10){
        index_t cols = _nrows(y0);//M
        matrix2d_t y = reshape(y0.head_cols(cols * M), cols, M);
        matrix1d_t ystd = (y - mean(y, 1)) / stddev(y, 1);
        _no_finite_to_zero(ystd);
        ystd = vectorise(diff(ystd, 1, 1));
        return mean(mean(abs(ystd)));
    }

    inline double_t _measure_roughness(matrix2d_t& X, index_t M=10){
        double_t roughness = 0;
        for_range(k, 0, _nrows(X)){
            matrix1d_t Xk = _row(X, k);
            roughness +=  _measure_roughness_proposed(Xk, M);
        }
        return roughness/_nrows(X);
    }

    void test_measure_roughness(){
        matrix2d_t X1 = {
            {57, -21, 71, 45, -9, 52, -90, -13, 3, 99, 
            -52, -63, -64, -56, -35, -32, 83, 67, -65, 38, 
            -55, -1, -40, -93, -93, 57, 53, -64, -24, 32, 
            12, 83, -75, -48, 39, 87, 28, -17, 71, 78, 
            72, -57, 64, 80, -60, 67, -89, 14, 62, 60, 
            -4, -19, 18, -4, 10, 51, -51, 74, 2, 15, 
            -41, 71, -56, 99, -30, -95, -67, -44, 65, -46, 
            -21, -70, 95, 72, 41, 11, -98, -72, -70, 75, 
            -28, -2, -79, -21, -3, 86, 0, -58, 79, 14, -96, 
            -63, 84, -100, -55, 85, -94, 53, -49, 27},
            {12, 50, 66, 79, 52, 81, -62, 79, 64, -34, 
            -26, -88, 69, 58, -15, -19, -89, 23, -27, -82, 
            -16, 16, -46, 99, -48, 31, 61, 46, 2, -66, 
            -41, -32, -1, 31, 52, -56, -14, -26, 48, 30, 
            63, 28, 61, 56, -27, 32, -52, -86, 74, -55, 
            67, 97, -30, 24, -42, -67, -99, -40, 49, -19,
            -61, 55, 79, -47, -17, -52, 88, 78, -65, 90, 
            -95, 50, -96, -21, 73, -22, -5, -45, 55, 86, 
            -39, 79, -25, -79, 64, 38, 18, -8, 49, -48, 
            -93, -67, -88, -4, 15, -90, -67, 74, 68, -1}
        };
        matrix2d_t X = X1;
        ASSERT(abs(_measure_roughness(X) - 1.25) < 0.1, "");
        X = _one_matrix(_nrows(X1), _ncols(X1));
        ASSERT(abs(_measure_roughness(X) - 0) < 0.1, "");
        X = 10 * X1;
        ASSERT(abs(_measure_roughness(X) - 1.25) < 0.1, "");
        X = 1 + 0.0002 * X1;
        X.row(0).fill(10);
        ASSERT(abs(_measure_roughness(X) - 0.6) < 0.1, "");
    }

    double_t _mean_squared_error(matrix2d_t& Y, matrix2d_t& Ypred){
        return accu(pow(Y - Ypred, 2)) / (_ncols(Y) - _nrows(Y) - 1);
    }
    
    void test_mean_squared_error(){
        matrix2d_t Y = {
            { 1,  3,  5,  7,  9},
            {11, 13, 15, 17, 19}
        };
        matrix2d_t Ypred = {
            { 1,  3,  5,  7,  9},
            {11, 13, 15, 17, 19}
        };
        ASSERT(_mean_squared_error(Y, Ypred) == 0, "");
        Ypred = {
            { 0,  3,  5,  7,  9},
            {11, 13, 14, 17, 17}
        };
        ASSERT(_mean_squared_error(Y, Ypred) == 3.0, "");
    }


    ///////////////////////////////////////////////////////////////////////////
    ///  Invariant State-Space Models
    ///////////////////////////////////////////////////////////////////////////

    template<typename SSMParameters_t>
    BaseKalmanSmoother kalman_smoother_from_parameters(matrix2d_t& Y, SSMParameters_t& f);

    struct SSMParameters{
        matrix2d_t F;
        matrix2d_t H;
        matrix2d_t Q;
        matrix2d_t R;
        matrix2d_t X0;
        matrix2d_t P0;
        int_t obs_dim;
        int_t lat_dim;

        SSMParameters(): F(), H(), Q(), R(), X0(), P0(), obs_dim(-1), lat_dim(-1){
        }

        void set_dimensions(int_t obs_dim, int_t lat_dim){
            this->obs_dim = obs_dim;
            this->lat_dim = lat_dim;
        }

        int_t latent_signal_dimension(){
            return this->lat_dim;
        }

        int_t observable_signal_dimension(){
            return this->obs_dim;
        }

        void show(){
            cout << "[" << typeid(this).name() << ":" << (index_t) this << "]" << endl;
            cout << " Transition matrix: " << this->F << endl;
            cout << " Observation matrix: " << this->H << endl;
            cout << " Latent var-covar matrix: " << this->Q << endl;
            cout << " Observation var-covar matrix: " << this->R << endl;
            cout << " Initial latent signal: " << this->X0 << endl;
            cout << " Initial latent var-covar matrix: " << this->P0 << endl;
        }

        void random_initialize(bool init_F=true, bool init_H=true, bool init_Q=true, bool init_R=true, bool init_X0=true, bool init_P0=true){
            if(this->lat_dim < 0){
                throw logic_error("Latent signal dimension is unset!");
            }
            if( this->obs_dim < 0){
                throw logic_error("Observable signal dimension is unset!");
            }
            if (init_F) this->F = _create_noised_ones(this->lat_dim, this->lat_dim);
            if (init_Q) this->Q = _create_noised_diag(this->lat_dim, this->lat_dim);
            if (init_X0) this->X0 = _create_noised_values(this->lat_dim, 1);
            if (init_P0) this->P0 = _create_noised_diag(this->lat_dim, this->lat_dim);
            if (init_H) this->H = _create_noised_ones(this->obs_dim, this->lat_dim);
            if (init_R) this->R = _create_noised_diag(this->obs_dim, this->obs_dim);
        }
        
        void simulate(/*out*/ matrix2d_t& X, /*out*/ matrix2d_t& Y, index_t N, matrix2d_t& X0=empty_matrix2d, matrix2d_t& P0=empty_matrix2d){
            X = matrix2d_t();
            Y = matrix2d_t();
            if (N == 0){
                return;
            }
            if (is_none(X0)){
                X0 = this->X0;
            }
            if (is_none(P0)){
                P0 = this->P0;
            }
            matrix2d_t R0 = _zero_matrix(_ncols(this->R), 1);
            matrix2d_t Q0 = _zero_matrix(_ncols(this->Q), 1);
            matrix2d_t X1 = _mvn_sample(X0, P0);
            matrix2d_t Y1 = _dot(this->H, X1) + _mvn_sample(R0, this->R);
            // observation_sequence
            X = _zero_matrix(_ncols(this->Q), N);
            // hidden_state_sequence
            Y = _zero_matrix(_ncols(this->R), N);
            _set_col(X, 0, X1);
            _set_col(Y, 0, Y1);
            for_range(i, 1, N){
                X1 = _dot(this->F, X1) + _mvn_sample(Q0, this->Q);
                _set_col(X, i, X1);
                Y1 = _dot(this->H, X1) + _mvn_sample(R0, this->R);
                _set_col(Y, i, Y1);
            }
        }

        auto copy(){
            SSMParameters p;
            p.F = matrix2d_t(this->F);
            p.H = matrix2d_t(this->H);
            p.Q = matrix2d_t(this->Q);
            p.R = matrix2d_t(this->R);
            p.X0 = matrix2d_t(this->X0);
            p.P0 = matrix2d_t(this->P0);
            p.obs_dim = this->obs_dim;
            p.lat_dim = this->lat_dim;
            return p;
        }
        
        void copy_from(SSMParameters& p){
            this->F = matrix2d_t(p.F);
            this->H = matrix2d_t(p.H);
            this->Q = matrix2d_t(p.Q);
            this->R = matrix2d_t(p.R);
            this->X0 = matrix2d_t(p.X0);
            this->P0 = matrix2d_t(p.P0);
            this->obs_dim = p.obs_dim;
            this->lat_dim = p.lat_dim;
        }
        
        ///////////////////////////////////////////////////////////////////////
        // Evaluation
        ///////////////////////////////////////////////////////////////////////

        inline double_t _penalize_low_std_to_mean_ratio(matrix2d_t& X0, matrix2d_t& P0){
            /*
            We expect that the variance must be
            low compared with the mean:
            Mean      Std       penalty
                1         0            undefined
                1         1            100 = 100 * std/mean
                10        1            10 = 100 * std/mean
                20        1            5 = 100 * std/mean
                50        1            2 = 100 * std/mean
            */
            double_t mean_ = abs(mean(mean(X0)));
            double_t std = mean(mean(P0));
            if (abs(mean_) < 1e-3)
                return 0;
            return 100 * std/mean_;
        }
            
        inline double_t _penalize_low_variance(matrix2d_t& X){
            /*
            penalty RULE:
            0.0001          10
            0.001           1
            0.01            0.1
            0.1             0.01
            1               0.001
            10              0.0001
            */
            return 0.1/ ((X / (pow(mean(X, 1), 2))).max());
        }
        
        inline double_t _penalize_inestable_system(matrix2d_t& X){
            /*
            penalty
            eigenvalues of X    penalty
            1   1   1           ~ 27
            0.1 0.1 0.1         ~ 0.08
            0.5 0.9 0.5         ~ 8
            */
            matrix1d_t eigval;
            eig_sym(eigval, X);
            return sum(pow(abs(eigval), 2));
        }

        inline double_t _mean_squared_error(matrix2d_t& Y, matrix2d_t& Ys){
            return mean(pow(vectorise(Y) - vectorise(Ys), 2));
        }

        inline double_t _penalize_mean_squared_error(matrix2d_t& Y, matrix2d_t& Ys){
            double_t maxv = -1e10;
            for_range(i, 0, _nrows(Y)){
                matrix2d_t Yi = _row(Y, i);
                matrix2d_t Ysi = _row(Ys, i);
                double_t v = _mean_squared_error(Yi, Ysi);
                if(v > maxv){
                    maxv = v;
                }
            }
            return maxv;
        }
        
        inline double_t _penalize_roughness(matrix2d_t& X){
            return _measure_roughness(X);
        }

        
        std::string performance_parameters_line(matrix2d_t& Y){
            double_t loglikelihood;
            double_t low_std_to_mean_penalty;
            double_t low_variance_Q_penalty;
            double_t low_variance_R_penalty;
            double_t low_variance_P0_penalty;
            double_t system_inestability_penalty;
            double_t mean_squared_error_penalty;
            double_t roughness_X_penalty;
            double_t roughness_Y_penalty;
            stringstream ss;
            ss.precision(2);
            this->performance_parameters(
                loglikelihood,
                low_std_to_mean_penalty,
                low_variance_Q_penalty,
                low_variance_R_penalty,
                low_variance_P0_penalty,
                system_inestability_penalty,
                mean_squared_error_penalty,
                roughness_X_penalty,
                roughness_Y_penalty,
                Y);
            ss  << "LL: "
                << loglikelihood
                << " | std/avg: "
                << low_std_to_mean_penalty
                << " | lQ: "
                << low_variance_Q_penalty
                << " lR: "
                << low_variance_R_penalty
                << " lP0: "
                << low_variance_P0_penalty
                << " | inest: "
                << system_inestability_penalty
                << " mse: "
                << mean_squared_error_penalty
                << " | roX: "
                << roughness_X_penalty
                << " roY: "
                << roughness_Y_penalty 
                << "\n";
            return ss.str();
        }

        void performance_parameters(
                /*out*/ double_t& loglikelihood, 
                /*out*/ double_t& low_std_to_mean_penalty, 
                /*out*/ double_t& low_variance_Q_penalty, 
                /*out*/ double_t& low_variance_R_penalty, 
                /*out*/ double_t& low_variance_P0_penalty, 
                /*out*/ double_t& system_inestability_penalty, 
                /*out*/ double_t& mean_squared_error_penalty, 
                /*out*/ double_t& roughness_X_penalty, 
                /*out*/ double_t& roughness_Y_penalty, matrix2d_t& Y){
            BaseKalmanSmoother ks = kalman_smoother_from_parameters(Y, *this);
            loglikelihood = ks.loglikelihood();
            low_std_to_mean_penalty = this->_penalize_low_std_to_mean_ratio(ks.X0(), ks.P0());
            low_variance_Q_penalty = this->_penalize_low_variance(ks.Q());
            low_variance_R_penalty = this->_penalize_low_variance(ks.R());
            low_variance_P0_penalty = this->_penalize_low_variance(ks.P0());
            system_inestability_penalty = this->_penalize_inestable_system(ks.F());
            mean_squared_error_penalty = this->_penalize_mean_squared_error(Y, ks.Ys());
            roughness_X_penalty = this->_penalize_roughness(ks.Xs());
            roughness_Y_penalty = this->_penalize_roughness(Y);
        }
        
    };

    static SSMParameters empty_ssm_parameters;
    inline bool is_none(SSMParameters& X){
        return addressof(X) == addressof(empty_ssm_parameters);
    }
    
    SSMParameters _create_params_ones_kx1(const matrix1d_t& M, const matrix1d_t& K0){
        matrix2d_t K = K0;
        SSMParameters params;
        params.F = colvec({1-1e-10});
        params.H = K;
        params.Q = colvec({0.001});
        params.R = 0.001 * _create_noised_diag(_nrows(K), _nrows(K));
        params.X0 = M;
        params.P0 = colvec({0.001});
        params.set_dimensions(_nrows(K), 1);
        return params;
    }

    void test_simulations_params(){
        matrix1d_t a = colvec({-50});
        matrix1d_t b = colvec({10});
        matrix1d_t c = colvec({2});
        matrix1d_t d = colvec({-100, 100});
        {
            SSMParameters params = _create_params_ones_kx1(a, b);
            matrix2d_t x(1, 100);
            matrix2d_t y(1, 100);
            params.simulate(x, y, 100);
            ASSERT(abs(round(mean(_row(y, 0))) - -500)/100 < 0.1 * 500, "Failed simulation");
        }
        SSMParameters params = _create_params_ones_kx1(c, d);
        {
            matrix2d_t x(1, 1);
            matrix2d_t y(1, 1);
            params.simulate(x, y, 1);
            ASSERT(abs(round(mean(_row(y, 0))) - -200)/100 < 0.1 * 200, "Failed simulation");
            ASSERT(abs(round(mean(_row(y, 1))) - -200)/100 < 0.1 * 200, "Failed simulation");
        }
        matrix2d_t x(1, 100);
        matrix2d_t y(1, 100);
        params.simulate(x, y, 100);
        ASSERT(abs(round(mean(_row(y, 0))) - -200)/100 < 0.1 * 200, "Failed simulation");
        ASSERT(abs(round(mean(_row(y, 1))) - -200)/100 < 0.1 * 200, "Failed simulation");
    }

    struct SSMEstimated{
        matrix2d_t X;
        matrix2d_t Y;
        matrix3d_t P;
        matrix3d_t V;
        SSMEstimated(): X(), Y(), P(), V(){}
        
        matrix2d_t& signal(){return this->X;}

        matrix3d_t& variance(){return this->P;}
        
        matrix3d_t& autocovariance(){return this->V;}
        
        void init(index_t dimX, index_t dimY, index_t length, bool fill_ACV=false){
            this->X = _zero_matrix(dimX, length);
            this->Y = _zero_matrix(dimY, length);
            this->P = _zero_cube(dimX, dimX, length);
            if(fill_ACV){
                this->V = _zero_cube(dimX, dimX, length - 1);
            }
        }
    };

    matrix2d_t _predict_expected_ssm(matrix2d_t& H, matrix2d_t& Xpred){
        matrix2d_t Ypred = _zero_matrix(_nrows(H), _ncols(Xpred));
        for_range(t, 0, _ncols(Xpred)){
            _set_col(Ypred, t, _dot(H, _col(Xpred, t)));
        }
        return Ypred;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Kalman Filter
    ///////////////////////////////////////////////////////////////////////////

    /*
    X{t} = F X{t-1} + N(0, Q)
    Y{t} = H X{t} + N(0, R)
    */
    struct KalmanFilter: public BaseKalmanSmoother{
        SSMParameters parameters;
        SSMEstimated filtered_estimates;
        SSMEstimated predicted_estimates;
        matrix2d_t _Y;

        KalmanFilter():
            BaseKalmanSmoother(),
            parameters(), 
            filtered_estimates(), 
            predicted_estimates(), 
            _Y(empty_matrix2d){}
        
        index_t T(){ return _ncols(this->Y()); }
        
        index_t obs_dim(){ return this->parameters.obs_dim; }
        
        void set_obs_dim(index_t v){ this->parameters.obs_dim = v; }
        
        index_t lat_dim(){ return this->parameters.lat_dim; }
        
        void set_lat_dim(index_t v){ this->parameters.lat_dim = v; }

        matrix2d_t& F(){ return this->parameters.F; }

        void set_F(matrix2d_t& v){ this->parameters.F = v; }

        matrix2d_t& H(){ return this->parameters.H; }
        
        void set_H(matrix2d_t& v){ this->parameters.H = v; }

        matrix2d_t& Q(){ return this->parameters.Q; }
        
        void set_Q(matrix2d_t& v){ this->parameters.Q = v; }

        matrix2d_t& R(){ return this->parameters.R; }
        
        void set_R(matrix2d_t& v){ this->parameters.R = v; }

        matrix2d_t& X0(){ return this->parameters.X0; }
        
        void set_X0(matrix2d_t& v){ this->parameters.X0 = v; }

        matrix2d_t& P0(){ return this->parameters.P0; }
        
        void set_P0(matrix2d_t& v){ this->parameters.P0 = v; }

        matrix2d_t& Y(){ return this->_Y; }
        
        void set_Y(matrix2d_t& v){ this->_Y = v; }
        
        matrix2d_t& Xf(){ return this->filtered_estimates.X; }

        matrix3d_t& Pf(){ return this->filtered_estimates.P; }
        
        matrix2d_t& Yf(){ return this->filtered_estimates.Y; }

        matrix2d_t& Xp(){ return this->predicted_estimates.X; }
            
        matrix2d_t& Yp(){ return this->predicted_estimates.Y; }

        matrix3d_t& Pp(){ return this->predicted_estimates.P; }

        virtual double_t loglikelihood(){
            //https://pdfs.semanticscholar.org/6654/c13f556035c1ea9e7b6a7cf53d13c98af6e7.pdf
            double_t log_likelihood = 0;
            for_range(k, 1, this->T()){
                matrix2d_t Sigma_k = _dot(this->H(), _slice(this->Pf(), k-1), _t(this->H())) + this->R();
                double_t current_likelihood = _mvn_logprobability(_col(this->Y(), k), _dot(this->H(), _col(this->Xf(), k)), Sigma_k);
                if(is_finite(current_likelihood)){
                    log_likelihood += current_likelihood;
                }
            }
            return log_likelihood / (this->T() - 1);
        }

        void verify_parameters(){
            if(this->lat_dim() == 0){
                throw logic_error("Observation sequence has no samples");
            }
            if(!(
                _nrows(this->Y()) == _nrows(this->H()) &&
                //_ncols(this->R()) == _nrows(this->Q()) &&
                _ncols(this->R()) == _nrows(this->H())
            )){
                stringstream ss;
                ss << "There is no concordance in the dimension of observed signal. "
                   << "Y: " << size(this->Y())
                   << "H: " << size(this->H())
                   << "R: " << size(this->R())
                   << endl;
                throw logic_error(ss.str().c_str());
            }
            if(!(
                _nrows(this->P0()) == _ncols(this->P0()) &&
                _nrows(this->X0()) == _ncols(this->P0()) &&
                _nrows(this->X0()) == _ncols(this->H()) &&
                _nrows(this->F()) == _ncols(this->H()) &&
                _nrows(this->F()) == _ncols(this->Q()) &&
                _ncols(this->F()) == _nrows(this->F()) &&
                _ncols(this->Q()) == _nrows(this->Q())
            )){
                stringstream ss;
                ss << "There is no concordance in the dimension of latent signal. "
                   << "X0: " << size(this->X0())
                   << "P0: " << size(this->P0())
                   << "F: " << size(this->F())
                   << "H: " << size(this->H())
                   << "R: " << size(this->R())
                   << "Q: " << size(this->Q())
                   << endl;
                throw logic_error(ss.str().c_str());
            }
        }

        void filter(){
            this->verify_parameters();
            this->filtered_estimates.init(this->lat_dim(), this->obs_dim(), this->T());
            this->predicted_estimates.init(this->lat_dim(), this->obs_dim(), this->T());
            
            _set_col(this->Xp(), 0, this->X0());
            _set_slice(this->Pp(), 0, this->P0());

            index_t k = 0;
            // Kalman gain
            matrix2d_t G = _dot(_slice(this->Pp(), k), _t(this->H()), _inv(_dot(this->H(), _slice(this->Pp(), k), _t(this->H())) + this->R()));
            // State estimate update
            _set_col(this->Xf(), k, _col(this->Xp(), k) + _dot(G, _col(this->Y(), k) - _dot(this->H(), _col(this->Xp(), k))));
            // Error covariance update
            _set_slice(this->Pf(), k, _slice(this->Pp(), k) - _dot(G, this->H(), _slice(this->Pp(), k)));

            for_range(k, 1, this->T()){
                // State estimate propagation
                _set_col(this->Xp(), k, _dot(this->F(), _col(this->Xf(), k - 1)));
                // Error covariance propagation
                _set_slice(this->Pp(), k, _dot(this->F(), _slice(this->Pf(), k-1), _t(this->F())) + this->Q());
                // Kalman gain
                G = _dot(_slice(this->Pp(), k), _t(this->H()), _inv(_dot(this->H(), _slice(this->Pp(), k), _t(this->H())) + this->R()));
                // State estimate update
                _set_col(this->Xf(), k, _col(this->Xp(), k) + _dot(G, _col(this->Y(), k) - _dot(this->H(), _col(this->Xp(), k))));
                // Error covariance update
                _set_slice(this->Pf(), k, _slice(this->Pp(), k) - _dot(G, this->H(), _slice(this->Pp(), k)));
            }
            //
            this->predicted_estimates.Y = _predict_expected_ssm(this->H(), this->predicted_estimates.X);
            this->filtered_estimates.Y = _predict_expected_ssm(this->H(), this->filtered_estimates.X);
        }
    };

    void test_filter_1(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        KalmanFilter kf;
        kf.set_Y(y);
        kf.parameters = params;
        kf.filter();
        ASSERT((abs(mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean");
        ASSERT((abs(mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
    }

    // {{export}}
    KalmanFilter kalman_filter(matrix2d_t& Y, matrix2d_t& F, matrix2d_t& H, matrix2d_t& Q, matrix2d_t& R, matrix2d_t& X0, matrix2d_t& P0){
        KalmanFilter kf;
        kf.set_F(F);
        kf.set_H(H);
        kf.set_Q(Q);
        kf.set_R(R);
        kf.set_X0(X0);
        kf.set_P0(P0);
        kf.set_Y(Y);
        kf.set_obs_dim(_nrows(Y));
        kf.set_lat_dim(_nrows(X0));
        kf.filter();
        return kf;
    }

    KalmanFilter kalman_filter_from_parameters(matrix2d_t& Y, SSMParameters& params){
        KalmanFilter kf;
        kf.parameters = params;
        kf.set_Y(Y);
        kf.set_obs_dim(_nrows(Y));
        kf.set_lat_dim(_nrows(params.X0));
        kf.filter();
        return kf;
    }

    // {{export}}
    void kalman_filter_results(
        /*out*/ matrix2d_t& Xp, /*out*/ matrix3d_t& Pp, /*out*/ matrix2d_t& Yp, 
        /*out*/ matrix2d_t& Xf, /*out*/ matrix3d_t& Pf, /*out*/ matrix2d_t& Yf,
        KalmanFilter& kf
    ){
        Xp = kf.Xp();
        Pp = kf.Pp();
        Yp = kf.Yp();
        Xf = kf.Xf();
        Pf = kf.Pf();
        Yf = kf.Yf();
    }

    // {{export}}
    void kalman_filter_parameters(
        /*out*/ matrix2d_t& F, /*out*/ matrix2d_t& H, /*out*/ matrix2d_t& Q,
        /*out*/ matrix2d_t& R, /*out*/ matrix2d_t& X0, /*out*/ matrix2d_t& P0,
        KalmanFilter& kf){
        F = kf.F();
        H = kf.H();
        Q = kf.Q();
        R = kf.R();
        X0 = kf.X0();
        P0 = kf.P0();
    }

    void test_filter_2(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(1, 100);
        matrix2d_t y(1, 100);
        params.simulate(x, y, 100);
        KalmanFilter kf = kalman_filter(y, params.F, params.H, params.Q, params.R, params.X0, params.P0);
        ASSERT((abs(mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean");
        ASSERT((abs(mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        kf = kalman_filter_from_parameters(y, params);
        ASSERT((abs(mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean");
        ASSERT((abs(mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
    }

    ///////////////////////////////////////////////////////////////////////////
    // Kalman Smoother
    ///////////////////////////////////////////////////////////////////////////

    struct KalmanSmoother: public KalmanFilter{
        string type_of_likelihood;
        SSMEstimated smoothed_estimates;
        
        KalmanSmoother(): 
            KalmanFilter(), 
            type_of_likelihood("smooth"),
            smoothed_estimates(){}
        
        matrix2d_t& Xs(){ return this->smoothed_estimates.X; }
        
        matrix3d_t& Ps(){ return this->smoothed_estimates.P; }

        matrix2d_t& Ys(){ return this->smoothed_estimates.Y; }

        matrix3d_t& Cs(){ return this->smoothed_estimates.V; }

        double_t loglikelihood_smooth(){
            double_t log_likelihood = 0;
            for_range(k, 1, this->T()){
                matrix2d_t Sigma_k = _dot(this->H(), _slice(this->Ps(), k-1), _t(this->H())) + this->R();
                double_t current_likelihood = _mvn_logprobability(_col(this->Y(), k), _dot(this->H(), _col(this->Xs(), k)), Sigma_k);
                if(is_finite(current_likelihood)){
                    log_likelihood += current_likelihood;
                }
            }
            return log_likelihood / (this->T() - 1);
        }
        
        double_t loglikelihood_filter(){
            //http://support.sas.com/documentation/cdl/en/imlug/65547/HTML/default/viewer.htm//imlug_timeseriesexpls_sect035.htm
            //Why this is better:
            //https://stats.stackexchange.com/questions/296598/why-is-the-likelihood-in-kalman-filter-computed-using-filter-results-instead-of
            double_t log_likelihood = 0;
            for_range(k, 1, this->T()){
                matrix2d_t Sigma_k = _dot(this->H(), _slice(this->Pf(), k-1), _t(this->H())) + this->R();
                double_t current_likelihood = _mvn_logprobability(_col(this->Y(), k), _dot(this->H(), _col(this->Xf(), k)), Sigma_k);
                if(is_finite(current_likelihood)){
                    log_likelihood += current_likelihood;
                }
            }
            return log_likelihood / (this->T() - 1);
        }
        
        double_t loglikelihood_qfunction(){
            double_t log_likelihood = 0;//_mvn_logprobability(this->X0(), this->X0(), this->P0()) <= 0
            for_range(k, 1, this->T()){
                double_t current_likelihood = _mvn_logprobability(_col(this->Xs(), k), _dot(this->F(), _col(this->Xs(), k - 1)), this->Q());
                if(is_finite(current_likelihood)){ //Temporal patch
                    log_likelihood += current_likelihood;
                }
            }
            for_range(k, 0, this->T()){
                double_t current_likelihood = _mvn_logprobability(_col(this->Y(), k), _dot(this->H(), _col(this->Xs(), k)), this->R());
                if(is_finite(current_likelihood)){
                    log_likelihood += current_likelihood;
                }
            }
            return log_likelihood / (this->T() - 1);
        }
        
        double_t loglikelihood(){
            if(this->type_of_likelihood == "filter")
                return this->loglikelihood_filter();
            if(this->type_of_likelihood == "smooth")
                return this->loglikelihood_filter();
            if(this->type_of_likelihood == "function-q")
                return this->loglikelihood_filter();
            throw logic_error("Wrong loglikelihood type!");
        }
        
        void smooth(bool filter=true){
            if(filter){
                this->filter();
            }
            this->smoothed_estimates.init(this->lat_dim(), this->obs_dim(), this->T(), true);
            
            index_t k = this->T() - 1;
            _set_col(this->Xs(), k, _col(this->Xf(), k));
            _set_slice(this->Ps(), k, _slice(this->Pf(), k));
            k -= 1;
            matrix2d_t A_prev;
            while(k >= 0){
                matrix2d_t A = _dot(_slice(this->Pf(), k), _t(this->F()), _inv(_slice(this->Pp(), k + 1)));
                _no_finite_to_zero(A);
                _set_slice(this->Ps(), k, _slice(this->Pf(), k) - _dot(A, _slice(this->Ps(), k + 1) - _slice(this->Pf(), k + 1), _t(A))); //Ghahramani
                _set_col(this->Xs(), k, _col(this->Xf(), k) + _dot(A, _col(this->Xs(), k + 1) - _col(this->Xp(), k + 1)));
                if(k == this->T() - 2){
                    matrix2d_t G = _dot(_slice(this->Pp(), k + 1), _t(this->H()), _inv(_dot(this->H(), _slice(this->Pp(), k + 1), _t(this->H())) + this->R()));
                    _set_slice(this->Cs(), k, _dot(this->F(), _slice(this->Pf(), k)) - _dot(G, this->H(), this->F(), _slice(this->Pf(), k)));
                }else{
                    _set_slice(this->Cs(), k, _dot(_slice(this->Pf(), k + 1), _t(A)) + _dot(A_prev, _t(_slice(this->Cs(), k + 1)) - _dot(this->F(), _slice(this->Pf(), k + 1)), _t(A_prev)));
                }
                A_prev = A;
                k -= 1;
            }
            this->smoothed_estimates.Y = _predict_expected_ssm(this->H(), this->smoothed_estimates.X);
        }

    };


    void test_smoother_1(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        KalmanSmoother kf;
        kf.parameters = params;
        kf.set_Y(y);
        kf.smooth();
        ASSERT((abs(mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean");
        ASSERT((abs(mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean");
        ASSERT((abs(mean(kf.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }
        

    void test_smoother_2(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        KalmanFilter kf;
        kf.parameters = params;
        kf.set_Y(y);
        kf.filter();
        KalmanSmoother ks;
        ks.parameters = params;
        ks.set_Y(y);
        ks.smooth();
        ASSERT((abs(mean(ks.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean");
        ASSERT((abs(mean(ks.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean");
        ASSERT((abs(mean(ks.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean");
        //ASSERT(round(std(ks.Xp()), 2) >= round(std(ks.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(ks.Xf()), 2) >= round(std(ks.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    // {{export}}
    KalmanSmoother kalman_smoother(matrix2d_t& Y, matrix2d_t& F, matrix2d_t& H, matrix2d_t& Q, matrix2d_t& R, matrix2d_t& X0, matrix2d_t& P0){
        KalmanSmoother kf;
        kf.set_F(F);
        kf.set_H(H);
        kf.set_Q(Q);
        kf.set_R(R);
        kf.set_X0(X0);
        kf.set_P0(P0);
        kf.set_Y(Y);
        kf.set_obs_dim(_nrows(Y));
        kf.set_lat_dim(_nrows(X0));
        kf.smooth();
        return kf;
    }

    // it sends a reference of params, not a copy
    KalmanSmoother kalman_smoother_from_parameters(matrix2d_t& Y, SSMParameters& params){
        KalmanSmoother kf;
        kf.parameters = params;
        kf.set_Y(Y);
        kf.set_obs_dim(_nrows(Y));
        kf.set_lat_dim(_nrows(params.X0));
        kf.smooth();
        return kf;
    }

    // {{export}}
    void kalman_smoother_results(
        /*out*/ matrix2d_t& Xp, /*out*/ matrix3d_t& Pp, /*out*/ matrix2d_t& Yp, 
        /*out*/ matrix2d_t& Xf, /*out*/ matrix3d_t& Pf, /*out*/ matrix2d_t& Yf,
        /*out*/ matrix2d_t& Xs, /*out*/ matrix3d_t& Ps, /*out*/ matrix2d_t& Ys,
        KalmanSmoother& kf
    ){
        Xp = kf.Xp();
        Pp = kf.Pp();
        Yp = kf.Yp();
        Xf = kf.Xf();
        Pf = kf.Pf();
        Yf = kf.Yf();
        Xs = kf.Xs();
        Ps = kf.Ps();
        Ys = kf.Ys();
    }

    // {{export}}
    void kalman_smoother_parameters(
        /*out*/ matrix2d_t& F, /*out*/ matrix2d_t& H, /*out*/ matrix2d_t& Q,
        /*out*/ matrix2d_t& R, /*out*/ matrix2d_t& X0, /*out*/ matrix2d_t& P0,
        KalmanSmoother& kf)
    {
        F = kf.F();
        H = kf.H();
        Q = kf.Q();
        R = kf.R();
        X0 = kf.X0();
        P0 = kf.P0();
    }

    void test_smoother_3(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        KalmanSmoother kf = kalman_smoother(y, params.F, params.H, params.Q, params.R, params.X0, params.P0);
        ASSERT((abs(mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean");
        ASSERT((abs(mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean");
        ASSERT((abs(mean(kf.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
        kf = kalman_smoother_from_parameters(y, params);
        ASSERT((abs(mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean");
        ASSERT((abs(mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean");
        ASSERT((abs(mean(kf.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }


    ///////////////////////////////////////////////////////////////////////////
    ///  X
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    ///  X
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    ///  X
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    ///  X
    ///////////////////////////////////////////////////////////////////////////


}
