#pragma once
#include <iostream>
#include <typeinfo>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <functional>
#include <sstream>
#include <ctime>
#include <cstdio>
#include <armadillo>

#include <SSM/Utils.hpp>
using namespace std;
using namespace arma;

#define __inline__ inline
//#define __inline__

namespace SSM::TimeInvariant {
    ///////////////////////////////////////////////////////////////////////////
    ///  Basic typedef
    ///////////////////////////////////////////////////////////////////////////
    typedef long int_t;
    typedef double double_t;
    //typedef unsigned long long index_t;
    typedef long long index_t;
    typedef Col<double_t> matrix1d_t;
    typedef Mat<double_t> matrix2d_t;
    typedef Cube<double_t> matrix3d_t;
    typedef map<string, double_t> config_map;

    static matrix1d_t empty_matrix1d;
    static matrix2d_t empty_matrix2d;
    static matrix3d_t empty_matrix3d;

    static config_map empty_config_map;

    ///////////////////////////////////////////////////////////////////////////
    ///  Math cross-platform functions
    ///////////////////////////////////////////////////////////////////////////

    __inline__ bool is_none(const matrix1d_t& X){
        return addressof(X) == addressof(empty_matrix1d);
    }

    __inline__ bool is_none(const matrix2d_t& X){
        return addressof(X) == addressof(empty_matrix2d);
    }

    __inline__ bool is_none(const matrix3d_t& X){
        return addressof(X) == addressof(empty_matrix3d);
    }

    __inline__ auto size(matrix1d_t& X){
        return arma::size((const matrix1d_t&) X);
    }
    __inline__ auto size(matrix2d_t& X){
        return arma::size((const matrix2d_t&) X);
    }
    __inline__ auto size(matrix3d_t& X){
        return arma::size((const matrix3d_t&) X);
    }

    __inline__ index_t _nrows(const matrix2d_t& X){
        return X.n_rows;
    }
    
    __inline__ index_t _ncols(const matrix2d_t& X){
        return X.n_cols;
    }
    
    __inline__ index_t _nrows(const matrix3d_t& X){
        return X.n_rows;
    }
    
    __inline__ index_t _ncols(const matrix3d_t& X){
        return X.n_cols;
    }
    
    __inline__ index_t _nslices(const matrix3d_t& X){
        return X.n_slices;
    }

    __inline__ matrix2d_t _inv(const matrix2d_t& X){
        return pinv(X);
    }

    __inline__ matrix2d_t _create_noised_values(index_t L, index_t M){
        return randn(L, M);
    }

    __inline__ matrix2d_t _create_noised_ones(index_t L, index_t M, double factor=0.5){
        return ones<matrix2d_t>(L, M) + factor * randn(L, M);
    }

    __inline__ matrix2d_t _create_noised_zeros(index_t L, index_t M, double factor=0.5){
        return zeros<matrix2d_t>(L, M) + factor * randn(L, M);
    }

    __inline__ matrix2d_t _create_noised_diag(index_t L, index_t M, double factor=0.5){
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

    __inline__ matrix2d_t _sum_slices(const matrix3d_t& X){
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

    __inline__ matrix2d_t _sum_cols(const matrix2d_t& X){
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

    __inline__ double_t mean2(const matrix2d_t& X){
        return accu(X) / (_nrows(X) + _ncols(X));
    }

    __inline__ double_t mean3(const matrix3d_t& X){
        return accu(X) / (_nrows(X) + _ncols(X) + _nslices(X));
    }

    __inline__ void _set_diag_values_positive(matrix2d_t& X){
        X.diag() = abs(X.diag());
    }

    __inline__ void _subsample(/*out*/ index_t& i0, /*out*/ matrix2d_t& Ysampled, const matrix2d_t& Y, index_t sample_size){
        if(sample_size >= _ncols(Y)){
            i0 = 0;
            Ysampled = Y;//matrix2d_t(Y);
            return;
        }
        i0 = randi<index_t>(
            distr_param(0, max((index_t) 0, _ncols(Y) - sample_size - 1))
        );
        Ysampled = matrix2d_t(Y.cols(i0, i0 + sample_size));
    }

    __inline__ matrix2d_t _dot(initializer_list<matrix2d_t> vars){
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

    __inline__ matrix2d_t _dot(const matrix2d_t& v1){
        return v1;
    }

    __inline__ matrix2d_t _dot(const matrix2d_t& v1, const matrix2d_t& v2){
        return v1 * v2;
    }

    __inline__ matrix2d_t _dot(const matrix2d_t& v1, const matrix2d_t& v2, const matrix2d_t& v3){
        return v1 * v2 * v3;
    }

    __inline__ matrix2d_t _dot(const matrix2d_t& v1, const matrix2d_t& v2, const matrix2d_t& v3, const matrix2d_t& v4){
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

    __inline__ matrix2d_t _t(const matrix2d_t& X){
        return X.t();
    }

    __inline__ matrix2d_t _row(const matrix2d_t& X, index_t k){
        return X.row(k);
    }

    void test_row(){
        matrix2d_t X = {{0, 1}, {1, 2}, {3, 4}};
        matrix2d_t Y = _row(X, 0);
        ASSERT(_ncols(Y) == 2, "");
        ASSERT(_nrows(Y) == 1, "");
    }

    __inline__ matrix2d_t _col(matrix2d_t& X, index_t k){
        return X.col(k);
    }

    void test_col(){
        matrix2d_t X = {{0, 1}, {1, 2}, {3, 4}};
        matrix2d_t Y = _col(X, 0);
        ASSERT(_ncols(Y) == 1, "");
        ASSERT(_nrows(Y) == 3, "");
    }

    __inline__ matrix2d_t _slice(matrix3d_t& X, index_t k){
        return X.slice(k);
    }

    void test_slice(){
        matrix3d_t X(3, 2, 1);
        X.slice(0) = {{0, 1}, {1, 2}, {3, 4}};
        matrix2d_t Y = _slice(X, 0);
        ASSERT(_ncols(Y) == 2, "");
        ASSERT(_nrows(Y) == 3, "");
    }

    __inline__ void _set_row(matrix2d_t& X, index_t k, const matrix2d_t& v){
        X.row(k) = v;
    }

    __inline__ void _set_col(matrix2d_t& X, index_t k, const matrix2d_t& v){
        X.col(k) = v;
    }
    
    __inline__ void _set_slice(matrix3d_t& X, index_t k, const matrix2d_t& v){
        X.slice(k) = v;
    }
    
    __inline__ matrix2d_t _one_matrix(index_t L, index_t M){
        return ones<matrix2d_t>(L, M);
    }
    
    __inline__ matrix2d_t _zero_matrix(index_t L, index_t M){
        return zeros<matrix2d_t>(L, M);
    }
    
    __inline__ matrix3d_t _zero_cube(index_t L, index_t M, index_t N){
        return zeros<matrix3d_t>(L, M, N);
    }
    
    __inline__ matrix2d_t _diag_matrix(index_t L, index_t M){
        return eye<matrix2d_t>(L, M);
    }
    
    __inline__ void _no_finite_to_zero(matrix2d_t& A){
        A.elem(find_nonfinite(A)).zeros();
    }
    
    __inline__ matrix3d_t _head_slices(matrix3d_t& X){
        return matrix3d_t(X.head_slices(_nslices(X) - 1));
    }

    __inline__ matrix3d_t _tail_slices(matrix3d_t& X){
        return matrix3d_t(X.tail_slices(_nslices(X) - 1));
    }

    __inline__ matrix2d_t _head_cols(matrix2d_t& X){
        return matrix2d_t(X.head_cols(_ncols(X) - 1));
    }

    __inline__ matrix2d_t _tail_cols(matrix2d_t& X){
        return matrix2d_t(X.tail_cols(_ncols(X) - 1));
    }


    ///////////////////////////////////////////////////////////////////////////
    ///  Stats helpers
    ///////////////////////////////////////////////////////////////////////////

    
    __inline__ double _mvn_probability(const matrix2d_t& x, const matrix2d_t& mean, const matrix2d_t& cov){
        return exp(-0.5 * as_scalar(_dot(_t(x - mean), _inv(cov), (x - mean)))) / sqrt(2 * datum::pi * det(cov));
    }

    __inline__ double _mvn_logprobability(const matrix2d_t& x, const matrix2d_t& mean, const matrix2d_t& cov){
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

    __inline__ matrix2d_t _mvn_sample(matrix2d_t& mean, matrix2d_t& cov){
        return mvnrnd(mean, cov);
    }

    __inline__ matrix2d_t _covariance_matrix_estimation(const matrix2d_t& X1){// # unbiased estimation
        matrix2d_t& X = const_cast<matrix2d_t&>(X1);
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
    
    __inline__ matrix2d_t standarized_signal(matrix2d_t& y){
        matrix2d_t y_mean = mean(y, 1);
        matrix2d_t y_stddev = stddev(y, 0, 1);
        matrix2d_t y_std(_nrows(y), _ncols(y));
        for_range(t, 0, _nrows(y)){
            _set_col(y_std, t, (_col(y_std, t) - y_mean) / y_stddev);
        }
        return y_std;
    }
    __inline__ double_t _measure_roughness_proposed(matrix2d_t& y0, index_t M=10){
        index_t cols = _nrows(y0);//M
        matrix2d_t y = reshape(y0.head_cols(cols * M), cols, M);
        matrix2d_t ystd = standarized_signal(y);
        _no_finite_to_zero(ystd);
        ystd = vectorise(diff(ystd, 1, 1));
        return mean(mean(abs(ystd)));
    }

    __inline__ double_t _measure_roughness(matrix2d_t& X, index_t M=10){
        double_t roughness = 0;
        for_range(k, 0, _nrows(X)){
            matrix2d_t Xk = _row(X, k);
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

    void _performance_parameters(
                /*out*/ double_t& loglikelihood, 
                /*out*/ double_t& low_std_to_mean_penalty, 
                /*out*/ double_t& low_variance_Q_penalty, 
                /*out*/ double_t& low_variance_R_penalty, 
                /*out*/ double_t& low_variance_P0_penalty, 
                /*out*/ double_t& system_inestability_penalty, 
                /*out*/ double_t& mean_squared_error_penalty, 
                /*out*/ double_t& roughness_X_penalty, 
                /*out*/ double_t& roughness_Y_penalty, 
                matrix2d_t& Y,
                void* _parameters);
    
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
        
        void copy_from(const SSMParameters& p){
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

        __inline__ double_t _penalize_low_std_to_mean_ratio(matrix2d_t& X0, matrix2d_t& P0){
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
            double_t mean_ = abs(mean2(X0));
            double_t std = mean2(P0);
            if (abs(mean_) < 1e-3)
                return 0;
            return 100 * std/mean_;
        }
            
        __inline__ double_t _penalize_low_variance(matrix2d_t& X){
            /*
            penalty RULE:
            0.0001          10
            0.001           1
            0.01            0.1
            0.1             0.01
            1               0.001
            10              0.0001
            */
            double_t maxval = -1e10;
            for_range(i, 0, _ncols(X)){
                double v = (_col(X, i) / pow(mean(X, 1), 2)).max();
                maxval = max(maxval, v);
            }
            return 0.1/maxval;
            //return 0.1/ ((X / (pow(mean(X, 1), 2))).max());
        }
        
        __inline__ double_t _penalize_inestable_system(matrix2d_t& X){
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

        __inline__ double_t _mean_squared_error(matrix2d_t& Y, matrix2d_t& Ys){
            //return mean(pow(vectorise(Y) - vectorise(Ys), 2));
            return mean(pow(vectorise(Y) - vectorise(Ys), 2)) / (
                mean(pow(vectorise(Y), 2)) *
                mean(pow(vectorise(Ys), 2))
                + 1e-100
            );
        }

        __inline__ double_t _penalize_mean_squared_error(matrix2d_t& Y, matrix2d_t& Ys){
            double_t maxv = -1e10;
            for_range(i, 0, _nrows(Y)){
                matrix2d_t Yi = _row(Y, i);
                matrix2d_t Ysi = _row(Ys, i);
                double_t v = _mean_squared_error(Yi, Ysi);
                if(v > maxv){
                    maxv = v;
                }
            }
            return sqrt(maxv);
        }
        
        __inline__ double_t _penalize_roughness(matrix2d_t& X){
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
            _performance_parameters(
                loglikelihood, 
                low_std_to_mean_penalty, 
                low_variance_Q_penalty, 
                low_variance_R_penalty, 
                low_variance_P0_penalty, 
                system_inestability_penalty, 
                mean_squared_error_penalty, 
                roughness_X_penalty, 
                roughness_Y_penalty,
                Y,
                (void*) this
            );
        }
        
    };

    static SSMParameters empty_ssm_parameters;
    __inline__ bool is_none(const SSMParameters& X){
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
    struct KalmanFilter{
        SSMParameters parameters;
        SSMEstimated filtered_estimates;
        SSMEstimated predicted_estimates;
        matrix2d_t _Y;

        KalmanFilter():
            parameters(), 
            filtered_estimates(), 
            predicted_estimates(), 
            _Y(empty_matrix2d){}
        
        index_t T(){ return _ncols(this->Y()); }
        
        index_t obs_dim(){ return this->parameters.obs_dim; }
        
        void set_obs_dim(index_t v){ this->parameters.obs_dim = v; }
        
        index_t lat_dim(){ return this->parameters.lat_dim; }
        
        void set_lat_dim(index_t v){ this->parameters.lat_dim = v; }

        virtual matrix2d_t& F(){ return this->parameters.F; }

        void set_F(matrix2d_t& v){ this->parameters.F = v; }

        virtual matrix2d_t& H(){ return this->parameters.H; }
        
        void set_H(matrix2d_t& v){ this->parameters.H = v; }

        virtual matrix2d_t& Q(){ return this->parameters.Q; }
        
        void set_Q(matrix2d_t& v){ this->parameters.Q = v; }

        virtual matrix2d_t& R(){ return this->parameters.R; }
        
        void set_R(matrix2d_t& v){ this->parameters.R = v; }

        virtual matrix2d_t& X0(){ return this->parameters.X0; }
        
        void set_X0(matrix2d_t& v){ this->parameters.X0 = v; }

        virtual matrix2d_t& P0(){ return this->parameters.P0; }
        
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

    const double_t MAX_LIKELIHOOD_ALLOWED = 100;

    struct KalmanSmoother: public KalmanFilter{
        
        string type_of_likelihood;
        SSMEstimated smoothed_estimates;
        
        KalmanSmoother(): 
            KalmanFilter(), 
            type_of_likelihood("smooth"),
            smoothed_estimates(){}
        
        virtual matrix2d_t& Xs(){ return this->smoothed_estimates.X; }
        
        matrix3d_t& Ps(){ return this->smoothed_estimates.P; }

        virtual matrix2d_t& Ys(){ return this->smoothed_estimates.Y; }

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
        
        virtual double_t loglikelihood(){
            double_t ll = 0;
            if(this->type_of_likelihood == "filter")
                ll = this->loglikelihood_filter();
            else if(this->type_of_likelihood == "smooth")
                ll = this->loglikelihood_smooth();
            else if(this->type_of_likelihood == "function-q")
                ll = this->loglikelihood_qfunction();
            else
                throw logic_error("Wrong loglikelihood type!");
            return min(ll, MAX_LIKELIHOOD_ALLOWED);
        }
        
        void smooth(bool filter=true){
            if(filter){
                this->filter();
            }
            this->smoothed_estimates.init(this->lat_dim(), this->obs_dim(), this->T(), true);

            int_t k = this->T() - 1;
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

    KalmanSmoother kalman_smoother_from_parameters(matrix2d_t& Y, const SSMParameters& params){
        return kalman_smoother_from_parameters(Y, const_cast<SSMParameters&>(params));
    }

    KalmanSmoother kalman_smoother_from_parameters(matrix2d_t& Y, SSMParameters& params){
        // Avoid memory leak in Win10 with the previous one.
        KalmanSmoother kf;
        kf.parameters = params;
        kf.set_Y(Y);
        kf.set_obs_dim(_nrows(Y));
        kf.set_lat_dim(_nrows(params.X0));
        kf.smooth();
        return kf;
    }
    void _performance_parameters(
                /*out*/ double_t& loglikelihood, 
                /*out*/ double_t& low_std_to_mean_penalty, 
                /*out*/ double_t& low_variance_Q_penalty, 
                /*out*/ double_t& low_variance_R_penalty, 
                /*out*/ double_t& low_variance_P0_penalty, 
                /*out*/ double_t& system_inestability_penalty, 
                /*out*/ double_t& mean_squared_error_penalty, 
                /*out*/ double_t& roughness_X_penalty, 
                /*out*/ double_t& roughness_Y_penalty, 
                matrix2d_t& Y,
                void* _parameters){
        SSMParameters& parameters = *((SSMParameters*)_parameters);
        KalmanSmoother smoother = kalman_smoother_from_parameters(Y, parameters);
        loglikelihood = smoother.loglikelihood();
        low_std_to_mean_penalty = parameters._penalize_low_std_to_mean_ratio(smoother.X0(), smoother.P0());
        low_variance_Q_penalty = parameters._penalize_low_variance(smoother.Q());
        low_variance_R_penalty = parameters._penalize_low_variance(smoother.R());
        low_variance_P0_penalty = parameters._penalize_low_variance(smoother.P0());
        system_inestability_penalty = parameters._penalize_inestable_system(smoother.F());
        mean_squared_error_penalty = parameters._penalize_mean_squared_error(Y, smoother.Ys());
        roughness_X_penalty = parameters._penalize_roughness(smoother.Xs());
        roughness_Y_penalty = parameters._penalize_roughness(Y);
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
    // EM SSM Estimator
    ///////////////////////////////////////////////////////////////////////////

    // https://emtiyaz.github.io/papers/TBME-00664-2005.R2-preprint.pdf

    struct ExpectationMaximizationEstimator{
        
        SSMParameters parameters;
        matrix2d_t Y;
        bool estimate_F;
        bool estimate_H;
        bool estimate_Q;
        bool estimate_R;
        bool estimate_X0;
        bool estimate_P0;
        vector<double_t> loglikelihood_record;
        int_t max_iterations;
        int_t min_iterations;
        double_t min_improvement;
        
        
        ExpectationMaximizationEstimator():
            parameters(empty_ssm_parameters),
            Y(),
            estimate_F(true),
            estimate_H(true),
            estimate_Q(true),
            estimate_R(true),
            estimate_X0(true),
            estimate_P0(true),
            loglikelihood_record(),
            max_iterations(10),
            min_iterations(1),
            min_improvement(0.01){}
        
        //
        // Trick in C++ for optional references
        // static double _dummy_foobar;
        // void foo(double &bar, double &foobar = _dummy_foobar)
        //
        void set_parameters(matrix2d_t& Y,
                           SSMParameters& parameters=empty_ssm_parameters,
                           bool est_F=true, bool est_H=true, bool est_Q=true, 
                           bool est_R=true, bool est_X0=true, bool est_P0=true, 
                           int_t lat_dim=-1)
        {
            if(!is_none(parameters)){
                this->parameters = parameters;
            }else{
                if(lat_dim < 0){
                    throw logic_error("lat_dim unset!");
                }
                this->parameters = SSMParameters();
                this->parameters.obs_dim = _ncols(Y);
                this->parameters.lat_dim = lat_dim;
                this->parameters.random_initialize();
            }
            this->Y = Y;
            this->estimate_F = est_F;
            this->estimate_H = est_H;
            this->estimate_Q = est_Q;
            this->estimate_R = est_R;
            this->estimate_X0 = est_X0;
            this->estimate_P0 = est_P0;
        }

        void estimation_iteration(){
            KalmanSmoother ks = kalman_smoother_from_parameters(this->Y, this->parameters);
            this->loglikelihood_record.push_back(ks.loglikelihood());
            index_t T = _ncols(ks.Y());
            index_t L = this->parameters.latent_signal_dimension();
            index_t O = this->parameters.observable_signal_dimension();
            matrix3d_t P = _zero_cube(L, L, T);
            matrix3d_t ACF = _zero_cube(L, L, T - 1);

            for_range(i, 0, T){
                _set_slice(P, i, _slice(ks.Ps(), i) + _dot(_col(ks.Xs(), i), _t(_col(ks.Xs(), i))));
                if(i < T - 1){
                    _set_slice(ACF, i, _slice(ks.Cs(), i) + _dot(_col(ks.Xs(), i + 1), _t(_col(ks.Xs(), i))));
                }
            }

            if(this->estimate_H){
                this->parameters.H = _inv(_sum_slices(P));
                matrix2d_t H1 = _zero_matrix(O, L);
                for_range(t, 0, T){
                    H1 += _dot(_col(ks.Y(), t), _t(_col(ks.Xs(), t)));
                    //print(" ", t, _dot(_col(ks.Y(), t), _t(_col(ks.Xs(), t))))
                }
                this->parameters.H = _dot(H1, this->parameters.H);
                //print("**",this->parameters.H)
            }
            if(this->estimate_R){
                this->parameters.R = _zero_matrix(O, O);
                for_range(t, 0, T){
                    this->parameters.R += _dot(_col(ks.Y(), t), _t(_col(ks.Y(), t))) - _dot(this->parameters.H, _col(ks.Xs(), t), _t(_col(ks.Y(), t)));
                }
                this->parameters.R /= T;
                // Fix math rounding errors
                _set_diag_values_positive(this->parameters.R);
            }
            if(this->estimate_F){
                this->parameters.F = _dot(_sum_slices(ACF), _inv(_sum_slices(_head_slices(P))));
            }
            if(this->estimate_Q){
                this->parameters.Q = _sum_slices(_tail_slices(P)) - _dot(this->parameters.F, _t(_sum_slices(ACF)));
                this->parameters.Q /= (T - 1);
                _set_diag_values_positive(this->parameters.Q);
            }
            if(this->estimate_X0){
                this->parameters.X0 = _col(ks.Xs(), 0);
            }
            if(this->estimate_P0){
                this->parameters.P0 = _slice(ks.Ps(), 0);
                _set_diag_values_positive(this->parameters.P0);
            }
            //this->parameters.show(); print("-"*80)
        }

        void estimate_parameters(){
            this->estimation_iteration();
            for_range(i, 0, this->max_iterations){
                this->estimation_iteration();
                double_t ll_1 = this->loglikelihood_record[this->loglikelihood_record.size() - 1];
                double_t ll_2 = this->loglikelihood_record[this->loglikelihood_record.size() - 2];
                bool unsufficient_increment = (ll_1 - ll_2) <= this->min_improvement;
                if(unsufficient_increment && i > this->min_iterations){
                    break;
                }
            }
            KalmanSmoother ks = kalman_smoother_from_parameters(this->Y, this->parameters);
            this->loglikelihood_record.push_back(ks.loglikelihood());
        }
        
        KalmanSmoother smoother(){
            return kalman_smoother_from_parameters(this->Y, this->parameters);
        }
    };

    void test_expectation_maximization_1(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        params.X0(0, 0) += 1;
        params.H(0, 0) -= 0.3;
        params.F(0, 0) -= 0.1;
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        ExpectationMaximizationEstimator kf;
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    void test_expectation_maximization_2(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        //params.X0{0, 0} += 1
        //params.H{0, 0} -= 0.3
        //params.F{0, 0} -= 0.1
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        ExpectationMaximizationEstimator kf;
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean(params.X0) - mean(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean(params.F) - mean(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean(params.H) - mean(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean(params.X0) - mean(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean(params.F) - mean(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean(params.H) - mean(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    // {{export}}
    void estimate_using_em(
            /*out*/ KalmanSmoother& ks,
            /*out*/ vector<double_t>& loglikelihood_record,
            matrix2d_t& Y,
            const string& estimates="",
            matrix2d_t& F0=empty_matrix2d,
            matrix2d_t& H0=empty_matrix2d, 
            matrix2d_t& Q0=empty_matrix2d, 
            matrix2d_t& R0=empty_matrix2d, 
            matrix2d_t& X00=empty_matrix2d, 
            matrix2d_t& P00=empty_matrix2d, 
            index_t min_iterations=1,
            index_t max_iterations=10, 
            double_t min_improvement=0.01,
            int_t lat_dim=-1)
    {
        ExpectationMaximizationEstimator estimator;
        estimator.Y = Y;
        estimator.estimate_F = contains(estimates, "F");
        estimator.estimate_H = contains(estimates, "H");
        estimator.estimate_Q = contains(estimates, "Q");
        estimator.estimate_R = contains(estimates, "R");
        estimator.estimate_X0 = contains(estimates, "X0");
        estimator.estimate_P0 = contains(estimates, "P0");
        estimator.parameters = SSMParameters();
        estimator.parameters.F = F0;
        if(!is_none(F0)){
            estimator.parameters.lat_dim = _nrows(F0);
        }
        estimator.parameters.H = H0;
        if(!is_none(H0)){
            estimator.parameters.lat_dim = _ncols(H0);
        }
        estimator.parameters.Q = Q0;
        if(!is_none(Q0)){
            estimator.parameters.lat_dim = _ncols(Q0);
        }
        estimator.parameters.R = R0;
        estimator.parameters.X0 = X00;
        if(!is_none(X00)){
            estimator.parameters.lat_dim = _nrows(X00);
        }
        estimator.parameters.P0 = P00;
        if(!is_none(P00)){
            estimator.parameters.lat_dim = _ncols(P00);
        }
        if(lat_dim > 0){
            estimator.parameters.lat_dim = lat_dim;
        }
        estimator.parameters.obs_dim = _nrows(Y);
        estimator.parameters.obs_dim = _nrows(Y);
        //
        estimator.min_iterations = min_iterations;
        estimator.max_iterations = max_iterations;
        estimator.min_improvement = min_improvement;
        //
        estimator.parameters.random_initialize(
            is_none(F0), is_none(H0), is_none(Q0), 
            is_none(R0), is_none(X00), is_none(P00));
        estimator.estimate_parameters();
        ks = estimator.smoother();
        ks.smooth();
        loglikelihood_record = estimator.loglikelihood_record;
    }

    void test_expectation_maximization_3(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        KalmanSmoother ks;
        vector<double_t> records;
        estimate_using_em(ks, records, y,
            "F H Q R X0 P0",
            params.F, params.H, params.Q, params.R, params.X0, params.P0,
            1, 10, 0.01, -1);
        SSMParameters neoparams = ks.parameters;
        //print(records)
        //neoparams.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
    }

    
    ///////////////////////////////////////////////////////////////////////////
    // PSO Heuristic SSM Estimator
    ///////////////////////////////////////////////////////////////////////////

    struct PSOHeuristicEstimatorParticle{
        SSMParameters params;
        double_t metric;
        double_t loglikelihood;
        SSMParameters best_params;
        double_t best_metric;
        double_t best_loglikelihood;
        
        double_t penalty_factor_low_variance_Q;
        double_t penalty_factor_low_variance_R;
        double_t penalty_factor_low_variance_P0;
        double_t penalty_factor_low_std_mean_ratio;
        double_t penalty_factor_inestable_system;
        double_t penalty_factor_mse;
        double_t penalty_factor_roughness_X;
        double_t penalty_factor_roughness_Y;

        bool estimate_F;
        bool estimate_H;
        bool estimate_Q;
        bool estimate_R;
        bool estimate_X0;
        bool estimate_P0;

        PSOHeuristicEstimatorParticle():
            params(),
            metric(-1e100),
            loglikelihood(-1e100),
            best_params(),
            best_metric(-1e150),
            best_loglikelihood(-1e150),
            
            penalty_factor_low_variance_Q(0.0),
            penalty_factor_low_variance_R(0.0),
            penalty_factor_low_variance_P0(0.0),
            penalty_factor_low_std_mean_ratio(0.0),
            penalty_factor_inestable_system(0.0),
            penalty_factor_mse(0.0),
            penalty_factor_roughness_X(0.0),
            penalty_factor_roughness_Y(0.0),

            estimate_F(true),
            estimate_H(true),
            estimate_Q(true),
            estimate_R(true),
            estimate_X0(true),
            estimate_P0(true){}
            
        // Assume that any param not null is fixed
        void init(index_t obs_dim, index_t lat_dim, matrix2d_t& F0=empty_matrix2d, matrix2d_t& H0=empty_matrix2d, matrix2d_t& Q0=empty_matrix2d, matrix2d_t& R0=empty_matrix2d, matrix2d_t& X00=empty_matrix2d, matrix3d_t& P00=empty_matrix3d){
            //this->params = SSMParameters()
            this->params.F = matrix2d_t(F0);
            //if(!is_none(F0)){
            //    this->params.lat_dim = _nrows(F0)
            this->params.H = matrix2d_t(H0);
            //if(!is_none(H0)){
            //    this->params.lat_dim = _ncols(H0)
            this->params.Q = matrix2d_t(Q0);
            //if(!is_none(Q0)){
            //    this->params.lat_dim = _ncols(Q0)
            this->params.R = matrix2d_t(R0);
            this->params.X0 = matrix2d_t(X00);
            //if(!is_none(X00)){
            //    this->params.lat_dim = _nrows(X00)
            this->params.P0 = matrix2d_t(P00);
            //if(!is_none(P00)){
            //    this->params.lat_dim = _ncols(P00)
            //if(!is_none(lat_dim)){
            //    this->params.lat_dim = lat_dim
            this->params.lat_dim = lat_dim;
            this->params.obs_dim = obs_dim;
            this->params.random_initialize(is_none(F0), is_none(H0), is_none(Q0), is_none(R0), is_none(X00), is_none(P00));
            this->best_params = this->params.copy();
        }

        void init_with_parameters(index_t obs_dim, const SSMParameters& parameters=empty_ssm_parameters,
            bool est_F=true, bool est_H=true, bool est_Q=true,
            bool est_R=true, bool est_X0=true, bool est_P0=true,
            index_t lat_dim=-1)
        {
            this->params.obs_dim = obs_dim;
            if(!is_none(parameters)){
                //this->params = parameters.copy()
                this->params.copy_from(parameters);
            }else{
                if(lat_dim < 0){
                    throw logic_error("lat_dim unset!");
                }
                //this->params = SSMParameters()
                this->params.obs_dim = obs_dim;
                this->params.lat_dim = lat_dim;
                this->params.random_initialize();
            }
            this->params.random_initialize(est_F, est_H, est_Q, est_R, est_X0, est_P0);
            this->best_params = this->params.copy();
        }
        
        void set_penalty_factors(double_t low_variance_Q=1,
                                double_t low_variance_R=1,
                                double_t low_variance_P0=1, 
                                double_t low_std_mean_ratio=1, 
                                double_t inestable_system=1, 
                                double_t mse=0.5, 
                                double_t roughness_X=1, 
                                double_t roughness_Y=1){
            this->penalty_factor_low_variance_Q = low_variance_Q;
            this->penalty_factor_low_variance_R = low_variance_R;
            this->penalty_factor_low_variance_P0 = low_variance_P0;
            this->penalty_factor_low_std_mean_ratio = low_std_mean_ratio;
            this->penalty_factor_inestable_system = inestable_system;
            this->penalty_factor_mse = mse;
            this->penalty_factor_roughness_X = roughness_X;
            this->penalty_factor_roughness_Y = roughness_Y;
        }
        
        virtual void evaluate(matrix2d_t& Y, index_t index=0){
            double_t loglikelihood;
            double_t low_std_to_mean_penalty;
            double_t low_variance_Q_penalty;
            double_t low_variance_R_penalty;
            double_t low_variance_P0_penalty;
            double_t system_inestability_penalty;
            double_t mean_squared_error_penalty;
            double_t roughness_X_penalty;
            double_t roughness_Y_penalty;
            this->params.performance_parameters(
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
            
            this->loglikelihood = loglikelihood;
            this->metric = loglikelihood;
            this->metric -= this->penalty_factor_low_std_mean_ratio * low_std_to_mean_penalty;
            this->metric -= this->penalty_factor_low_variance_Q * low_variance_Q_penalty;
            this->metric -= this->penalty_factor_low_variance_R * low_variance_R_penalty;
            this->metric -= this->penalty_factor_low_variance_P0 * low_variance_P0_penalty;
            this->metric -= this->penalty_factor_inestable_system * system_inestability_penalty;
            this->metric -= this->penalty_factor_mse * mean_squared_error_penalty;
            this->metric -= this->penalty_factor_roughness_X * roughness_X_penalty;
            this->metric -= this->penalty_factor_roughness_Y * roughness_Y_penalty;
            if(this->metric > this->best_metric){
                this->best_metric = this->metric;
                this->best_params = this->params.copy();
                this->best_loglikelihood = this->loglikelihood;
            }
        }
        
        void set_movable_params(bool est_F, bool est_H, bool est_Q, bool est_R, bool est_X0, bool est_P0){
            this->estimate_F = est_F;
            this->estimate_H = est_H;
            this->estimate_Q = est_Q;
            this->estimate_R = est_R;
            this->estimate_X0 = est_X0;
            this->estimate_P0 = est_P0;
            //!//////print("====>", this->estimate_F, this->estimate_H, this->estimate_Q, this->estimate_R, this->estimate_X0, this->estimate_P0,)
        }

        //def __move_fix(self, best_particle):
        void move(PSOHeuristicEstimatorParticle& best_particle){
            //print("==**>", this->estimate_F, this->estimate_H, this->estimate_Q, this->estimate_R, this->estimate_X0, this->estimate_P0,)
            double_t move_to_self_best = 2 * randu();
            double_t move_to_global_best = 2 * randu();
            if(this->estimate_F){
                this->params.F += move_to_self_best * (this->best_params.F - this->params.F) + move_to_global_best * (best_particle.best_params.F - this->params.F);
            }
            if(this->estimate_H){
                this->params.H += move_to_self_best * (this->best_params.H - this->params.H) + move_to_global_best * (best_particle.best_params.H - this->params.H);
            }
            if(this->estimate_Q){
                this->params.Q += move_to_self_best * (this->best_params.Q - this->params.Q) + move_to_global_best * (best_particle.best_params.Q - this->params.Q);
                this->params.Q = 0.5 * (this->params.Q + _t(this->params.Q));
                _set_diag_values_positive(this->params.Q);
            }
            if(this->estimate_R){
                this->params.R += move_to_self_best * (this->best_params.R - this->params.R) + move_to_global_best * (best_particle.best_params.R - this->params.R);
                this->params.R = 0.5 * (this->params.R + _t(this->params.R));
                _set_diag_values_positive(this->params.R);
            }
            if(this->estimate_X0){
                this->params.X0 += move_to_self_best * (this->best_params.X0 - this->params.X0) + move_to_global_best * (best_particle.best_params.X0 - this->params.X0);
            }
            if(this->estimate_P0){
                this->params.P0 += move_to_self_best * (this->best_params.P0 - this->params.P0) + move_to_global_best * (best_particle.best_params.P0 - this->params.P0);
                this->params.P0 = 0.5 * (this->params.P0 + _t(this->params.P0));
                _set_diag_values_positive(this->params.P0);
            }
        }
        
        void move_flexible(PSOHeuristicEstimatorParticle& best_particle){
            //print("==**>", this->estimate_F, this->estimate_H, this->estimate_Q, this->estimate_R, this->estimate_X0, this->estimate_P0,)
            double_t k1 = 2.0;
            double_t k2 = 2.0;
            if(this->estimate_F){
                this->params.F += k1 * randu() * (this->best_params.F - this->params.F) + k2 * randu() * (best_particle.best_params.F - this->params.F);
            }
            if(this->estimate_H){
                this->params.H += k1 * randu() * (this->best_params.H - this->params.H) + k2 * randu() * (best_particle.best_params.H - this->params.H);
            }
            if(this->estimate_Q){
                this->params.Q += k1 * randu() * (this->best_params.Q - this->params.Q) + k2 * randu() * (best_particle.best_params.Q - this->params.Q);
                this->params.Q = 0.5 * (this->params.Q + _t(this->params.Q));
                _set_diag_values_positive(this->params.Q);
            }
            if(this->estimate_R){
                this->params.R += k1 * randu() * (this->best_params.R - this->params.R) + k2 * randu() * (best_particle.best_params.R - this->params.R);
                this->params.R = 0.5 * (this->params.R + _t(this->params.R));
                _set_diag_values_positive(this->params.R);
            }
            if(this->estimate_X0){
                this->params.X0 += k1 * randu() * (this->best_params.X0 - this->params.X0) + k2 * randu() * (best_particle.best_params.X0 - this->params.X0);
            }
            if(this->estimate_P0){
                this->params.P0 += k1 * randu() * (this->best_params.P0 - this->params.P0) + k2 * randu() * (best_particle.best_params.P0 - this->params.P0);
                this->params.P0 = 0.5 * (this->params.P0 + _t(this->params.P0));
                _set_diag_values_positive(this->params.P0);
            }
        }
        
        void copy_best_from(PSOHeuristicEstimatorParticle& other, bool force_copy=false){
            if(other.best_metric > this->metric || force_copy){
                this->metric = other.best_metric;
                this->best_metric = other.best_metric;
                this->loglikelihood = other.loglikelihood;
                this->best_loglikelihood = other.best_loglikelihood;
                this->params.copy_from(other.best_params);
                this->best_params.copy_from(other.best_params);
            }
        }
    };
        

    struct PurePSOHeuristicEstimator{
        SSMParameters parameters;
        matrix2d_t Y;
        bool estimate_F;
        bool estimate_H;
        bool estimate_Q;
        bool estimate_R;
        bool estimate_X0;
        bool estimate_P0;
        vector<double_t> loglikelihood_record;
        int_t max_iterations;
        int_t min_iterations;
        double_t min_improvement;
        
        int_t sample_size;
        int_t population_size;
        vector<PSOHeuristicEstimatorParticle> particles;
        PSOHeuristicEstimatorParticle best_particle;

        double_t penalty_factor_low_variance_Q;
        double_t penalty_factor_low_variance_R;
        double_t penalty_factor_low_variance_P0;
        double_t penalty_factor_low_std_mean_ratio;
        double_t penalty_factor_inestable_system;
        double_t penalty_factor_mse;
        double_t penalty_factor_roughness_X;
        double_t penalty_factor_roughness_Y;

        PurePSOHeuristicEstimator():
            parameters(empty_ssm_parameters),
            Y(),
            estimate_F(true),
            estimate_H(true),
            estimate_Q(true),
            estimate_R(true),
            estimate_X0(true),
            estimate_P0(true),
            loglikelihood_record(),
            max_iterations(10),
            min_iterations(1),
            min_improvement(0.01),
            sample_size(30),
            population_size(50),
            particles(),
            best_particle(),
            penalty_factor_low_variance_Q(0.5),
            penalty_factor_low_variance_R(0.5),
            penalty_factor_low_variance_P0(0.5),
            penalty_factor_low_std_mean_ratio(0.5),
            penalty_factor_inestable_system(1),
            penalty_factor_mse(0.25),
            penalty_factor_roughness_X(2),
            penalty_factor_roughness_Y(2){}

        virtual PSOHeuristicEstimatorParticle _create_particle(){
            return PSOHeuristicEstimatorParticle();
        }

        void set_parameters(matrix2d_t& Y,
                           SSMParameters& parameters=empty_ssm_parameters,
                           bool est_F=true, bool est_H=true, bool est_Q=true, 
                           bool est_R=true, bool est_X0=true, bool est_P0=true, 
                           int_t lat_dim=-1)
        {
            //!//////print("****==>", this->estimate_F, this->estimate_H, this->estimate_Q, this->estimate_R, this->estimate_X0, this->estimate_P0,)
            //
            if(!is_none(parameters)){
                this->parameters = parameters;
            }else{
                this->parameters = SSMParameters();
                this->parameters.obs_dim = _nrows(Y);
                this->parameters.lat_dim = lat_dim;
                if(lat_dim < 0){
                    throw logic_error("lat_dim unset!");
                }
                this->parameters.random_initialize(est_F, est_H, est_Q, est_R, est_X0, est_P0);
            }
            //
            this->Y = Y;
            //this->sample_size = _ncols(Y)
            this->estimate_F = est_F;
            this->estimate_H = est_H;
            this->estimate_Q = est_Q;
            this->estimate_R = est_R;
            this->estimate_X0 = est_X0;
            this->estimate_P0 = est_P0;
            //!//////print("****==>", this->estimate_F, this->estimate_H, this->estimate_Q, this->estimate_R, this->estimate_X0, this->estimate_P0,)
            //
            this->best_particle = this->_create_particle();
            this->particles = {};
            //parameters.show()
            for_range(i, 0, this->population_size){
                this->particles.push_back(this->_create_particle());
                if(i == 0){
                    this->particles[i].init_with_parameters(_nrows(Y), parameters.copy(), false, false, false, false, false, false, parameters.lat_dim);
                }else{
                    this->particles[i].init_with_parameters(_nrows(Y), parameters.copy(), est_F, est_H, est_Q, est_R, est_X0, est_P0, lat_dim);
                }
                //this->particles{i}.params.show()
                this->particles[i].set_movable_params(est_F, est_H, est_Q, est_R, est_X0, est_P0);
                //!//////s = this->particles{i}
                //!//////print("****==>", s.estimate_F, s.estimate_H, s.estimate_Q, s.estimate_R, s.estimate_X0, s.estimate_P0,)
                //this->particles[i].params.show();
                
                /*
                auto OP = _create_particle();
                OP.init_with_parameters(4, empty_ssm_parameters, true, true, true, true, true, true, 2);
                OP.params.show();
                OP.evaluate(this->Y);
                double_t evald = OP.metric;
                OP.params.show();
                throw logic_error("x");
                */
                this->particles[i].set_penalty_factors(
                    this->penalty_factor_low_variance_Q,
                    this->penalty_factor_low_variance_R,
                    this->penalty_factor_low_variance_P0,
                    this->penalty_factor_low_std_mean_ratio,
                    this->penalty_factor_inestable_system,
                    this->penalty_factor_mse,
                    this->penalty_factor_roughness_X,
                    this->penalty_factor_roughness_Y
                );
                index_t y0;
                matrix2d_t subY;
                _subsample(y0, subY, this->Y, this->sample_size);
                //this->particles[i].params.show();
                this->particles[i].evaluate(subY, y0);
                ///
                this->best_particle.copy_best_from(this->particles[i], true);
                //////print(" **  ", this->particles{i}.metric)
                //////this->particles{i}.params.show()
                //print("."*80); this->particles{i}.params.show()
            }
            //
            //////
            //this->parameters.show()
            this->parameters.copy_from(this->best_particle.best_params);
            //////
            //this->parameters.show()
            //
        }

        void estimation_iteration_heuristic(){
            for_range(i, 0, this->population_size){
                //this->particles{i}.evaluate(this->Y)
                //print("-"*80); this->particles{i}.params.show()
                index_t y0;
                matrix2d_t subY;
                _subsample(y0, subY, this->Y, this->sample_size);
                try{
                    this->particles[i].evaluate(subY, y0);
                }catch(...){ //Avoid SVD convergence issues
                    this->particles[i].metric = -1e100;
                    this->particles[i].loglikelihood = -1e100;
                }
                this->best_particle.copy_best_from(this->particles[i]);
                this->particles[i].move(this->best_particle);
                //print("."*80); this->particles{i}.params.show()
            }
            this->loglikelihood_record.push_back(this->best_particle.best_loglikelihood);
            this->parameters.copy_from(this->best_particle.best_params);
        }

        void estimate_parameters(){
            this->estimation_iteration_heuristic();
            for_range(i, 0, this->max_iterations){
                this->estimation_iteration_heuristic();
                double_t ll_1 = this->loglikelihood_record[this->loglikelihood_record.size() - 1];
                double_t ll_2 = this->loglikelihood_record[this->loglikelihood_record.size() - 2];
                bool unsufficient_increment = (ll_1 - ll_2) <= this->min_improvement;
                if(unsufficient_increment && i > this->min_iterations){
                    break;
                }
            }
            KalmanSmoother ks = kalman_smoother_from_parameters(this->Y, this->parameters);
            this->loglikelihood_record.push_back(ks.loglikelihood());
        }
        
        KalmanSmoother smoother(){
            return kalman_smoother_from_parameters(this->Y, this->parameters);
        }

    };

    void test_pure_pso_1(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        params.X0(0, 0) += 1;
        params.H(0, 0) -= 0.3;
        params.F(0, 0) -= 0.1;
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        PurePSOHeuristicEstimator kf;
        //kf.penalty_factor_roughness_X = 1
        //kf.penalty_factor_roughness_Y = 1
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show();
        KalmanSmoother s = kf.smoother();
        s.smooth();
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    void test_pure_pso_2(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        //params.X0{0, 0} += 1
        //params.H{0, 0} -= 0.3
        //params.F{0, 0} -= 0.1
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        PurePSOHeuristicEstimator kf;
        //kf.penalty_factor_roughness_X = 1
        //kf.penalty_factor_roughness_Y = 1
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show()
        //params.show()
        //params_orig.show()
        KalmanSmoother s = kf.smoother();
        s.smooth();
        ASSERT((abs(mean2(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }


    // {{export}}
    void estimate_using_pso(
            /*out*/ KalmanSmoother& ks,
            /*out*/ vector<double_t>& loglikelihood_record,
            matrix2d_t& Y,
            const string& estimates="",
            matrix2d_t& F0=empty_matrix2d,
            matrix2d_t& H0=empty_matrix2d, 
            matrix2d_t& Q0=empty_matrix2d, 
            matrix2d_t& R0=empty_matrix2d, 
            matrix2d_t& X00=empty_matrix2d, 
            matrix2d_t& P00=empty_matrix2d, 
            index_t min_iterations=1,
            index_t max_iterations=10, 
            double_t min_improvement=0.01,
            int_t lat_dim=-1,
            index_t sample_size=30,
            index_t population_size=50, 
            const config_map& penalty_factors=empty_config_map
            )
    {
        PurePSOHeuristicEstimator estimator;
        estimator.Y = Y;
        estimator.estimate_F = contains(estimates, "F");
        estimator.estimate_H = contains(estimates, "H");
        estimator.estimate_Q = contains(estimates, "Q");
        estimator.estimate_R = contains(estimates, "R");
        estimator.estimate_X0 = contains(estimates, "X0");
        estimator.estimate_P0 = contains(estimates, "P0");
        
        estimator.sample_size = sample_size;
        estimator.population_size = population_size;
        estimator.min_iterations = min_iterations;
        estimator.max_iterations = max_iterations;
        estimator.min_improvement = min_improvement;
        
        estimator.penalty_factor_low_variance_Q = get(penalty_factors, "low_variance_Q", 0.5);
        estimator.penalty_factor_low_variance_R = get(penalty_factors, "low_variance_R", 0.5);
        estimator.penalty_factor_low_variance_P0 = get(penalty_factors, "low_variance_P0", 0.5);
        estimator.penalty_factor_low_std_mean_ratio = get(penalty_factors, "low_std_mean_ratio", 0.5);
        estimator.penalty_factor_inestable_system = get(penalty_factors, "inestable_system", 10);
        estimator.penalty_factor_mse = get(penalty_factors, "mse", 1e-5);
        estimator.penalty_factor_roughness_X = get(penalty_factors, "roughness_X", 0.5);
        estimator.penalty_factor_roughness_Y = get(penalty_factors, "roughness_Y", 0.5);
        
        SSMParameters parameters;
        parameters.F = F0;
        if(!is_none(F0)){
            parameters.lat_dim = _nrows(F0);
        }
        parameters.H = H0;
        if(!is_none(H0)){
            parameters.lat_dim = _ncols(H0);
        }
        parameters.Q = Q0;
        if(!is_none(Q0)){
            parameters.lat_dim = _ncols(Q0);
        }
        parameters.R = R0;
        parameters.X0 = X00;
        if(!is_none(X00)){
            parameters.lat_dim = _nrows(X00);
        }
        parameters.P0 = P00;
        if(!is_none(P00)){
            parameters.lat_dim = _ncols(P00);
        }
        if(lat_dim > 0){
            parameters.lat_dim = lat_dim;
        }
        parameters.obs_dim = _nrows(Y);
        parameters.obs_dim = _nrows(Y);
        parameters.random_initialize(is_none(F0), is_none(H0), is_none(Q0), is_none(R0), is_none(X00), is_none(P00));
        estimator.set_parameters(Y, parameters, contains(estimates, "F"), contains(estimates, "H"), contains(estimates, "Q"), contains(estimates, "R"), contains(estimates, "X0"), contains(estimates, "P0"), -1);
        estimator.estimate_parameters();
        ks = estimator.smoother();
        ks.smooth();
        loglikelihood_record = estimator.loglikelihood_record;
    }

    void test_pure_pso_3(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(1000, 1);
        matrix2d_t y(1000, 1);
        params.simulate(x, y, 1000);
        KalmanSmoother ks;///?
        vector<double_t> records;///?
        estimate_using_pso(ks, records, y,
            "F H Q R X0 P0",
            params.F, params.H, params.Q, params.R, params.X0, params.P0,
            5, 30, 0.01, -1,
            30, 50
            //penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
        );
        SSMParameters neoparams = ks.parameters;
        //print(records)
        //neoparams.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
    }

    void test_pure_pso_4(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(1000, 1);
        matrix2d_t y(1000, 1);
        params.simulate(x, y, 1000);
        KalmanSmoother ks;///?
        vector<double_t> records;///?
        estimate_using_pso(ks, records, y,
            "F H Q R X0 P0",
            params.F, params.H, params.Q, params.R, params.X0, params.P0,
            10, 20, 0.01, -1,
            10, 100
            //penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
        );
        SSMParameters neoparams = ks.parameters;
        //print(records)
        //neoparams.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
    }

    
    ///////////////////////////////////////////////////////////////////////////
    // Self improvable estimator
    ///////////////////////////////////////////////////////////////////////////
    struct SelfImproveHeuristicEstimatorParticle: public PSOHeuristicEstimatorParticle{

        SelfImproveHeuristicEstimatorParticle():
            PSOHeuristicEstimatorParticle(){}
        
        virtual void improve_params(matrix2d_t& Y, index_t index=0){
            //DO nothing;
        }
        
        virtual void evaluate(matrix2d_t& Y, index_t index=0){
            PSOHeuristicEstimatorParticle::evaluate(Y, index);
            double_t prev_metric = this->metric;
            SSMParameters prev_parameters = this->params.copy();
            this->improve_params(Y, index);
            PSOHeuristicEstimatorParticle::evaluate(Y, index);
            if(prev_metric > this->metric){
                this->metric = prev_metric;
                this->params.copy_from(prev_parameters);
            }
        }
    };
    ///////////////////////////////////////////////////////////////////////////
    // LSE Heuristic SSM Estimator
    ///////////////////////////////////////////////////////////////////////////
    struct LSEHeuristicEstimatorParticle: public SelfImproveHeuristicEstimatorParticle{

        LSEHeuristicEstimatorParticle():
            SelfImproveHeuristicEstimatorParticle(){}
        
        void improve_params(matrix2d_t& Y, index_t index=0){
            //this->params.show(); print()
            try{
                KalmanSmoother ks = kalman_smoother_from_parameters(Y, this->params.copy());
                matrix2d_t X = ks.Xs();
                if(this->estimate_H){
                    // Y = H X + N(0, R)
                    // <H> = Y * X' * inv(X * X')' 
                    this->params.H = _dot(Y, _t(X), _t(_inv(_dot(X, _t(X)))));
                }
                if(this->estimate_R){
                    // <R> = var(Y - H X)
                    this->params.R = _covariance_matrix_estimation(Y - _dot(this->params.H, X));
                }
                if(this->estimate_F){
                    // X{1..T} = F X{0..T-1} + N(0, R)
                    // <F> = X1 * X0' * inv(X0 * X0')'
                    matrix2d_t X0 = _head_cols(X);
                    matrix2d_t X1 = _tail_cols(X);
                    this->params.F = _dot(X1, _t(X0), _t(_inv(_dot(X0, _t(X0)))));
                }
                matrix2d_t inv_H = _inv(this->params.H);
                if(this->estimate_X0){
                    // Y{0} = H X{0} + N(0, R)
                    // <X{0}> = inv(H) Y{0}
                    matrix2d_t X0 = _col(X, 0);//_dot(inv_H, _col(Y, 0))
                    matrix2d_t inv_F = _inv(this->params.F);
                    // IMPROVE!
                    for_range(_, 0, index){
                        X0 = _dot(inv_F, X0);
                    }
                    this->params.X0 = X0;
                }
                if(this->estimate_P0){
                    // <X{0}> = inv(H) Y{0} => VAR<X{0}> = inv(H) var(Y{0}) inv(H)' 
                    // P{0} = inv(H) R inv(H)'
                    this->params.P0 = _dot(inv_H, this->params.R, _t(inv_H));
                }
                if(this->estimate_Q){
                    // <Q> = var(X1 - F X0)
                    matrix2d_t X0 = _head_cols(X);
                    matrix2d_t X1 = _tail_cols(X);
                    this->params.Q = _covariance_matrix_estimation(X1 - _dot(this->params.F, X0));
                }
                //this->params.show()
                //sys.exit(0)
            }catch(...){
            }
        }
    };

    struct LSEHeuristicEstimator: public PurePSOHeuristicEstimator{
        LSEHeuristicEstimator(): PurePSOHeuristicEstimator(){}

        virtual PSOHeuristicEstimatorParticle _create_particle(){
            return LSEHeuristicEstimatorParticle();
        }
    };

    void test_pure_lse_pso_1(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        params.X0(0, 0) += 1;
        params.H(0, 0) -= 0.3;
        params.F(0, 0) -= 0.1;
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        LSEHeuristicEstimator kf;
        //kf.penalty_factor_roughness_X = 1
        //kf.penalty_factor_roughness_Y = 1
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show();
        KalmanSmoother s = kf.smoother();
        s.smooth();
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    void test_pure_lse_pso_2(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        //params.X0{0, 0} += 1
        //params.H{0, 0} -= 0.3
        //params.F{0, 0} -= 0.1
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        LSEHeuristicEstimator kf;
        //kf.penalty_factor_roughness_X = 1
        //kf.penalty_factor_roughness_Y = 1
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show()
        //params.show()
        //params_orig.show()
        KalmanSmoother s = kf.smoother();
        s.smooth();
        ASSERT((abs(mean2(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    // {{export}}
    void estimate_using_lse_pso(
            /*out*/ KalmanSmoother& ks,
            /*out*/ vector<double_t>& loglikelihood_record,
            matrix2d_t& Y,
            const string& estimates="",
            matrix2d_t& F0=empty_matrix2d,
            matrix2d_t& H0=empty_matrix2d, 
            matrix2d_t& Q0=empty_matrix2d, 
            matrix2d_t& R0=empty_matrix2d, 
            matrix2d_t& X00=empty_matrix2d, 
            matrix2d_t& P00=empty_matrix2d, 
            index_t min_iterations=1,
            index_t max_iterations=10, 
            double_t min_improvement=0.01,
            int_t lat_dim=-1,
            index_t sample_size=30,
            index_t population_size=50, 
            const config_map& penalty_factors=empty_config_map
            )
    {
        LSEHeuristicEstimator estimator;
        estimator.Y = Y;
        estimator.estimate_F = contains(estimates, "F");
        estimator.estimate_H = contains(estimates, "H");
        estimator.estimate_Q = contains(estimates, "Q");
        estimator.estimate_R = contains(estimates, "R");
        estimator.estimate_X0 = contains(estimates, "X0");
        estimator.estimate_P0 = contains(estimates, "P0");
        
        estimator.sample_size = sample_size;
        estimator.population_size = population_size;
        estimator.min_iterations = min_iterations;
        estimator.max_iterations = max_iterations;
        estimator.min_improvement = min_improvement;
        
        estimator.penalty_factor_low_variance_Q = get(penalty_factors, "low_variance_Q", 0.5);
        estimator.penalty_factor_low_variance_R = get(penalty_factors, "low_variance_R", 0.5);
        estimator.penalty_factor_low_variance_P0 = get(penalty_factors, "low_variance_P0", 0.5);
        estimator.penalty_factor_low_std_mean_ratio = get(penalty_factors, "low_std_mean_ratio", 0.5);
        estimator.penalty_factor_inestable_system = get(penalty_factors, "inestable_system", 10);
        estimator.penalty_factor_mse = get(penalty_factors, "mse", 1e-5);
        estimator.penalty_factor_roughness_X = get(penalty_factors, "roughness_X", 0.5);
        estimator.penalty_factor_roughness_Y = get(penalty_factors, "roughness_Y", 0.5);
        
        SSMParameters parameters;
        parameters.F = F0;
        if(!is_none(F0)){
            parameters.lat_dim = _nrows(F0);
        }
        parameters.H = H0;
        if(!is_none(H0)){
            parameters.lat_dim = _ncols(H0);
        }
        parameters.Q = Q0;
        if(!is_none(Q0)){
            parameters.lat_dim = _ncols(Q0);
        }
        parameters.R = R0;
        parameters.X0 = X00;
        if(!is_none(X00)){
            parameters.lat_dim = _nrows(X00);
        }
        parameters.P0 = P00;
        if(!is_none(P00)){
            parameters.lat_dim = _ncols(P00);
        }
        if(lat_dim > 0){
            parameters.lat_dim = lat_dim;
        }
        parameters.obs_dim = _nrows(Y);
        parameters.obs_dim = _nrows(Y);
        parameters.random_initialize(is_none(F0), is_none(H0), is_none(Q0), is_none(R0), is_none(X00), is_none(P00));
        estimator.set_parameters(Y, parameters, contains(estimates, "F"), contains(estimates, "H"), contains(estimates, "Q"), contains(estimates, "R"), contains(estimates, "X0"), contains(estimates, "P0"), -1);
        estimator.estimate_parameters();
        ks = estimator.smoother();
        ks.smooth();
        loglikelihood_record = estimator.loglikelihood_record;
    }

    void test_pure_lse_pso_3(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(1000, 1);
        matrix2d_t y(1000, 1);
        params.simulate(x, y, 1000);
        KalmanSmoother ks;///?
        vector<double_t> records;///?
        estimate_using_lse_pso(ks, records, y,
            "F H Q R X0 P0",
            params.F, params.H, params.Q, params.R, params.X0, params.P0,
            5, 30, 0.01, -1,
            30, 50
            //penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
        );
        SSMParameters neoparams = ks.parameters;
        //print(records)
        //neoparams.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
    }

    void test_pure_lse_pso_4(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(1000, 1);
        matrix2d_t y(1000, 1);
        params.simulate(x, y, 1000);
        KalmanSmoother ks;///?
        vector<double_t> records;///?
        estimate_using_lse_pso(ks, records, y,
            "F H Q R X0 P0",
            params.F, params.H, params.Q, params.R, params.X0, params.P0,
            10, 20, 0.01, -1,
            10, 100
            //penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
        );
        SSMParameters neoparams = ks.parameters;
        //print(records)
        //neoparams.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
    }


    ///////////////////////////////////////////////////////////////////////////
    // EM Heuristic SSM Estimator
    ///////////////////////////////////////////////////////////////////////////
    struct EMHeuristicEstimatorParticle: public SelfImproveHeuristicEstimatorParticle{

        EMHeuristicEstimatorParticle():
            SelfImproveHeuristicEstimatorParticle(){}
        
        void improve_params(matrix2d_t& Y, index_t index=0){
            ExpectationMaximizationEstimator subestimator;
            subestimator.Y = Y;
            subestimator.estimate_F = this->estimate_F;
            subestimator.estimate_H = this->estimate_H;
            subestimator.estimate_Q = this->estimate_Q;
            subestimator.estimate_R = this->estimate_R;
            subestimator.estimate_X0 = this->estimate_X0;
            subestimator.estimate_P0 = this->estimate_P0;
            //print(
            //    "subestimator:",
            //    subestimator.estimate_F,
            //    subestimator.estimate_H,
            //    subestimator.estimate_Q,
            //    subestimator.estimate_R,
            //    subestimator.estimate_X0,
            //    subestimator.estimate_P0,
            //)
            subestimator.parameters = this->params;
            //estimator.parameters.random_initialize(is_none(F0), is_none(H0), is_none(Q0), is_none(R0), is_none(X00), is_none(P00))
            subestimator.estimation_iteration();
            this->params.copy_from(subestimator.parameters);
        }
    };

    struct EMHeuristicEstimator: public PurePSOHeuristicEstimator{
        EMHeuristicEstimator(): PurePSOHeuristicEstimator(){}

        virtual PSOHeuristicEstimatorParticle _create_particle(){
            return EMHeuristicEstimatorParticle();
        }
    };

    void test_pure_em_pso_1(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        params.X0(0, 0) += 1;
        params.H(0, 0) -= 0.3;
        params.F(0, 0) -= 0.1;
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        EMHeuristicEstimator kf;
        //kf.penalty_factor_roughness_X = 1
        //kf.penalty_factor_roughness_Y = 1
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show();
        KalmanSmoother s = kf.smoother();
        s.smooth();
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    void test_pure_em_pso_2(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        //params.X0{0, 0} += 1
        //params.H{0, 0} -= 0.3
        //params.F{0, 0} -= 0.1
        //params.show()
        matrix2d_t x(100, 1);
        matrix2d_t y(100, 1);
        params.simulate(x, y, 100);
        EMHeuristicEstimator kf;
        //kf.penalty_factor_roughness_X = 1
        //kf.penalty_factor_roughness_Y = 1
        kf.set_parameters(y, params);
        kf.estimate_parameters();
        //kf.parameters.show()
        //params.show()
        //params_orig.show()
        KalmanSmoother s = kf.smoother();
        s.smooth();
        ASSERT((abs(mean2(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)");
        ASSERT((abs(mean2(params.X0) - mean2(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)");
        ASSERT((abs(mean2(params.F) - mean2(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)");
        ASSERT((abs(mean2(params.H) - mean2(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)");
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xp()), 2) >= round(std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)")
        //ASSERT(round(std(kf.Xf()), 2) >= round(std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)")
    }

    // {{export}}
    void estimate_using_em_pso(
            /*out*/ KalmanSmoother& ks,
            /*out*/ vector<double_t>& loglikelihood_record,
            matrix2d_t& Y,
            const string& estimates="",
            matrix2d_t& F0=empty_matrix2d,
            matrix2d_t& H0=empty_matrix2d, 
            matrix2d_t& Q0=empty_matrix2d, 
            matrix2d_t& R0=empty_matrix2d, 
            matrix2d_t& X00=empty_matrix2d, 
            matrix2d_t& P00=empty_matrix2d, 
            index_t min_iterations=1,
            index_t max_iterations=10, 
            double_t min_improvement=0.01,
            int_t lat_dim=-1,
            index_t sample_size=30,
            index_t population_size=50, 
            const config_map& penalty_factors=empty_config_map
            )
    {
        EMHeuristicEstimator estimator;
        estimator.Y = Y;
        estimator.estimate_F = contains(estimates, "F");
        estimator.estimate_H = contains(estimates, "H");
        estimator.estimate_Q = contains(estimates, "Q");
        estimator.estimate_R = contains(estimates, "R");
        estimator.estimate_X0 = contains(estimates, "X0");
        estimator.estimate_P0 = contains(estimates, "P0");
        
        estimator.sample_size = sample_size;
        estimator.population_size = population_size;
        estimator.min_iterations = min_iterations;
        estimator.max_iterations = max_iterations;
        estimator.min_improvement = min_improvement;
        
        estimator.penalty_factor_low_variance_Q = get(penalty_factors, "low_variance_Q", 0.5);
        estimator.penalty_factor_low_variance_R = get(penalty_factors, "low_variance_R", 0.5);
        estimator.penalty_factor_low_variance_P0 = get(penalty_factors, "low_variance_P0", 0.5);
        estimator.penalty_factor_low_std_mean_ratio = get(penalty_factors, "low_std_mean_ratio", 0.5);
        estimator.penalty_factor_inestable_system = get(penalty_factors, "inestable_system", 10);
        estimator.penalty_factor_mse = get(penalty_factors, "mse", 1e-5);
        estimator.penalty_factor_roughness_X = get(penalty_factors, "roughness_X", 0.5);
        estimator.penalty_factor_roughness_Y = get(penalty_factors, "roughness_Y", 0.5);
        
        SSMParameters parameters;
        parameters.F = F0;
        if(!is_none(F0)){
            parameters.lat_dim = _nrows(F0);
        }
        parameters.H = H0;
        if(!is_none(H0)){
            parameters.lat_dim = _ncols(H0);
        }
        parameters.Q = Q0;
        if(!is_none(Q0)){
            parameters.lat_dim = _ncols(Q0);
        }
        parameters.R = R0;
        parameters.X0 = X00;
        if(!is_none(X00)){
            parameters.lat_dim = _nrows(X00);
        }
        parameters.P0 = P00;
        if(!is_none(P00)){
            parameters.lat_dim = _ncols(P00);
        }
        if(lat_dim > 0){
            parameters.lat_dim = lat_dim;
        }
        parameters.obs_dim = _nrows(Y);
        parameters.obs_dim = _nrows(Y);
        parameters.random_initialize(is_none(F0), is_none(H0), is_none(Q0), is_none(R0), is_none(X00), is_none(P00));
        estimator.set_parameters(Y, parameters, contains(estimates, "F"), contains(estimates, "H"), contains(estimates, "Q"), contains(estimates, "R"), contains(estimates, "X0"), contains(estimates, "P0"), -1);
        estimator.estimate_parameters();
        ks = estimator.smoother();
        ks.smooth();
        loglikelihood_record = estimator.loglikelihood_record;
    }

    void test_pure_em_pso_3(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(1000, 1);
        matrix2d_t y(1000, 1);
        params.simulate(x, y, 1000);
        KalmanSmoother ks;///?
        vector<double_t> records;///?
        estimate_using_em_pso(ks, records, y,
            "F H Q R X0 P0",
            params.F, params.H, params.Q, params.R, params.X0, params.P0,
            5, 30, 0.01, -1,
            30, 50
            //penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
        );
        SSMParameters neoparams = ks.parameters;
        //print(records)
        //neoparams.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
    }

    void test_pure_em_pso_4(){
        SSMParameters params = _create_params_ones_kx1(colvec({-50}), colvec({10}));
        matrix2d_t x(1000, 1);
        matrix2d_t y(1000, 1);
        params.simulate(x, y, 1000);
        KalmanSmoother ks;///?
        vector<double_t> records;///?
        estimate_using_em_pso(ks, records, y,
            "F H Q R X0 P0",
            params.F, params.H, params.Q, params.R, params.X0, params.P0,
            10, 20, 0.01, -1,
            10, 100
            //penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
        );
        SSMParameters neoparams = ks.parameters;
        //print(records)
        //neoparams.show()
        //params.show()
        //params_orig.show()
        ASSERT((abs(mean2(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean");
        ASSERT((abs(mean2(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean");
        ASSERT((abs(mean2(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean");
    }


    ///////////////////////////////////////////////////////////////////////////
    ///  C interface
    ///////////////////////////////////////////////////////////////////////////
    
    // {{export-c}}
    #define create_pointer_handler_empty(Name, KlassType, EmptyName) \
        __inline__ KlassType* c_new_##Name(){ return new KlassType();} \
        __inline__ void c_del_##Name(KlassType* p){ if(addressof(*p) != addressof(EmptyName)) delete p;} \
        export_function void* _new_##Name(){ return reinterpret_cast<void*>(new KlassType());} \
        export_function void _del_##Name(void* p){ delete reinterpret_cast<KlassType*>(p);}
    
    #define create_pointer_handler(Name, KlassType) \
        __inline__ KlassType* c_new_##Name(){ return new KlassType();} \
        __inline__ void c_del_##Name(KlassType* p){ delete p;} \
        export_function void* _new_##Name(){ return reinterpret_cast<void*>(new KlassType());} \
        export_function void _del_##Name(void* p){ delete reinterpret_cast<KlassType*>(p);}

    create_pointer_handler_empty(matrix2d, matrix2d_t, empty_matrix2d)
    create_pointer_handler_empty(matrix3d, matrix3d_t, empty_matrix3d)
    create_pointer_handler(ssm_parameters, SSMParameters)
    create_pointer_handler(ssm_estimated, SSMEstimated)
    create_pointer_handler(kalman_filter, KalmanFilter)
    create_pointer_handler(kalman_smoother, KalmanSmoother)
    create_pointer_handler(expectation_maximization_estimator, ExpectationMaximizationEstimator)
    create_pointer_handler(pso_heuristic_estimator_particle, PSOHeuristicEstimatorParticle)
    create_pointer_handler(pso_heuristic_estimator, PurePSOHeuristicEstimator)
    create_pointer_handler(lse_heuristic_estimator_particle, LSEHeuristicEstimatorParticle)
    create_pointer_handler(lse_heuristic_estimator, LSEHeuristicEstimator)
    create_pointer_handler(em_heuristic_estimator_particle, EMHeuristicEstimatorParticle)
    create_pointer_handler(em_heuristic_estimator, EMHeuristicEstimator)

    // {{export-c}}
    export_function void* _create_matrix2d_from(double* data, index_t n_rows, index_t n_cols){
        if(data == nullptr) return reinterpret_cast<void*>(&empty_matrix2d);
        matrix2d_t* matrix = c_new_matrix2d();
        (*matrix).set_size(n_rows, n_cols);
        index_t k = 0;
        for_range(r, 0, n_rows){
            for_range(c, 0, n_cols){
                (*matrix)(r, c) = data[k++];
            }
        }
        return reinterpret_cast<void*>(matrix);
    }
    
    __inline__ matrix2d_t* c_create_matrix2d_from(double* data, index_t n_rows, index_t n_cols){
        return reinterpret_cast<matrix2d_t*>(
            _create_matrix2d_from(data, n_rows, n_cols)
        );
    }

    __inline__ void c_fill_array_from2d(double* data, matrix2d_t* matrix){
        if(matrix == nullptr) return;
        index_t n_rows = matrix->n_rows;
        index_t n_cols = matrix->n_cols;
        index_t k = 0;
        for_range(r, 0, n_rows){
            for_range(c, 0, n_cols){
                data[k++] = (*matrix)(r, c);
            }
        }
    }
    
    export_function void _fill_array_from2d(double* data, void* matrix){
        c_fill_array_from2d(data, reinterpret_cast<matrix2d_t*>(matrix));
    }

    __inline__ void c_fill_array_from3d(double* data, matrix3d_t* matrix){
        if(matrix == nullptr) return;
        index_t n_rows = matrix->n_rows;
        index_t n_cols = matrix->n_cols;
        index_t n_slices = matrix->n_slices;
        index_t k = 0;
        for_range(r, 0, n_rows){
            for_range(c, 0, n_cols){
                for_range(s, 0, n_slices){
                    data[k++] = (*matrix)(r, c, s);
                }
            }
        }
    }
    
    export_function void _fill_array_from3d(double* data, void* matrix){
        c_fill_array_from3d(data, reinterpret_cast<matrix3d_t*>(matrix));
    }

    __inline__ void c_fill_array_from_vector(double* data, vector<double_t>& vec){
        index_t k = 0;
        for (auto i: vec){
            data[k++] = i;
        }
    }

    // {{export-c}}
    export_function void* _kalman_filter(
        index_t obs_dim, index_t lat_dim, index_t T,
        double_t* Y, double_t* F, double_t* H, 
        double_t* Q, double_t* R, 
        double_t* X0, double_t* P0)
    {
        matrix2d_t* _Y = c_create_matrix2d_from(Y, obs_dim, T);
        matrix2d_t* _F = c_create_matrix2d_from(F, lat_dim, lat_dim);
        matrix2d_t* _H = c_create_matrix2d_from(H, obs_dim, lat_dim);
        matrix2d_t* _Q = c_create_matrix2d_from(Q, lat_dim, lat_dim);
        matrix2d_t* _R = c_create_matrix2d_from(R, obs_dim, obs_dim);
        matrix2d_t* _X0 = c_create_matrix2d_from(X0, lat_dim, 1);
        matrix2d_t* _P0 = c_create_matrix2d_from(P0, lat_dim, lat_dim);
        KalmanFilter* kf = c_new_kalman_filter();
        kf->set_F(*_F);
        kf->set_H(*_H);
        kf->set_Q(*_Q);
        kf->set_R(*_R);
        kf->set_X0(*_X0);
        kf->set_P0(*_P0);
        kf->set_Y(*_Y);
        kf->set_obs_dim(_nrows(*_Y));
        kf->set_lat_dim(_nrows(*_X0));
        kf->filter();
        c_del_matrix2d(_Y);
        c_del_matrix2d(_F);
        c_del_matrix2d(_H);
        c_del_matrix2d(_Q);
        c_del_matrix2d(_R);
        c_del_matrix2d(_X0);
        c_del_matrix2d(_P0);
        return reinterpret_cast<void*>(kf);
    }

    // {{export-c}}
    export_function void _kalman_filter_results(
        /*out*/ double* Xp, /*out*/ double* Pp, /*out*/ double* Yp, 
        /*out*/ double* Xf, /*out*/ double* Pf, /*out*/ double* Yf,
        void* kf)
    {
        KalmanFilter* _kf = reinterpret_cast<KalmanFilter*>(kf);
        c_fill_array_from2d(Xp, &(_kf->Xp()));
        c_fill_array_from3d(Pp, &(_kf->Pp()));
        c_fill_array_from2d(Yp, &(_kf->Yp()));
        //
        c_fill_array_from2d(Xf, &(_kf->Xf()));
        c_fill_array_from3d(Pf, &(_kf->Pf()));
        c_fill_array_from2d(Yf, &(_kf->Yf()));
    }

    // {{export-c}}
    export_function void _kalman_filter_parameters(
        /*out*/ index_t* obs_dim, /*out*/ index_t* lat_dim, /*out*/ index_t* T,
        /*out*/ double* F, /*out*/ double* H,  /*out*/ double* Q,
        /*out*/ double* R, /*out*/ double* X0, /*out*/ double* P0,
        void* kalman_filter)
    {
        KalmanFilter* kf = reinterpret_cast<KalmanFilter*>(kalman_filter);
        *lat_dim = kf->lat_dim();
        *obs_dim = kf->obs_dim();
        *T = kf->T();
        c_fill_array_from2d(F, &(kf->F()));
        c_fill_array_from2d(H, &(kf->H()));
        c_fill_array_from2d(Q, &(kf->Q()));
        c_fill_array_from2d(R, &(kf->R()));
        c_fill_array_from2d(X0, &(kf->X0()));
        c_fill_array_from2d(P0, &(kf->P0()));
    }

    // {{export-c}}
    export_function void* _kalman_smoother(
        index_t obs_dim, index_t lat_dim, index_t T,
        double_t* Y, double_t* F, double_t* H, 
        double_t* Q, double_t* R, 
        double_t* X0, double_t* P0)
    {
        matrix2d_t* _Y = c_create_matrix2d_from(Y, obs_dim, T);
        matrix2d_t* _F = c_create_matrix2d_from(F, lat_dim, lat_dim);
        matrix2d_t* _H = c_create_matrix2d_from(H, obs_dim, lat_dim);
        matrix2d_t* _Q = c_create_matrix2d_from(Q, lat_dim, lat_dim);
        matrix2d_t* _R = c_create_matrix2d_from(R, obs_dim, obs_dim);
        matrix2d_t* _X0 = c_create_matrix2d_from(X0, lat_dim, 1);
        matrix2d_t* _P0 = c_create_matrix2d_from(P0, lat_dim, lat_dim);
        KalmanSmoother* kf = c_new_kalman_smoother();
        kf->set_F(*_F);
        kf->set_H(*_H);
        kf->set_Q(*_Q);
        kf->set_R(*_R);
        kf->set_X0(*_X0);
        kf->set_P0(*_P0);
        kf->set_Y(*_Y);
        kf->set_obs_dim(_nrows(*_Y));
        kf->set_lat_dim(_nrows(*_X0));
        kf->smooth();
        c_del_matrix2d(_Y);
        c_del_matrix2d(_F);
        c_del_matrix2d(_H);
        c_del_matrix2d(_Q);
        c_del_matrix2d(_R);
        c_del_matrix2d(_X0);
        c_del_matrix2d(_P0);
        return reinterpret_cast<void*>(kf);
    }
    
    
    // {{export-c}}
    export_function void _kalman_smoother_results(
        /*out*/ double* Xp, /*out*/ double* Pp, /*out*/ double* Yp, 
        /*out*/ double* Xf, /*out*/ double* Pf, /*out*/ double* Yf,
        /*out*/ double* Xs, /*out*/ double* Ps, /*out*/ double* Ys,
        void* kalman_smoother)
    {
        KalmanSmoother* _kf = reinterpret_cast<KalmanSmoother*>(kalman_smoother);
        c_fill_array_from2d(Xp, &(_kf->Xp()));
        c_fill_array_from3d(Pp, &(_kf->Pp()));
        c_fill_array_from2d(Yp, &(_kf->Yp()));
        //
        c_fill_array_from2d(Xf, &(_kf->Xf()));
        c_fill_array_from3d(Pf, &(_kf->Pf()));
        c_fill_array_from2d(Yf, &(_kf->Yf()));
        //
        c_fill_array_from2d(Xs, &(_kf->Xs()));
        c_fill_array_from3d(Ps, &(_kf->Ps()));
        c_fill_array_from2d(Ys, &(_kf->Ys()));
    }

    // {{export-c}}
    export_function void _kalman_smoother_parameters(
        /*out*/ index_t* obs_dim, /*out*/ index_t* lat_dim, /*out*/ index_t* T,
        /*out*/ double* F, /*out*/ double* H,  /*out*/ double* Q,
        /*out*/ double* R, /*out*/ double* X0, /*out*/ double* P0,
        void* kalman_smoother)
    {
        KalmanSmoother* kf = reinterpret_cast<KalmanSmoother*>(kalman_smoother);
        *lat_dim = kf->lat_dim();
        *obs_dim = kf->obs_dim();
        *T = kf->T();
        c_fill_array_from2d(F, &(kf->F()));
        c_fill_array_from2d(H, &(kf->H()));
        c_fill_array_from2d(Q, &(kf->Q()));
        c_fill_array_from2d(R, &(kf->R()));
        c_fill_array_from2d(X0, &(kf->X0()));
        c_fill_array_from2d(P0, &(kf->P0()));
    }

    // {{export-c}}
    export_function void* _estimate_using_em(
            const char* estimates,
            index_t obs_dim, index_t lat_dim, index_t T,
            double_t* Y,
            double_t* F, double_t* H, 
            double_t* Q, double_t* R, 
            double_t* X0, double_t* P0,
            index_t min_iterations,
            index_t max_iterations, 
            double_t min_improvement,
            double_t* loglikelihood_record=nullptr)
    {
        matrix2d_t* _Y = c_create_matrix2d_from(Y, obs_dim, T);
        matrix2d_t* _F = c_create_matrix2d_from(F, lat_dim, lat_dim);
        matrix2d_t* _H = c_create_matrix2d_from(H, obs_dim, lat_dim);
        matrix2d_t* _Q = c_create_matrix2d_from(Q, lat_dim, lat_dim);
        matrix2d_t* _R = c_create_matrix2d_from(R, obs_dim, obs_dim);
        matrix2d_t* _X0 = c_create_matrix2d_from(X0, lat_dim, 1);
        matrix2d_t* _P0 = c_create_matrix2d_from(P0, lat_dim, lat_dim);
        string _estimates = estimates;
        //
        ExpectationMaximizationEstimator estimator;
        estimator.Y = *_Y;
        estimator.estimate_F = contains(_estimates, "F");
        estimator.estimate_H = contains(_estimates, "H");
        estimator.estimate_Q = contains(_estimates, "Q");
        estimator.estimate_R = contains(_estimates, "R");
        estimator.estimate_X0 = contains(_estimates, "X0");
        estimator.estimate_P0 = contains(_estimates, "P0");
        estimator.parameters = SSMParameters();
        estimator.parameters.F = *_F;
        if(F != nullptr){
            estimator.parameters.lat_dim = _nrows(*_F);
        }
        estimator.parameters.H = *_H;
        if(H != nullptr){
            estimator.parameters.lat_dim = _ncols(*_H);
        }
        estimator.parameters.Q = *_Q;
        if(Q != nullptr){
            estimator.parameters.lat_dim = _ncols(*_Q);
        }
        estimator.parameters.R = *_R;
        estimator.parameters.X0 = *_X0;
        if(X0 != nullptr){
            estimator.parameters.lat_dim = _nrows(*_X0);
        }
        estimator.parameters.P0 = *_P0;
        if(P0 != nullptr){
            estimator.parameters.lat_dim = _ncols(*_P0);
        }
        estimator.parameters.lat_dim = lat_dim;
        estimator.parameters.obs_dim = obs_dim;
        //
        estimator.min_iterations = min_iterations;
        estimator.max_iterations = max_iterations;
        estimator.min_improvement = min_improvement;
        //
        estimator.parameters.random_initialize(
            F == nullptr, H == nullptr, Q == nullptr, 
            R == nullptr, X0 == nullptr, P0 == nullptr);
        estimator.estimate_parameters();
        KalmanSmoother* ks = c_new_kalman_smoother();
        *ks = estimator.smoother();
        ks->smooth();
        if(loglikelihood_record != nullptr){
            c_fill_array_from_vector(loglikelihood_record, estimator.loglikelihood_record);
        }
        //
        c_del_matrix2d(_Y);
        c_del_matrix2d(_F);
        c_del_matrix2d(_H);
        c_del_matrix2d(_Q);
        c_del_matrix2d(_R);
        c_del_matrix2d(_X0);
        c_del_matrix2d(_P0);
        //
        return reinterpret_cast<void*>(ks);
    }
    
    // {{export-c}}
    template<typename PSOEstimator>
    void* _estimate_using_x_pso(
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
            double_t* loglikelihood_record=nullptr,
            double_t penalty_low_variance_Q=0.5,
            double_t penalty_low_variance_R=0.5,
            double_t penalty_low_variance_P0=0.5,
            double_t penalty_low_std_mean_ratio=0.5,
            double_t penalty_inestable_system=10,
            double_t penalty_mse=1e-5,
            double_t penalty_roughness_X=0.5,
            double_t penalty_roughness_Y=0.5)
    {
        matrix2d_t* _Y = c_create_matrix2d_from(Y, obs_dim, T);
        matrix2d_t* _F = c_create_matrix2d_from(F, lat_dim, lat_dim);
        matrix2d_t* _H = c_create_matrix2d_from(H, obs_dim, lat_dim);
        matrix2d_t* _Q = c_create_matrix2d_from(Q, lat_dim, lat_dim);
        matrix2d_t* _R = c_create_matrix2d_from(R, obs_dim, obs_dim);
        matrix2d_t* _X0 = c_create_matrix2d_from(X0, lat_dim, 1);
        matrix2d_t* _P0 = c_create_matrix2d_from(P0, lat_dim, lat_dim);
        string _estimates = estimates;
        //
        PSOEstimator estimator;
        estimator.Y = *_Y;
        estimator.estimate_F = contains(_estimates, "F");
        estimator.estimate_H = contains(_estimates, "H");
        estimator.estimate_Q = contains(_estimates, "Q");
        estimator.estimate_R = contains(_estimates, "R");
        estimator.estimate_X0 = contains(_estimates, "X0");
        estimator.estimate_P0 = contains(_estimates, "P0");
        //
        estimator.sample_size = sample_size;
        estimator.population_size = population_size;
        //
        estimator.min_iterations = min_iterations;
        estimator.max_iterations = max_iterations;
        estimator.min_improvement = min_improvement;
        //
        estimator.penalty_factor_low_variance_Q = penalty_low_variance_Q;
        estimator.penalty_factor_low_variance_R = penalty_low_variance_R;
        estimator.penalty_factor_low_variance_P0 = penalty_low_variance_P0;
        estimator.penalty_factor_low_std_mean_ratio = penalty_low_std_mean_ratio;
        estimator.penalty_factor_inestable_system = penalty_inestable_system;
        estimator.penalty_factor_mse = penalty_mse;
        estimator.penalty_factor_roughness_X = penalty_roughness_X;
        estimator.penalty_factor_roughness_Y = penalty_roughness_Y;
        
        SSMParameters parameters;
        parameters.F = *_F;
        if(F != nullptr){
            parameters.lat_dim = _nrows(*_F);
        }
        parameters.H = *_H;
        if(H != nullptr){
            parameters.lat_dim = _ncols(*_H);
        }
        parameters.Q = *_Q;
        if(Q != nullptr){
            parameters.lat_dim = _ncols(*_Q);
        }
        parameters.R = *_R;
        parameters.X0 = *_X0;
        if(X0 != nullptr){
            parameters.lat_dim = _nrows(*_X0);
        }
        parameters.P0 = *_P0;
        if(P0 != nullptr){
            parameters.lat_dim = _ncols(*_P0);
        }
        parameters.lat_dim = lat_dim;
        parameters.obs_dim = obs_dim;
        //
        parameters.random_initialize(
            F == nullptr, H == nullptr, Q == nullptr, 
            R == nullptr, X0 == nullptr, P0 == nullptr);
        estimator.set_parameters(*_Y, parameters,
                contains(estimates, "F"), contains(estimates, "H"), 
                contains(estimates, "Q"), contains(estimates, "R"), 
                contains(estimates, "X0"), contains(estimates, "P0"), -1);
        estimator.estimate_parameters();
        KalmanSmoother* ks = c_new_kalman_smoother();
        *ks = estimator.smoother();
        ks->smooth();
        if(loglikelihood_record != nullptr){
            c_fill_array_from_vector(loglikelihood_record, estimator.loglikelihood_record);
        }
        //
        c_del_matrix2d(_Y);
        c_del_matrix2d(_F);
        c_del_matrix2d(_H);
        c_del_matrix2d(_Q);
        c_del_matrix2d(_R);
        c_del_matrix2d(_X0);
        c_del_matrix2d(_P0);
        //
        return reinterpret_cast<void*>(ks);
    }
    
    // {{export-c}}
    export_function void* _estimate_using_pso(
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
            double_t* loglikelihood_record=nullptr,
            double_t penalty_low_variance_Q=0.5,
            double_t penalty_low_variance_R=0.5,
            double_t penalty_low_variance_P0=0.5,
            double_t penalty_low_std_mean_ratio=0.5,
            double_t penalty_inestable_system=10,
            double_t penalty_mse=1e-5,
            double_t penalty_roughness_X=0.5,
            double_t penalty_roughness_Y=0.5)
    {
        return _estimate_using_x_pso<PurePSOHeuristicEstimator>(
            estimates,
            obs_dim, lat_dim, T,
            Y, F, H, Q, R, X0, P0,
            min_iterations, max_iterations, min_improvement,
            sample_size, population_size,
            loglikelihood_record,
            penalty_low_variance_Q,
            penalty_low_variance_R,
            penalty_low_variance_P0,
            penalty_low_std_mean_ratio,
            penalty_inestable_system,
            penalty_mse,
            penalty_roughness_X,
            penalty_roughness_Y);
    }

    // {{export-c}}
    export_function void* _estimate_using_lse_pso(
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
            double_t* loglikelihood_record=nullptr,
            double_t penalty_low_variance_Q=0.5,
            double_t penalty_low_variance_R=0.5,
            double_t penalty_low_variance_P0=0.5,
            double_t penalty_low_std_mean_ratio=0.5,
            double_t penalty_inestable_system=10,
            double_t penalty_mse=1e-5,
            double_t penalty_roughness_X=0.5,
            double_t penalty_roughness_Y=0.5)
    {
        return _estimate_using_x_pso<LSEHeuristicEstimator>(
            estimates,
            obs_dim, lat_dim, T,
            Y, F, H, Q, R, X0, P0,
            min_iterations, max_iterations, min_improvement,
            sample_size, population_size,
            loglikelihood_record,
            penalty_low_variance_Q,
            penalty_low_variance_R,
            penalty_low_variance_P0,
            penalty_low_std_mean_ratio,
            penalty_inestable_system,
            penalty_mse,
            penalty_roughness_X,
            penalty_roughness_Y);
    }

    // {{export-c}}
    export_function void* _estimate_using_em_pso(
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
            double_t* loglikelihood_record=nullptr,
            double_t penalty_low_variance_Q=0.5,
            double_t penalty_low_variance_R=0.5,
            double_t penalty_low_variance_P0=0.5,
            double_t penalty_low_std_mean_ratio=0.5,
            double_t penalty_inestable_system=10,
            double_t penalty_mse=1e-5,
            double_t penalty_roughness_X=0.5,
            double_t penalty_roughness_Y=0.5)
    {
        return _estimate_using_x_pso<EMHeuristicEstimator>(
            estimates,
            obs_dim, lat_dim, T,
            Y, F, H, Q, R, X0, P0,
            min_iterations, max_iterations, min_improvement,
            sample_size, population_size,
            loglikelihood_record,
            penalty_low_variance_Q,
            penalty_low_variance_R,
            penalty_low_variance_P0,
            penalty_low_std_mean_ratio,
            penalty_inestable_system,
            penalty_mse,
            penalty_roughness_X,
            penalty_roughness_Y);
    }

    // {{export-c}}
    export_function void _performance_of_parameters(
            /*out*/ double_t* loglikelihood, 
            /*out*/ double_t* low_std_to_mean_penalty, 
            /*out*/ double_t* low_variance_Q_penalty, 
            /*out*/ double_t* low_variance_R_penalty, 
            /*out*/ double_t* low_variance_P0_penalty, 
            /*out*/ double_t* system_inestability_penalty, 
            /*out*/ double_t* mean_squared_error_penalty, 
            /*out*/ double_t* roughness_X_penalty, 
            /*out*/ double_t* roughness_Y_penalty, 
            index_t obs_dim, index_t lat_dim, index_t T,
            double_t* Y,
            double_t* F, double_t* H, 
            double_t* Q, double_t* R, 
            double_t* X0, double_t* P0)
    {
        matrix2d_t* _Y = c_create_matrix2d_from(Y, obs_dim, T);
        matrix2d_t* _F = c_create_matrix2d_from(F, lat_dim, lat_dim);
        matrix2d_t* _H = c_create_matrix2d_from(H, obs_dim, lat_dim);
        matrix2d_t* _Q = c_create_matrix2d_from(Q, lat_dim, lat_dim);
        matrix2d_t* _R = c_create_matrix2d_from(R, obs_dim, obs_dim);
        matrix2d_t* _X0 = c_create_matrix2d_from(X0, lat_dim, 1);
        matrix2d_t* _P0 = c_create_matrix2d_from(P0, lat_dim, lat_dim);
        //
        SSMParameters parameters;
        parameters.F = *_F;
        parameters.H = *_H;
        parameters.Q = *_Q;
        parameters.R = *_R;
        parameters.X0 = *_X0;
        parameters.P0 = *_P0;
        parameters.lat_dim = lat_dim;
        parameters.obs_dim = obs_dim;
        //
        KalmanSmoother smoother = kalman_smoother_from_parameters(*_Y, parameters);
        *loglikelihood = smoother.loglikelihood();
        *low_std_to_mean_penalty = parameters._penalize_low_std_to_mean_ratio(smoother.X0(), smoother.P0());
        *low_variance_Q_penalty = parameters._penalize_low_variance(smoother.Q());
        *low_variance_R_penalty = parameters._penalize_low_variance(smoother.R());
        *low_variance_P0_penalty = parameters._penalize_low_variance(smoother.P0());
        *system_inestability_penalty = parameters._penalize_inestable_system(smoother.F());
        *mean_squared_error_penalty = parameters._penalize_mean_squared_error(*_Y, smoother.Ys());
        *roughness_X_penalty = parameters._penalize_roughness(smoother.Xs());
        *roughness_Y_penalty = parameters._penalize_roughness(*_Y);

        //
        c_del_matrix2d(_Y);
        c_del_matrix2d(_F);
        c_del_matrix2d(_H);
        c_del_matrix2d(_Q);
        c_del_matrix2d(_R);
        c_del_matrix2d(_X0);
        c_del_matrix2d(_P0);
        //
    }

    ///////////////////////////////////////////////////////////////////////////
    ///  Generic interface
    ///////////////////////////////////////////////////////////////////////////
    
    // {{export-c}}
    export_function void* _estimate_ssm(
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
            double_t* loglikelihood_record=nullptr,
            double_t penalty_low_variance_Q=0.5,
            double_t penalty_low_variance_R=0.5,
            double_t penalty_low_variance_P0=0.5,
            double_t penalty_low_std_mean_ratio=0.5,
            double_t penalty_inestable_system=10,
            double_t penalty_mse=1e-5,
            double_t penalty_roughness_X=0.5,
            double_t penalty_roughness_Y=0.5)
    {
        string _type_of_estimator = type_of_estimator;
        _type_of_estimator = to_upper(_type_of_estimator);
        if(_type_of_estimator == "EM"){
            return _estimate_using_em(
                estimates,
                obs_dim, lat_dim, T,
                Y, F, H, Q, R, X0, P0,
                min_iterations, max_iterations, min_improvement,
                loglikelihood_record);
        }else if(_type_of_estimator == "PSO"){
            return _estimate_using_x_pso<PurePSOHeuristicEstimator>(
                estimates,
                obs_dim, lat_dim, T,
                Y, F, H, Q, R, X0, P0,
                min_iterations, max_iterations, min_improvement,
                sample_size, population_size,
                loglikelihood_record,
                penalty_low_variance_Q,
                penalty_low_variance_R,
                penalty_low_variance_P0,
                penalty_low_std_mean_ratio,
                penalty_inestable_system,
                penalty_mse,
                penalty_roughness_X,
                penalty_roughness_Y);
        }else if(_type_of_estimator == "LSE+PSO"){
            return _estimate_using_x_pso<LSEHeuristicEstimator>(
                estimates,
                obs_dim, lat_dim, T,
                Y, F, H, Q, R, X0, P0,
                min_iterations, max_iterations, min_improvement,
                sample_size, population_size,
                loglikelihood_record,
                penalty_low_variance_Q,
                penalty_low_variance_R,
                penalty_low_variance_P0,
                penalty_low_std_mean_ratio,
                penalty_inestable_system,
                penalty_mse,
                penalty_roughness_X,
                penalty_roughness_Y);
        }else if(_type_of_estimator == "EM+PSO"){
            return _estimate_using_x_pso<EMHeuristicEstimator>(
                estimates,
                obs_dim, lat_dim, T,
                Y, F, H, Q, R, X0, P0,
                min_iterations, max_iterations, min_improvement,
                sample_size, population_size,
                loglikelihood_record,
                penalty_low_variance_Q,
                penalty_low_variance_R,
                penalty_low_variance_P0,
                penalty_low_std_mean_ratio,
                penalty_inestable_system,
                penalty_mse,
                penalty_roughness_X,
                penalty_roughness_Y);
        }else{
            throw logic_error("Type of estimator not found!");
        }
    }

}
