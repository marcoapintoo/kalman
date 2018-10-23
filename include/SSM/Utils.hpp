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

using namespace std;
using namespace arma;

///////////////////////////////////////////////////////////////////////////
///  Macros definition
///////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
    #define ASSERT(Expr, Msg) \
        __assert2(#Expr, Expr, __FILE__, __LINE__, Msg)
        //__assert3(#Expr, [&]() -> bool { Expr; }, __FILE__, __LINE__, Msg)
    #define EXPECTED(ExprLeft, OP, ExprRight, Msg) \
        __assert_expectation(\
            OP, \
            #ExprLeft, #OP,  #ExprRight, \
            [&]() -> double { return ExprLeft; }, \
            [&]() -> double { return ExprRight; }, \
            __FILE__, __LINE__, Msg)
#else
    #define ASSERT(Expr, Msg) ;
    #define EXPECTED(ExprLeft, OP, ExprRight, Msg) ;
#endif
#define IS_EQUAL_TO 10
#define IS_ALMOST_EQUAL_TO 11
#define IS_NEAR_EQUAL_TO 12
#define IS_LESS_THAN 20
#define IS_LESS_EQUAL_THAN 21
#define IS_GREATER_THAN 30
#define IS_GREATER_EQUAL_THAN 31


//template<typename T>
void __assert_expectation(int type_expression,
                          const char* expr_str_left,
                          const char* expr_str_OP,
                          const char* expr_str_right,
                          function<double(void)> expr_left,
                          function<double(void)> expr_right,
                          const char* file, int line, const char* msg)
{
    stringstream expection_left;
    double result_left;
    stringstream expection_right;
    double result_right;
    try{
        result_left = expr_left();
    } catch (const std::exception &e){
        expection_left << "Evaluation failed when evaluating left side:\t" << expr_str_left << "\n"
             << "Expection thrown:\t" << e.what() << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
    } catch (...){
        expection_left << "Evaluation failed when evaluating left side:\t" << expr_str_left << "\n"
             << "Expection thrown:\t" << "[[UNKNOWN]]" << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
    }
    
    try{
        result_right = expr_right();
    } catch (const std::exception &e){
        expection_right << "Evaluation failed when evaluating left side:\t" << expr_str_left << "\n"
             << "Expection thrown:\t" << e.what() << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
    } catch (...){
        expection_right << "Evaluation failed when evaluating left side:\t" << expr_str_left << "\n"
             << "Expection thrown:\t" << "[[UNKNOWN]]" << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
    }
    string exc_left = expection_left.str();
    string exc_right = expection_right.str();
    if(!exc_left.empty() || !exc_right.empty()){
        if(!exc_left.empty()){
            cout << expection_left.str() << endl;
        }
        if(!exc_right.empty()){
            cout << expection_right.str() << endl;
        }
        return;
    }
    
    bool result = false;
    switch(type_expression){
        case IS_EQUAL_TO:
            result = (result_left == result_right);
        break;
        case IS_ALMOST_EQUAL_TO:
            result = abs(result_left - result_right) < 1e-6;
        break;
        case IS_NEAR_EQUAL_TO:
            result = abs(result_left - result_right) < 1e-2;
        break;
        case IS_LESS_THAN:
            result = (result_left < result_right);
        break;
        case IS_LESS_EQUAL_THAN:
            result = (result_left <= result_right);
        break;
        case IS_GREATER_THAN:
            result = (result_left > result_right);
        break;
        case IS_GREATER_EQUAL_THAN:
            result = (result_left >= result_right);
        break;
    }
    if(result) return;
    cerr << "Assert failed:\t" << msg << "\n"
             << "Expected:\t" << expr_str_left << " " << expr_str_OP << " " << expr_str_right << "\n"
             << "Obtained:\t" << (result_left) << " " << expr_str_OP << " " << (result_right) << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
    throw exception();
}
void __assert(const char* expr_str, function<bool(void)> expr, const char* file, int line, const char* msg)
{
    exception_ptr p;
    int err_type = 0;
    try{
        if(expr()) err_type = 1;
    }catch(...){ p = current_exception(); }
    if(err_type == 0){
        cerr << "Assert failed:\t" << msg << "\n"
             << "Expected:\t" << expr_str << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
        return;
    }
    try{
        if(p) rethrow_exception(p);
    }catch(const exception &e){
        cerr << "Assert failed:\t" << msg << "\n"
             << "Exception thrown:\t" << e.what() << "\n"
             << "Expected:\t" << expr_str << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

void __assert3(const char* expr_str, function<bool(void)> expr, const char* file, int line, const char* msg)
{
    try{
        if(!expr()){
            cerr << "Assert failed:\t" << msg << "\n"
             << "Expected:\t" << expr_str << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
        }
    } catch (const std::exception &e){
        cerr << "Assert failed:\t" << msg << "\n"
             << "Expected:\t" << expr_str << "\n"
             << "Expection thrown:\t" << e.what() << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
    } catch (...){
        cerr << "Assert failed:\t" << msg << "\n"
             << "Expected:\t" << expr_str << "\n"
             << "Expection thrown:\t" << "UNKNOWN" << "\n"
             << "Source:\t\t" << file << ", line " << line << "\n";
    }
}

void __assert2(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if(!expr){
        std::cerr << "Assert failed:\t" << msg << "\n"
                  << "Expected:\t" << expr_str << "\n"
                  << "Source:\t\t" << file << ", line " << line << "\n";
        //std::abort();
    }
}

// Export function
#if defined(_MSC_VER)
    #define export_function extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
    #define export_function extern "C" __attribute__((visibility("default")))
#else
    #define export_function
    #pragma warning Unknown dynamic link import/export semantics.
#endif

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

///////////////////////////////////////////////////////////////////////////
///  Utils for C++
///////////////////////////////////////////////////////////////////////////

bool contains(const string& complete_text, const string& test_to_find){
    return complete_text.find(test_to_find) != string::npos;
}
template <typename K, typename V>
V get(const map<K,V>& m, const K& key, const V& def_value){
   typename map<K,V>::const_iterator it = m.find(key);
   if(it == m.end()){
      return def_value;
   }else{
      return it->second;
   }
}

//template <string, double_t>
//template <>
double_t get(const map<string, double_t>& m, const char* key, const double_t& def_value){
   typename map<string, double_t>::const_iterator it = m.find(string(key));
   if(it == m.end()){
      return def_value;
   }else{
      return it->second;
   }
}

string to_upper(string str){
    string strToConvert = str;
    transform(strToConvert.begin(), strToConvert.end(), strToConvert.begin(), ::toupper);
    return strToConvert;
}
