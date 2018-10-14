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
        __assert(#Expr, [](){ Expr }, __FILE__, __LINE__, Msg)
#else
    #define ASSERT(Expr, Msg) ;
#endif

void __assert(const char* expr_str, function<bool(void)> expr, const char* file, int line, const char* msg)
{
    exception_ptr p;
    int err_type = 0;
    try{
        if(!expr()) err_type = 1;
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

void __assert2(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if(!expr){
        std::cerr << "Assert failed:\t" << msg << "\n"
                  << "Expected:\t" << expr_str << "\n"
                  << "Source:\t\t" << file << ", line " << line << "\n";
        std::abort();
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
