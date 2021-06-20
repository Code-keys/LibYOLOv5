#include <iostream>
#include <string>
#include<asssert.h>


namespace cx{

template<typename T>
inline T trans2num(const std::string s,T l = 8){
    assert (s.size() < l );   l = 0;
    for(char c:s){
        assert( c <= '9' && c >= '0');
        l = l*10 + int(c-'0');
    }
    return l;
};

inline int make_divisible(int const c2, int divisor=8){
    return c2 % divisor != 0 ?  (c2 / divisor + 1) * divisor :  c2 ; 
};































}