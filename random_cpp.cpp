#include <random>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/args.hpp>
#include <boost/atomic.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

#include <thread>
#include <atomic>

using std::thread, std::vector;

class Thread {
public:
    Thread(int size) : nth(thread::hardware_concurrency()),
    segSz(size/nth),  size(size), threads(new thread[nth]) {}

    ~Thread() {
        delete[]threads;
    }
    static int getnthreads() {return thread::hardware_concurrency(); }

    const int from(int t) const { return t*segSz; }
    const int to(int t) const { return ((t==nth-1) ? size : (t+1)*segSz); }

    void run(std::function<void(int, int, int)> const& lambda) {
        for (int t=0; t<nth; t++) {
            threads[t]=thread([this, lambda, t](){
                lambda(t, from(t), to(t));
            });
        }
        for (int t=0; t<nth; t++) threads[t].join();
    }
    void run(std::function<void(int)> const& lambda) {
        for (int t=0; t<nth; t++) {
            threads[t]=thread([this, lambda, t](){
                for (int i=from(t); i<to(t); i++)
                    lambda(i);
            });
        }
        for (int t=0; t<nth; t++) threads[t].join();
    }
    void run(std::function<void(void)> const& lambda) {
        for (int t=0; t<nth; t++) {
            threads[t]=thread([this, lambda, t](){
                for (int i=from(t); i<to(t); i++)
                    lambda();
            });
        }
        for (int t=0; t<nth; t++) threads[t].join();
    }
    int nth, segSz, size;
    thread *threads;

};


class Random {

    p::object own;
    float *rand_vect=nullptr;
    int g_seed=time(0);

public:
    Random() {
        Py_Initialize(); // init boost & numpy boost
        np::initialize();
    }
    ~Random() {
        if (rand_vect) delete[]rand_vect;
    }

    inline float fastrand() {
      g_seed = (214013*g_seed+2531011);
      return float((g_seed>>16)&0x7FFF) / 0x7fff;
    }

    np::ndarray random_cpp(int n) { // return numpy array to direct plot, image->numpy
         if (rand_vect) delete[]rand_vect; // self manage memory
         rand_vect=new float[n];

        Thread(n).run([&](int t, int from, int to){
            for (int i=from; i<to; i++)
                rand_vect[i]=fastrand();
        });

        return np::from_data(rand_vect,      // data ->
            np::dtype::get_builtin<float>(), // dtype -> float32
            p::make_tuple(n),                // shape -> n
            p::make_tuple(sizeof(float)), own);        // stride in bytes n*4, float
    }

    inline float sqr(float x) { return x*x; }

    // return mean, std from 1d numpy array
    p::tuple stat_cpp(np::ndarray v) {
        boost::atomic<double> s, sq;

        int n=v.shape(0);
        float *vf=(float*)v.get_data();

        Thread(n).run([&](int t, int from, int to) {
            double _s=0, _sq=0;
            for (int i=from; i<to; i++) {
                _s  += vf[i];
                _sq += sqr(vf[i]);
            }
            sq+=_sq;
            s+=_s;
        });

        return p::make_tuple( s/n, sqrt((sq - sqr(s) / n) / (n - 1)) );
    }
    unsigned long xorshf96(void) {          //period 2^96-1
        static unsigned long x=123456789, y=362436069, z=521288629;
        unsigned long t;

        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        t = x;    x = y;    y = z;
        return z = t ^ x ^ y;
    }
};


// aux funcs
static int g_seed=time(0); // must be external to func -> performance
static inline float fastrand() {
      g_seed = (214013*g_seed+2531011);
      return float((g_seed>>16)&0x7FFF) / 0x7fff;
}

inline float sqr(float x) { return x*x; }


// module funcs: init, randf, statf
static void init() {
    np::initialize();
}

// generate 1d size(n) numpy array
static np::ndarray randf(int n) {
    init();

    auto npv = np::empty( p::make_tuple(n),                // shape -> n
                         np::dtype::get_builtin<float>()); // dtype -> float32

    float*rand_vect=(float*)npv.get_data();

    Thread(n).run([&](int t, int from, int to) {
            for (int i=from; i<to; i++)
                rand_vect[i] = fastrand();
    });

    return npv;
}

// return mean, std from 1d numpy array
static p::tuple statf(np::ndarray &v) {
//    np::initialize(); // must call init before statf

    boost::atomic<double> s, sq;

    int n=v.shape(0);
    float *vf=(float*)v.get_data();

    Thread(n).run([&](int t, int from, int to) {
        double _s=0, _sq=0;
        for (int i=from; i<to; i++) {
            _s  += vf[i];
            _sq += sqr(vf[i]);
        }
        sq+=_sq;
        s+=_s;
    });

    return p::make_tuple( s/n, sqrt((sq - sqr(s) / n) / (n - 1)) );
}



BOOST_PYTHON_MODULE(random_cpp) {
   p::def("init", init);
   p::def("randf", randf);
   p::def("statf", statf);
}
