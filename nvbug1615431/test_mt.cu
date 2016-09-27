#include <string>
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <chrono>

using namespace std;

atomic<int> epoch;
atomic<int> count;
int num_threads;
int duration;
static int TOTAL_ITERATIONS_THREAD = 100;
static int TOTAL_ITERATIONS_PROGRAM = 100;
mutex thread_mutex;
bool serial;

double total_time_sync;
double total_time_launch;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ unsigned long long int gclock64() {
    unsigned long long int rv;
    asm volatile ( "mov.u64 %0, %%globaltimer;" : "=l"(rv) );
    return rv;
}

__global__ 
void spin(int *a, int runtime)
{
    unsigned long long int start_clock = gclock64();
    unsigned long long int clock_offset = 0;
    while (clock_offset < runtime)
    {
        clock_offset = gclock64() - start_clock;
    }
    a[0] = clock_offset;
}

void barrier(int& local_epoch)
{
    int count_snapshot = count.fetch_add(1);
    if(count_snapshot == num_threads - 1)
    {
        count = 0;
        epoch ++;
    }
    while(local_epoch != epoch) {}

    local_epoch++;
}

void task(int tid)
{
    int local_epoch = 1;
    int iteration = 0;
    double elapsed_time_sync = 0;
    double elapsed_time_launch = 0;

    cudaEvent_t event;
    cudaStream_t st;
    cudaStreamCreate(&st);

    int* t;
    gpuErrChk(cudaMalloc(&t, sizeof(int)));
    gpuErrChk(cudaMemset(t, 0, sizeof(int)));

    gpuErrChk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); 

    while(iteration < TOTAL_ITERATIONS_THREAD)
    {
        barrier(local_epoch);

        chrono::duration<double, micro> usec;

        if(serial) thread_mutex.lock();

        auto start_time_launch = chrono::high_resolution_clock::now();
        spin<<<1,1,0, st>>>(t, duration);
        auto end_time_launch = chrono::high_resolution_clock::now();
       
        //gpuErrChk(cudaEventRecord(event, st));
        auto start_time_sync = chrono::high_resolution_clock::now();
        //gpuErrChk(cudaEventSynchronize(event));
        gpuErrChk(cudaStreamSynchronize(st));
        auto end_time_sync = chrono::high_resolution_clock::now();


        
        if(serial) thread_mutex.unlock();

        usec = end_time_sync - start_time_sync;
        elapsed_time_sync = elapsed_time_sync + usec.count();
        usec = end_time_launch - start_time_launch;
        elapsed_time_launch = elapsed_time_launch + usec.count();
        
        iteration++;
    }

    // Report latency
    thread_mutex.lock();
    //cout << "Thread " << tid << ", launch: " << (float)elapsed_time_launch/(float)iteration
    //                         << ", sync: "   << (float)elapsed_time_sync/(float)iteration << endl;
    total_time_launch = total_time_launch + elapsed_time_launch/(double)iteration;
    total_time_sync = total_time_sync + elapsed_time_sync/(double)iteration;
    thread_mutex.unlock();

    cudaStreamDestroy(st);
   
}

int main(int argc, char* argv[])
{
  num_threads = 2;
  duration = 1000;
  serial = false;
  if (argc > 1)
  {
    num_threads = atoi(argv[1]);
  }
  if (argc > 2)
  {
    duration = atoi(argv[2]);
  }
  if(argc > 3)
  {
    serial = (atoi(argv[3]) == 1);
  }

    std::cout << "num_threads= " << num_threads << std::endl;
    std::cout << "duration= " << duration << std::endl;
    std::cout << "serial= " << serial << std::endl;


    for(int n = 0; n < TOTAL_ITERATIONS_PROGRAM; n++)
    {
        count = 0;
        epoch = 0;
        vector<thread*> t;
        t.resize(num_threads);

        for(int i = 0; i < num_threads; i++)
        {
            t[i] = new thread(task, i);
        }

        for(int i = 0; i < num_threads; i++)
        {
            t[i]->join();
            delete t[i];
        }
    }

    cout << num_threads << " " << serial <<
            " Launch: " << total_time_launch/(double)(TOTAL_ITERATIONS_PROGRAM*num_threads) <<
            " Sync: " << total_time_sync/(double)(TOTAL_ITERATIONS_PROGRAM*num_threads) << endl;

}
