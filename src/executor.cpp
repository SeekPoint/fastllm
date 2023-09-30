//
// Created by huangyuyang on 6/13/23.
//

#include "utils.h"

#include "executor.h"

#include "devices/cpu/cpudevice.h"

#ifdef USE_CUDA
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
    Executor::Executor() {
        this->devices.clear();
#ifdef USE_CUDA
        // 将一个指向 CudaDevice 类对象的指针插入到 devices 向量的末尾。
        // 这里通过 new 运算符创建了一个 CudaDevice 对象，并将返回的指针进行类型转换为 BaseDevice* 类型。
        this->devices.push_back((BaseDevice*) new CudaDevice());
#endif
        this->devices.push_back((BaseDevice*) new CpuDevice());
    }

    Executor::~Executor() {
        // 释放 devices 向量中的每个指针元素所占用的内存。
        for (int i = 0; i < devices.size(); i++) {
            delete devices[i];
        }
    }

    void Executor::ClearDevices() {
        // this->devices 指的是当前对象的 devices 成员，即指向 BaseDevice 类对象的指针向量。
        this->devices.clear();
    }

    // 该函数用于向 devices 向量中添加一个指向 BaseDevice 类对象的指针。
    void Executor::AddDevice(fastllm::BaseDevice *device) {
        this->devices.push_back(device);
    }

    void Executor::SetFirstDevice(const std::string &device) {
        auto temp = this->devices;
        this->devices.clear();
        for (int i = 0; i < temp.size(); i++) {
            if (StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
                this->devices.back()->deviceIds = ParseDeviceIds(device, temp[i]->deviceType);
            }
        }
        for (int i = 0; i < temp.size(); i++) {
            if (!StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
            }
        }
    }

    std::vector <int> Executor::GetDeviceIds(const std::string &device) {
        for (int i = 0; i < devices.size(); i++) {
            if (StartWith(devices[i]->deviceType, device)) {
                return devices[i]->deviceIds;
            }
        }
        return {0};
    }

    void Executor::Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams) {

        // 创建一个 st 变量，用于记录函数开始执行的时间。
        auto st = std::chrono::system_clock::now();

        // 创建一个布尔变量 lockInCPU，用于记录是否将数据锁定在 CPU 上。
        bool lockInCPU = false;

        // 在第一个 for 循环中，遍历数据字典 datas，查找是否有 "___batch" 后缀的参数，
        // 并根据情况设置 lockInCPU 的值。it.first 是数据字典中的键（key），it.second
        // 是对应的值（value）。如果存在 "___batch" 后缀的参数，则将 lockInCPU 设置为
        // 对应数据的 lockInCPU 属性（布尔值），否则设置为当前数据的 lockInCPU 属性。
        for (auto &it: datas) {
            if (intParams.find(it.first + "___batch") != intParams.end()) {
                int batch = intParams.find(it.first + "___batch")->second;
                for (int i = 0; i < batch; i++) {
                    lockInCPU |= (((Data**)it.second)[i] && ((Data**)it.second)[i]->lockInCPU);
                }
            } else {
                lockInCPU |= (it.second && it.second->lockInCPU);
            }
        }

        // 第二个 for 循环遍历 devices 向量中的所有设备指针 device。
        // 在循环中，首先检查 lockInCPU 是否为真，并且当前设备的类型不是 "cpu"，
        // 如果是，则跳过当前设备（continue）。这个检查是为了保证数据锁定在 CPU 上时，只执行 CPU 设备上的操作。
        for (auto device: devices) {
            if (lockInCPU && device->deviceType != "cpu") {
                continue;
            }
            // 然后，通过调用 device->CanRun(opType, datas, floatParams, intParams)
            // 检查当前设备是否可以运行指定的操作 opType。如果可以运行，则进行以下操作：
            if (device->CanRun(opType, datas, floatParams, intParams)) {
#ifdef USE_CUDA
                if (device->deviceType == "cuda" && device->deviceIds.size() > 0) {
                    FastllmCudaSetDevice(device->deviceIds[0]);
                }
#endif
                // 第三个 for 循环遍历数据字典 datas，如果存在 "___batch" 后缀的参数，
                // 则将对应数据转移到当前设备上；否则，将当前数据转移到当前设备上。
                for (auto &it: datas) {
                    if (intParams.find(it.first + "___batch") != intParams.end()) {
                        int batch = intParams.find(it.first + "___batch")->second;
                        for (int i = 0; i < batch; i++) {
                            if (((Data**)it.second)[i]) {
                                ((Data**)it.second)[i]->ToDevice((void *) device);
                            }
                        }
                    } else {
                        if (it.second) {
                            it.second->ToDevice((void *) device);
                        }
                    }
                }
                // 调用 device->Reshape(opType, datas, floatParams, intParams)
                // 进行形状推导，device上的形状推导调用了opType对应的op的形状推导，
                // 并且被各个不同的op重写。
                device->Reshape(opType, datas, floatParams, intParams);

                // 对opType对应的这个算子进行推理。
                device->Run(opType, datas, floatParams, intParams);
                break;
            }
        }

        // 最后，计算操作运行时间，并将其加入 profiler 成员变量，用于性能分析。
        float spend = GetSpan(st, std::chrono::system_clock::now());
        profiler[opType] += spend;
    }

    // 清除profile的信息
    void Executor::ClearProfiler() {
        profiler.clear();
    }

    // 打印profile信息，也即输出每个层的运行时间和模型的总运行时间
    void Executor::PrintProfiler() {
        float sum = 0.0;
        for (auto &it : profiler) {
            printf("%s spend %f\n", it.first.c_str(), it.second);
            sum += it.second;
        }
        printf("total spend %f\n", sum);
    }
}