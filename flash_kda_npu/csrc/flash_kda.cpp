#include <acl/acl_rt.h>
#include <acl/acl_rt_compile.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Kernel {
    aclrtFuncHandle function;
    aclrtBinHandle binary;
    std::string source;
    std::vector<size_t> argSizes;
};
using RuntimeKey = std::pair<uintptr_t, int32_t>;
// Keep handles for the process lifetime to avoid racing runtime teardown.
// GIL-protected entry points serialize access to the handle cache.
std::map<RuntimeKey, std::unordered_map<std::string, Kernel>> g_funcs;
std::unordered_map<std::string, std::vector<char>> g_binaries;
constexpr size_t MAX_ARGS = 32;

void Check(aclError err, const std::string &what) {
    if (err != ACL_SUCCESS) {
        throw std::runtime_error(what + " failed, aclError=" + std::to_string(err));
    }
}

RuntimeKey CurrentRuntime() {
    aclrtContext cur = nullptr;
    Check(aclrtGetCurrentContext(&cur), "aclrtGetCurrentContext (initialize torch.npu first)");
    if (cur == nullptr) {
        throw std::runtime_error("no current NPU context; initialize torch.npu first");
    }
    int32_t device = -1;
    Check(aclrtGetDevice(&device), "aclrtGetDevice");
    return {reinterpret_cast<uintptr_t>(cur), device};
}

struct Program {
    aclrtcProg value = nullptr;
    ~Program() { if (value != nullptr) aclrtcDestroyProg(&value); }
};

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("runtime_key", [] { return CurrentRuntime(); });
    m.def(
        "rtc_compile",
        [](const std::string &src, const std::string &funcName,
           const std::vector<size_t> &argSizes) {
            const auto runtime = CurrentRuntime();
            if (argSizes.empty() || argSizes.size() > MAX_ARGS) {
                throw std::invalid_argument("kernel ABI must have 1..32 arguments");
            }
            for (auto size : argSizes) {
                if (size != 4 && size != 8) throw std::invalid_argument("unsupported ABI argument size");
            }
            // Derive toolkit paths at runtime so RTC supports relocated installs
            // and different host architectures.
            const char *cannHomeEnv = std::getenv("ASCEND_HOME_PATH");
            const std::string cannHome =
                (cannHomeEnv != nullptr && cannHomeEnv[0] != '\0')
                    ? cannHomeEnv
                    : "/usr/local/Ascend/ascend-toolkit/latest";
            const std::string ascBase = cannHome +
#if defined(__aarch64__)
                "/aarch64-linux";
#else
                "/x86_64-linux";
#endif
            std::vector<std::string> optStrs = {
                "--npu-soc=Ascend910B3", "-O3", "-std=c++17",
                "-include" + cannHome + "/include/version/asc_devkit_version.h",
            };
            for (const char *sub : {"tikcpp/tikcfw", "tikcpp/tikcfw/interface",
                                     "tikcpp/tikcfw/impl", "asc", "asc/include",
                                     "asc/include/basic_api", "asc/include/adv_api",
                                     "asc/include/c_api", "asc/impl/basic_api",
                                     "asc/impl/adv_api", "asc/impl/c_api",
                                     "asc/impl/utils"}) {
                optStrs.push_back("-I" + ascBase + "/" + sub);
            }
            if (const char *extra = std::getenv("KDA_RTC_OPTS")) {
                std::stringstream ss(extra);
                std::string tok;
                while (ss >> tok) {
                    optStrs.push_back(tok);
                }
            }
            std::string key = funcName + '\0' + src;
            for (const auto &option : optStrs) key += '\0' + option;
            auto &functions = g_funcs[runtime];
            const auto loaded = functions.find(funcName);
            if (loaded != functions.end()) {
                if (loaded->second.source != key || loaded->second.argSizes != argSizes) {
                    throw std::invalid_argument("kernel already loaded with different source/options/ABI; use a fresh process");
                }
                return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(loaded->second.function));
            }
            auto cached = g_binaries.find(key);
            if (cached == g_binaries.end()) {
                Program owner;
                auto &prog = owner.value;
                Check(aclrtcCreateProg(&prog, src.c_str(), funcName.c_str(), 0, nullptr, nullptr),
                      "aclrtcCreateProg");
                std::vector<const char *> options;
                for (auto &o : optStrs) {
                    options.push_back(o.c_str());
                }
                aclError e = aclrtcCompileProg(prog, static_cast<int>(options.size()), options.data());
                if (e != ACL_SUCCESS) {
                    size_t logSize = 0;
                    aclrtcGetCompileLogSize(prog, &logSize);
                    std::string log(logSize, '\0');
                    if (logSize > 0) {
                        aclrtcGetCompileLog(prog, &log[0]);
                    }
                    throw std::runtime_error("aclrtcCompileProg failed, log:\n" + log);
                }
                size_t binSize = 0;
                Check(aclrtcGetBinDataSize(prog, &binSize), "aclrtcGetBinDataSize");
                std::vector<char> bin(binSize);
                Check(aclrtcGetBinData(prog, bin.data()), "aclrtcGetBinData");
                cached = g_binaries.emplace(key, std::move(bin)).first;
            }
            const auto &bin = cached->second;
            const auto binSize = bin.size();
            aclrtBinaryLoadOption opt;
            opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
            opt.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
            aclrtBinaryLoadOptions opts;
            opts.options = &opt;
            opts.numOpt = 1;
            aclrtBinHandle bh = nullptr;
            Check(aclrtBinaryLoadFromData(bin.data(), binSize, &opts, &bh), "aclrtBinaryLoadFromData");
            aclrtFuncHandle fn = nullptr;
            const auto error = aclrtBinaryGetFunction(bh, funcName.c_str(), &fn);
            if (error != ACL_SUCCESS) {
                aclrtBinaryUnLoad(bh);
                Check(error, "aclrtBinaryGetFunction");
            }
            functions.emplace(funcName, Kernel{fn, bh, key, argSizes});
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fn));
        },
        py::arg("src"), py::arg("func_name"), py::arg("arg_sizes"));
    m.def(
        "launch_argsarray_engine",
        [](const std::string &name, uint32_t numBlocks, uint64_t stream,
           const std::vector<std::string> &argBlobs, int32_t engine) {
            auto &functions = g_funcs[CurrentRuntime()];
            auto it = functions.find(name);
            if (it == functions.end()) {
                throw std::runtime_error("kernel not loaded in the current device/context: " + name);
            }
            if (numBlocks == 0 || (engine != ACL_RT_ENGINE_TYPE_AIC && engine != ACL_RT_ENGINE_TYPE_AIV)) {
                throw std::invalid_argument("invalid blocks or engine");
            }
            const auto &sizes = it->second.argSizes;
            if (argBlobs.size() != sizes.size()) {
                throw std::invalid_argument("kernel " + name + " expects " + std::to_string(sizes.size()) + " arguments, got " + std::to_string(argBlobs.size()));
            }
            std::array<void *, MAX_ARGS> args{};
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (argBlobs[i].size() != sizes[i]) {
                    throw std::invalid_argument("kernel " + name + " argument " + std::to_string(i) + " expects " + std::to_string(sizes[i]) + " bytes, got " + std::to_string(argBlobs[i].size()));
                }
                args[i] = const_cast<char *>(argBlobs[i].data());
            }
            aclrtLaunchKernelAttr attr;
            attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
            attr.value.engineType = static_cast<aclrtEngineType>(engine);
            aclrtLaunchKernelCfg cfg;
            cfg.attrs = &attr;
            cfg.numAttrs = 1;
            Check(aclrtLaunchKernelWithArgsArray(it->second.function, numBlocks,
                                                 reinterpret_cast<aclrtStream>(stream), &cfg,
                                                 args.data()),
                  "aclrtLaunchKernelWithArgsArray engine " + name);
        },
        py::arg("name"), py::arg("num_blocks"), py::arg("stream"), py::arg("arg_bytes"), py::arg("engine"));
}
