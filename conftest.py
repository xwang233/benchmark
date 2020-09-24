import os
import pytest
import torch

def pytest_addoption(parser):
    parser.addoption("--fuser", help="fuser to use for benchmarks")
    parser.addoption("--bailout_depth", type=int, help="setting for te fuser")

def set_fuser(fuser, bailout_depth=20):
    if fuser == "old":
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser == "te":
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(bailout_depth)
        print(f"using bailout_depth {bailout_depth}")
        torch._C._jit_set_num_profiled_runs(2)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)

def pytest_configure(config):
    set_fuser(config.getoption("fuser"), config.getoption("bailout_depth"))

def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info['pytorch_version'] = torch.__version__
    try:
        import torchtext
        machine_info['torchtext_version'] = torchtext.__version__
    except ImportError:
        machine_info['torchtext_version'] = '*not-installed*'

    try:
        import torchvision
        machine_info['torchvision_version'] = torchvision.__version__
    except ImportError:
        machine_info['torchvision_version'] = '*not-installed*'

    machine_info['circle_build_num'] = os.environ.get("CIRCLE_BUILD_NUM")
    machine_info['circle_project_name'] = os.environ.get("CIRCLE_PROJECT_REPONAME")
