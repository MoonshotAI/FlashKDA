import os
import torch


def pytest_configure(config):
    if os.environ.get("FLASH_KDA_DIST_GPU") != "1":
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker is not None:
        gpu_count = torch.cuda.device_count()
        worker_id = int(worker.replace("gw", ""))
        gpu_id = worker_id % gpu_count
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)
