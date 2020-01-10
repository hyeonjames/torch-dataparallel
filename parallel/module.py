import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Any

import torch
import torch.nn as nn


class DataParallelFuture(nn.Module):
    def __init__(self, module: nn.Module, devices: Union[None, List[Union[int, torch.device]]] = None,
                 output_device: Union[int, torch.device] = None) -> None:
        super(DataParallelFuture, self).__init__()

        if not torch.cuda.is_available():
            raise EnvironmentError("cuda is not available.")
            return

        if not devices:
            devices = [torch.device(x) for x in range(torch.cuda.device_count())]
        if not output_device:
            output_device = devices[0]

        self.module = module
        self.devices = devices
        self.output_device = output_device

    def parameters(self, *inputs):
        return self.module.parameters(*inputs)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        scattered = nn.parallel.scatter(inputs, self.devices)
        replicas = nn.parallel.replicate(self.module, self.devices)

        def model_run(module: nn.Module, inputs: List[Any]):
            return module(*inputs, **kwargs)

        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            results = []
            for idx, (x, device, module) in enumerate(zip(scattered, self.devices, replicas)):
                futures.append(executor.submit(model_run, module, x))
            for future in futures:
                results.append(future.result())
        return nn.parallel.scatter_gather.gather(results, target_device=self.output_device)


class DataParallelThreading(nn.Module):
    def __init__(self, module: nn.Module, devices: Union[None, List[Union[int, torch.device]]] = None,
                 output_device: Union[int, torch.device] = None) -> None:
        super(DataParallelThreading, self).__init__()
        if not torch.cuda.is_available():
            raise EnvironmentError("cuda is not available.")
            return
        if not devices:
            devices = [torch.device(x) for x in range(torch.cuda.device_count())]
        if not output_device:
            output_device = devices[0]

        self.module = module
        self.devices = devices
        self.output_device = output_device

    def parameters(self, *inputs: Any):
        return self.module.parameters(*inputs)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        scattered = nn.parallel.scatter(inputs, self.devices)
        replicas = nn.parallel.replicate(self.module, self.devices)
        lock = threading.Lock()
        results = []
        threads = []

        def model_run(module: nn.Module, index: int, inputs: List[Any]):
            result = module(*inputs, **kwargs)
            with lock:
                results.append(result)

        for idx, (x, device, module) in enumerate(zip(scattered, self.devices, replicas)):
            t = threading.Thread(target=model_run, args=(module, idx, x))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        return nn.parallel.gather(results, target_device=self.output_device)
