from datetime import datetime
import os
from statistics import stdev
from pathlib import Path

from .trainer import Trainer

from torch.cuda import Event
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from torchbenchmark.util.e2emodel import E2EBenchmarkModel

import torch
import torch.distributed as dist

class DeepSpeedTrainer(Trainer):
    DEFAULT_MEASURE_ITERATIONS = 10
    def __init__(self, args, model_class, batch_size=None, extra_args=[]):
        super().__init__(args, model_class, mode="SPMD")
        
        # TODO(whc) less hacky way to inject deepspeed config.  Right now it needs
        # to be passed to model_class ctor, but there isn't an arg to pass it via,
        # and it only works for hf_ models not all e2e models
        os.environ["USE_DEEPSPEED"] = "true"

        self.setup()
        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor
        model: E2EBenchmarkModel = model_class("train", batch_size, extra_args)

        expected_attrs = ["model", "optimizer", "train_dataloader", "accelerator"]
        assert all(attr in dir(model) for attr in expected_attrs), (
            "Missing attributes in the input E2EBenchmarkModel implementation: "
            f"{[attr for attr in expected_attrs if attr not in dir(model)]}"
        )

        self.model = model.model
        self.optimizer = model.optimizer
        self.dataloader = model.train_dataloader
        self.accelerator = model.accelerator

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
    
    def next_batch(self):
        return next(iter(self.dataloader))

    def forward(self, input):
        """
        compute model forward and return loss
        """
        # TODO(whc) bug: replace w/ self.ddp_model and/or use Accelerator for DDP
        return self.model(input).loss
    
    def backward(self, loss):
        self.accelerator.backward(loss)

    def optimizer_step(self):
        self.optimizer.step()

def test():
    from torchbenchmark.e2e_models.hf_bert import Model

    import os

    os.environ["LOCAL_RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    os.environ["RANK"] = str(0)

    trainer = DeepSpeedTrainer(Model)

    trainer.measure()

if __name__=="__main__":
    test()

