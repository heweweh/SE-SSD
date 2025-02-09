import os.path as osp

import torch

from ...utils import master_only
from .base import LoggerHook


class TensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, trainer):
        if torch.__version__ >= "1.1":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError('Please run "pip install future tensorboard" to install the dependencies to use torch.utils.tensorboard (applicable to PyTorch 1.1 or higher)')
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError("Please install tensorboardX to use " "TensorboardLoggerHook.")

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, trainer):
        for var in trainer.log_buffer.output:
            if var in ["time", "data_time"]:
                continue
            tag = "{}/{}".format(var, trainer.mode)
            record = trainer.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, trainer.iter)
            elif isinstance(record, list):
                for item in record:
                    self.writer.add_scalar(tag, item, trainer.iter)
            else:
                self.writer.add_scalar(tag, trainer.log_buffer.output[var], trainer.iter)

    @master_only
    def after_run(self, trainer):
        self.writer.close()
