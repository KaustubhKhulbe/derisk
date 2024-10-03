"""
PyTorch Profiler With TensorBoard
====================================
This tutorial demonstrates how to use TensorBoard plugin with PyTorch Profiler
to detect performance bottlenecks of the model.

Introduction
------------
PyTorch 1.8 includes an updated profiler API capable of
recording the CPU side operations as well as the CUDA kernel launches on the GPU side.
The profiler can visualize this information
in TensorBoard Plugin and provide analysis of the performance bottlenecks.

In this tutorial, we will use a simple Resnet model to demonstrate how to
use TensorBoard plugin to analyze model performance.

Setup
-----
To install ``torch`` and ``torchvision`` use the following command:

.. code-block::

   pip install torch torchvision


"""


######################################################################
# Steps
# -----
#
# 1. Prepare the data and model
# 2. Use profiler to record execution events
# 3. Run the profiler
# 4. Use TensorBoard to view results and analyze model performance
# 5. Improve performance with the help of profiler
# 6. Analyze performance with other advanced features
# 7. Additional Practices: Profiling PyTorch on AMD GPUs
#
# 1. Prepare the data and model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, import all necessary libraries:
#

import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

######################################################################
# Then prepare the input data. For this tutorial, we use the CIFAR10 dataset.
# Transform it to the desired format and use ``DataLoader`` to load each batch.

transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

######################################################################
# Next, create Resnet model, loss function, and optimizer objects.
# To run on GPU, move model and loss to GPU device.

device = torch.device("cuda:0")
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()


######################################################################
# Define the training step for each batch of input data.

def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


######################################################################
# 2. Use profiler to record execution events
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The profiler is enabled through the context manager and accepts several parameters,
# some of the most useful are:
#
# - ``schedule`` - callable that takes step (int) as a single parameter
#   and returns the profiler action to perform at each step.
#
#   In this example with ``wait=1, warmup=1, active=3, repeat=1``,
#   profiler will skip the first step/iteration,
#   start warming up on the second,
#   record the following three iterations,
#   after which the trace will become available and on_trace_ready (when set) is called.
#   In total, the cycle repeats once. Each cycle is called a "span" in TensorBoard plugin.
#
#   During ``wait`` steps, the profiler is disabled.
#   During ``warmup`` steps, the profiler starts tracing but the results are discarded.
#   This is for reducing the profiling overhead.
#   The overhead at the beginning of profiling is high and easy to bring skew to the profiling result.
#   During ``active`` steps, the profiler works and records events.
# - ``on_trace_ready`` - callable that is called at the end of each cycle;
#   In this example we use ``torch.profiler.tensorboard_trace_handler`` to generate result files for TensorBoard.
#   After profiling, result files will be saved into the ``./log/resnet18`` directory.
#   Specify this directory as a ``logdir`` parameter to analyze profile in TensorBoard.
# - ``record_shapes`` - whether to record shapes of the operator inputs.
# - ``profile_memory`` - Track tensor memory allocation/deallocation. Note, for old version of pytorch with version
#   before 1.10, if you suffer long profiling time, please disable it or upgrade to new version.
# - ``with_stack`` - Record source information (file and line number) for the ops.
#   If the TensorBoard is launched in VS Code (`reference <https://code.visualstudio.com/docs/datascience/pytorch-support#_tensorboard-integration>`_),
#   clicking a stack frame will navigate to the specific code line.

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        if step >= 1 + 1 + 3:
            break
        train(batch_data)