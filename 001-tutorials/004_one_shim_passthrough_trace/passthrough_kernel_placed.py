
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils


def my_passthrough_kernel(dev, in1_size, out_size, trace_size):
    in1_dtype = np.int32
    out_dtype = np.int32

    assert (
        out_size == in1_size
    ), "Output buffer size must be equal to input buffer size."

    @device(dev)
    def device_body():
        # define types
        vector_ty = np.ndarray[(64,), np.dtype[in1_dtype]]
        line_ty = np.ndarray[(16,), np.dtype[in1_dtype]]

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0,1)
        ComputeTile2 = tile(0, 2)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ShimTile,MemTile, ComputeTile2]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        of_in0 = object_fifo("in0", ShimTile, MemTile, 2, vector_ty)
        of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, line_ty)
        object_fifo_link(of_in0, of_in1)
        
        of_out0 = object_fifo("out0", ComputeTile2, MemTile, 2, line_ty)
        of_out1 = object_fifo("out1", MemTile, ShimTile, 2, vector_ty)
        object_fifo_link(of_out0, of_out1)


        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            for i in range(64):
                elemOut = of_out0.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in1.acquire(ObjectFifoPort.Consume, 1)
                for i in range(16):
                    elemOut[i] = elemIn[i]
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out0.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(vector_ty, vector_ty, vector_ty)
        def sequence(inTensor, outTensor, notUsed):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )
            in_task = shim_dma_single_bd_task(
                of_in0, inTensor, sizes=[1, 1, 1, 1024], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out1, outTensor, sizes=[1, 1, 1, 1024], issue_token=True
            )
            
            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 4:
    raise ValueError("[ERROR] Need at least 4 arguments (dev, in1_size, out_size)")


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
in1_size = int(opts.in1_size)
if in1_size % 64 != 0 or in1_size < 512:
    print(
        "In1 buffer size ("
        + str(in1_size)
        + ") must be a multiple of 64 and greater than or equal to 512"
    )
    raise ValueError
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    my_passthrough_kernel(dev, in1_size, out_size, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
