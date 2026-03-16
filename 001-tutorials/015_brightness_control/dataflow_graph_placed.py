#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

import aie.utils.trace as trace_utils

def aie_kernelAIE2(dev, width, height, ctrl):
    trace_size = 8192
    lineWidthInBytes = width
    tensorSize = 5760 * height

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(5760,), np.dtype[np.uint8]]
        line_ty = np.ndarray[(1920,), np.dtype[np.uint8]]

        # AIE Core Function declarations
        aie_kernelLine = external_func(
            "aie_kernelLine", inputs=[line_ty, line_ty, np.int32, np.int32]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0,1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        ComputeTile4 = tile(0, 4)

        tiles_to_trace = [ShimTile,MemTile,ComputeTile2,ComputeTile3,ComputeTile4]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)


        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, MemTile, 2, tensor_ty)
        of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, line_ty)
        of_in2 = object_fifo("in2", MemTile, ComputeTile3, 2, line_ty)
        of_in3 = object_fifo("in3", MemTile, ComputeTile4, 2, line_ty)
        object_fifo_link(of_in, [of_in1,of_in2,of_in3], [], [0,1920,3840])

        of_out1 = object_fifo("out1", ComputeTile2, MemTile, 2, line_ty)
        of_out2 = object_fifo("out2", ComputeTile3, MemTile, 2, line_ty)
        of_out3 = object_fifo("out3", ComputeTile4, MemTile, 2, line_ty)
        of_out = object_fifo("out", MemTile, ShimTile, 2, tensor_ty)
        object_fifo_link([of_out1,of_out2,of_out3],of_out,[0,1920,3840],[])
        
        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "aie_kernel.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(height):
                    elemOut = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    aie_kernelLine(elemIn, elemOut, width, ctrl)
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

        # Compute tile 3
        @core(ComputeTile3, "aie_kernel.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(height):
                    elemOut1 = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    elemIn1 = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    aie_kernelLine(elemIn1, elemOut1, width, ctrl)
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out2.release(ObjectFifoPort.Produce, 1)

        # Compute tile 4
        @core(ComputeTile4, "aie_kernel.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(height):
                    elemOut2 = of_out3.acquire(ObjectFifoPort.Produce, 1)
                    elemIn2 = of_in3.acquire(ObjectFifoPort.Consume, 1)
                    aie_kernelLine(elemIn2, elemOut2, width, ctrl)
                    of_in3.release(ObjectFifoPort.Consume, 1)
                    of_out3.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, notUsed, outTensor):
            if trace_size > 0:
                data_trace=trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )
            
            npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=inTensor, sizes=[1,1,1,tensorSize], issue_token=True)
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=outTensor, sizes=[1,1,1,tensorSize])
            dma_wait(of_in, of_out)
            trace_utils.gen_trace_done_aie2(ShimTile)


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1_1col
    elif device_name == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 512 if (len(sys.argv) != 5) else int(sys.argv[2])
    height = 9 if (len(sys.argv) != 5) else int(sys.argv[3])
    ctrl = 255 if (len(sys.argv) != 5) else int(sys.argv[4])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    aie_kernelAIE2(dev, width, height, ctrl)
    print(ctx.module)
