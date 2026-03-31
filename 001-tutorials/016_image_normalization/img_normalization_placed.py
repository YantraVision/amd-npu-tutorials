
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

def distribute_join_L2_4cols_simple(dev, width_val, height_val):
    width = width_val
    height = height_val
    width_header = width+64
    lineWidthInBytes = width
    tensorSize = width * height
    tensorSize_max = 64 * height
    tensorSize_norm = (width+64) * height

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(width,), np.dtype[np.uint8]]
        line_ty = np.ndarray[(width,), np.dtype[np.uint8]]
        line_header_ty = np.ndarray[(width_header,), np.dtype[np.uint8]]
        line_max_ty = np.ndarray[(64,), np.dtype[np.uint8]]

        maxKernel = external_func(
            "maxKernel", inputs=[line_ty, line_max_ty, np.int32]
        )
        normKernel = external_func(
            "normKernel", inputs=[line_header_ty, line_ty, np.int32]
        )

        # ---------------------------------------------------------
        # TILE DECLARATIONS 
        # ---------------------------------------------------------

        # Column 0
        Shim0 = tile(0, 0)
        Mem0  = tile(0, 1)
        C0_0  = tile(0, 2)

        # Column 1
        Shim1 = tile(1, 0)
        C1_0  = tile(1, 2)

        # ---------------------------------------------------------
        # FIFOS
        # ---------------------------------------------------------

        # Column 0
        in0  = object_fifo("in0",  Shim0, Mem0, 2, tensor_ty)
        in0_0 = object_fifo("in0_0", Mem0, C0_0, 2, line_ty)
        object_fifo_link(in0, [in0_0], [], [0])

        out0  = object_fifo("out0", Mem0, Shim0, 2, line_max_ty)
        out0_0 = object_fifo("out0_0", C0_0, Mem0, 2, line_max_ty)
        object_fifo_link([out0_0], out0, [0], [])

        # Column 1
        in1  = object_fifo("in1",  Shim1, C1_0, 2, line_header_ty)

        out1  = object_fifo("out1", C1_0, Shim1, 2, line_ty)

        # ---------------------------------------------------------
        # CORE - PROCESS
        # ---------------------------------------------------------
        
        # Column 0
        @core(C0_0, "normalization.cc.o")
        def core0_0():
        	for _ in range_(sys.maxsize):
                    for _ in range_(height):
                        x = in0_0.acquire(ObjectFifoPort.Consume, 1)
                        y = out0_0.acquire(ObjectFifoPort.Produce, 1)
                        maxKernel(x, y, width)
                        in0_0.release(ObjectFifoPort.Consume, 1)
                        out0_0.release(ObjectFifoPort.Produce, 1)


        # Column 1
        @core(C1_0, "normalization.cc.o")
        def core1_0():
            for _ in range_(sys.maxsize):
                for _ in range_(height):
                    x = in1.acquire(ObjectFifoPort.Consume, 1)
                    y = out1.acquire(ObjectFifoPort.Produce, 1)
                    normKernel(x, y, width_header)
                    in1.release(ObjectFifoPort.Consume, 1)
                    out1.release(ObjectFifoPort.Produce, 1)


        # ---------------------------------------------------------
        # RUNTIME SEQUENCE                                
        # ---------------------------------------------------------
                                                          
        @runtime_sequence(tensor_ty,line_header_ty,line_max_ty,tensor_ty)
        def sequence(i0,i1,o0,o1):

            # Column 0
            npu_dma_memcpy_nd(metadata=in0, bd_id=1, mem=i0, sizes=[1,1,1,tensorSize], issue_token=True)

            npu_dma_memcpy_nd(metadata=out0, bd_id=0, mem=o0, sizes=[1,1,1,tensorSize_max])
            
            # Column 1
            npu_dma_memcpy_nd(metadata=in1, bd_id=1, mem=i1, sizes=[1,1,1,tensorSize_norm], issue_token=True)
            npu_dma_memcpy_nd(metadata=out1, bd_id=0, mem=o1, sizes=[1,1,1,tensorSize])

            dma_wait(in0, out0)
            dma_wait(in1, out1)




try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1
    elif device_name == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 256 if (len(sys.argv) != 4) else int(sys.argv[2])
    height = 256 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    distribute_join_L2_4cols_simple(dev, width, height)
    print(ctx.module)
