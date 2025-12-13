
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

def distribute_join_L2_4cols_simple():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1)
        def device_body():

            tile24 = np.ndarray[(48,), np.dtype[np.int32]]
            tile8  = np.ndarray[(16,),  np.dtype[np.int32]]

            # ---------------------------------------------------------
            # TILE DECLARATIONS
            # ---------------------------------------------------------

            # Column 0
            Shim0 = tile(0, 0)
            Mem0  = tile(0, 1)
            C0_0  = tile(0, 2)

            # Column 1
            Shim1 = tile(1, 0)
            Mem1  = tile(1, 1)
            C1_0  = tile(1, 2)

            # Column 2
            Shim2 = tile(2, 0)
            Mem2  = tile(2, 1)
            C2_0  = tile(2, 2)

            # Column 3
            Shim3 = tile(3, 0)
            Mem3  = tile(3, 1)
            C3_0  = tile(3, 2)


            # ---------------------------------------------------------
            # FIFOS
            # ---------------------------------------------------------

            # Column 0
            in0  = object_fifo("in0",  Shim0, Mem0, 2, tile24)
            in0_0 = object_fifo("in0_0", Mem0, C0_0, 2, tile8)
            object_fifo_link(in0, [in0_0], [], [0])

            out0  = object_fifo("out0", Mem0, Shim0, 2, tile24)
            out0_0 = object_fifo("out0_0", C0_0, Mem0, 2, tile8)
            object_fifo_link([out0_0], out0, [0], [])

            # Column 1
            in1  = object_fifo("in1",  Shim1, Mem1, 2, tile24)
            in1_0 = object_fifo("in1_0", Mem1, C1_0, 2, tile8)
            object_fifo_link(in1, [in1_0], [], [0])

            out1  = object_fifo("out1", Mem1, Shim1, 2, tile24)
            out1_0 = object_fifo("out1_0", C1_0, Mem1, 2, tile8)
            object_fifo_link([out1_0], out1, [0], [])

            # Column 2
            in2  = object_fifo("in2",  Shim2, Mem2, 2, tile24)
            in2_0 = object_fifo("in2_0", Mem2, C2_0, 2, tile8)
            object_fifo_link(in2, [in2_0], [], [0])

            out2  = object_fifo("out2", Mem2, Shim2, 2, tile24)
            out2_0 = object_fifo("out2_0", C2_0, Mem2, 2, tile8)
            object_fifo_link([out2_0], out2, [0], [])

            # Column 3
            in3  = object_fifo("in3",  Shim3, Mem3, 2, tile24)
            in3_0 = object_fifo("in3_0", Mem3, C3_0, 2, tile8)
            object_fifo_link(in3, [in3_0], [], [0])

            out3  = object_fifo("out3", Mem3, Shim3, 2, tile24)
            out3_0 = object_fifo("out3_0", C3_0, Mem3, 2, tile8)
            object_fifo_link([out3_0], out3, [0], [])

            # ---------------------------------------------------------
            # CORE — PROCESS
            # ---------------------------------------------------------
            
            # Column 0
            @core(C0_0)
            def core0_0():
                for _ in range_(3):
                    x = in0_0.acquire(ObjectFifoPort.Consume, 1)
                    y = out0_0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        y[i] = x[i]
                    in0_0.release(ObjectFifoPort.Consume, 1)
                    out0_0.release(ObjectFifoPort.Produce, 1)
            
            # Column 1
            @core(C1_0)
            def core1_0():
                for _ in range_(3):
                    x = in1_0.acquire(ObjectFifoPort.Consume, 1)
                    y = out1_0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        y[i] = x[i]
                    in1_0.release(ObjectFifoPort.Consume, 1)
                    out1_0.release(ObjectFifoPort.Produce, 1)
            
            # Column 2
            @core(C2_0)
            def core2_0():
                for _ in range_(3):
                    x = in2_0.acquire(ObjectFifoPort.Consume, 1)
                    y = out2_0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        y[i] = x[i]
                    in2_0.release(ObjectFifoPort.Consume, 1)
                    out2_0.release(ObjectFifoPort.Produce, 1)
            
            # Column 3
            @core(C3_0)
            def core3_0():
                for _ in range_(3):
                    x = in3_0.acquire(ObjectFifoPort.Consume, 1)
                    y = out3_0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        y[i] = x[i]
                    in3_0.release(ObjectFifoPort.Consume, 1)
                    out3_0.release(ObjectFifoPort.Produce, 1)

            # ---------------------------------------------------------
            # RUNTIME SEQUENCE
            # ---------------------------------------------------------

            data48 = np.ndarray[(48,), np.dtype[np.int32]]

            @runtime_sequence(data48,data48,data48,data48,data48)
            def sequence(i0,o0,o1,o2,o3):

                # Column 0
                npu_dma_memcpy_nd(metadata=in0, bd_id=1, mem=i0, sizes=[1,1,1,48], issue_token=True)
                npu_dma_memcpy_nd(metadata=out0, bd_id=0, mem=o0, sizes=[1,1,1,48])
                
                # Column 1
                npu_dma_memcpy_nd(metadata=in1, bd_id=1, mem=i0, sizes=[1,1,1,48], issue_token=True)
                npu_dma_memcpy_nd(metadata=out1, bd_id=0, mem=o1, sizes=[1,1,1,48])
                
                # Column 2
                npu_dma_memcpy_nd(metadata=in2, bd_id=1, mem=i0, sizes=[1,1,1,48], issue_token=True)
                npu_dma_memcpy_nd(metadata=out2, bd_id=0, mem=o2, sizes=[1,1,1,48])
                
                # Column 3
                npu_dma_memcpy_nd(metadata=in3, bd_id=1, mem=i0, sizes=[1,1,1,48], issue_token=True)
                npu_dma_memcpy_nd(metadata=out3, bd_id=0, mem=o3, sizes=[1,1,1,48])

                dma_wait(in0, out0)
                dma_wait(in1, out1)
                dma_wait(in2, out2)
                dma_wait(in3, out3)


    print(ctx.module)


if __name__ == "__main__":
    distribute_join_L2_4cols_simple()

