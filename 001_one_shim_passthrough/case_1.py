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

            # ---------------------------------------------------------
            # CORE — PROCESS
            # ---------------------------------------------------------

            @core(C0_0)
            def core0_0():
                for _ in range_(3):
                    x = in0_0.acquire(ObjectFifoPort.Consume, 1)
                    y = out0_0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        y[i] = x[i]
                    in0_0.release(ObjectFifoPort.Consume, 1)
                    out0_0.release(ObjectFifoPort.Produce, 1)


        
            # ---------------------------------------------------------
            # RUNTIME SEQUENCE
            # ---------------------------------------------------------

            data48 = np.ndarray[(48,), np.dtype[np.int32]]

            @runtime_sequence(data48,data48,data48)
            def sequence(i0,i1,o0):

                # Column 0
                npu_dma_memcpy_nd(metadata=in0, bd_id=1, mem=i0, sizes=[1,1,1,48], issue_token=True)
                npu_dma_memcpy_nd(metadata=out0, bd_id=0, mem=o0, sizes=[1,1,1,48])
                dma_wait(in0, out0)


    print(ctx.module)


if __name__ == "__main__":
    distribute_join_L2_4cols_simple()

