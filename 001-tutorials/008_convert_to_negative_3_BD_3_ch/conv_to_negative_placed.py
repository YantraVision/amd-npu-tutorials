
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils

def conv_to_negativeAIE2(dev, width, height):
    trace_size = 131072
    lineWidthInBytes = width
    tensorSize = 5760 * height

    @device(dev)
    def device_body():
        # define types
        tensor_ty = np.ndarray[(5760,), np.dtype[np.int8]]
        line_ty = np.ndarray[(1920,), np.dtype[np.uint8]]
        line_ty_3 = line_ty

        # AIE Core Function declarations
        conv_to_negativeLine = external_func(
            "conv_to_negativeLine", inputs=[line_ty_3, line_ty_3, np.int32]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)


        tiles_to_trace = [ComputeTile2, ShimTile]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        #Input Buffers
        # First BD for channel R
        in_R_cons_prod_lock = lock(ComputeTile2, lock_id=0, init=1)
        in_R_cons_cons_lock = lock(ComputeTile2, lock_id=1, init=0)
        in_R_cons_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=line_ty_3,
            name="in_R_cons_buff_0",
        )

        # second BD for channel G
        in_G_cons_prod_lock = lock(ComputeTile2, lock_id=2, init=1)
        in_G_cons_cons_lock = lock(ComputeTile2, lock_id=3, init=0)
        in_G_cons_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=line_ty_3,
            name="in_G_cons_buff_0",
        )

        # third BD for channel B
        in_B_cons_prod_lock = lock(ComputeTile2, lock_id=4, init=1)
        in_B_cons_cons_lock = lock(ComputeTile2, lock_id=5, init=0)
        in_B_cons_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=line_ty_3,
            name="in_B_cons_buff_0",
        )
        
        #Output Buffers
        #first BD for channel R
        out_R_prod_lock = lock(ComputeTile2, lock_id=6, init=1)
        out_R_cons_lock = lock(ComputeTile2, lock_id=7, init=0)
        out_R_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=line_ty_3,
            name="out_R_buff_0",
        )

        #second BD for channel G
        out_G_prod_lock = lock(ComputeTile2, lock_id=8, init=1)
        out_G_cons_lock = lock(ComputeTile2, lock_id=9, init=0)
        out_G_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=line_ty_3,
            name="out_G_buff_0",
        )

        #third BD for channel B
        out_B_prod_lock = lock(ComputeTile2, lock_id=10, init=1)
        out_B_cons_lock = lock(ComputeTile2, lock_id=11, init=0)
        out_B_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=line_ty_3,
            name="out_B_buff_0",
        )


        # AIE-array data movement with flow connection
        flow(ShimTile, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 0)
        flow(ComputeTile2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 0)

        # ComputeTile DMA configuration
        @mem(ComputeTile2)
        def m(block):
            
            # channel allocation in S2MM direction, channel index 0
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[4])
            
            # BD chains are assigned to a channel as well, where the last BD is
            # either another channel allocation or the end BD
            with block[1]:   
                # wait on lock acquire
                use_lock(in_R_cons_prod_lock, LockAction.AcquireGreaterEqual)
                # receive incoming data in in1_cons_buff_0 buffer
                dma_bd(in_R_cons_buff_0)
                # release lock
                use_lock(in_R_cons_cons_lock, LockAction.Release)
                next_bd(block[2])
            
            with block[2]:
                
                # wait on lock acquire
                use_lock(in_G_cons_prod_lock, LockAction.AcquireGreaterEqual)
                # receive incoming data in in1_cons_buff_0 buffer
                dma_bd(in_G_cons_buff_0)
                # release lock
                use_lock(in_G_cons_cons_lock, LockAction.Release)
                next_bd(block[3])
            
            with block[3]:
                # wait on lock acquire
                use_lock(in_B_cons_prod_lock, LockAction.AcquireGreaterEqual)
                # receive incoming data in in1_cons_buff_0 buffer
                dma_bd(in_B_cons_buff_0)
                # release lock
                use_lock(in_B_cons_cons_lock, LockAction.Release)
                # BD loops forever on itself
                next_bd(block[1])
            
            with block[4]:
                # channel allocation in MM2S direction, channel index 0
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[5], chain=block[8])
                # BD chains are assigned to a channel as well, where the last BD is
                # either another channel allocation or the end BD
            
            with block[5]:
                # wait on lock acquire
                use_lock(out_R_cons_lock, LockAction.AcquireGreaterEqual)
                # output data from out_buff_0 buffer
                dma_bd(out_R_buff_0)
                # release lock
                use_lock(out_R_prod_lock, LockAction.Release)
                next_bd(block[6])
            
            with block[6]:
                # wait on lock acquire
                use_lock(out_G_cons_lock, LockAction.AcquireGreaterEqual)
                # output data from out_buff_0 buffer
                dma_bd(out_G_buff_0)
                # release lock
                use_lock(out_G_prod_lock, LockAction.Release)
                next_bd(block[7])
            
            with block[7]:
                # wait on lock acquire
                use_lock(out_B_cons_lock, LockAction.AcquireGreaterEqual)
                # output data from out_buff_0 buffer
                dma_bd(out_B_buff_0)
                # release lock
                use_lock(out_B_prod_lock, LockAction.Release)
                # BD loops forever on itself
                next_bd(block[5])

            with block[8]:
                EndOp()


        # Compute tile 2
        @core(ComputeTile2, "conv_to_negative.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(height):
                    # Compute channel R
                    use_lock(in_R_cons_cons_lock, LockAction.AcquireGreaterEqual)
                    use_lock(out_R_prod_lock, LockAction.AcquireGreaterEqual)

                    conv_to_negativeLine(in_R_cons_buff_0, out_R_buff_0, lineWidthInBytes)

                    use_lock(in_R_cons_prod_lock, LockAction.Release)
                    use_lock(out_R_cons_lock, LockAction.Release)
                    
                    # Compute channel G
                    use_lock(in_G_cons_cons_lock, LockAction.AcquireGreaterEqual)
                    use_lock(out_G_prod_lock, LockAction.AcquireGreaterEqual)

                    conv_to_negativeLine(in_G_cons_buff_0, out_G_buff_0, lineWidthInBytes)
                    
                    use_lock(in_G_cons_prod_lock, LockAction.Release)
                    use_lock(out_G_cons_lock, LockAction.Release)

                    # Compute channel B
                    use_lock(in_B_cons_cons_lock, LockAction.AcquireGreaterEqual)
                    use_lock(out_B_prod_lock, LockAction.AcquireGreaterEqual)
                    
                    conv_to_negativeLine(in_B_cons_buff_0, out_B_buff_0, lineWidthInBytes)

                    use_lock(in_B_cons_prod_lock, LockAction.Release)
                    use_lock(out_B_cons_lock, LockAction.Release)

        shim_dma_allocation("of_in1", DMAChannelDir.MM2S, 0, 0)
        shim_dma_allocation("of_out", DMAChannelDir.S2MM, 0, 0)

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, notUsed, outTensor):
            if trace_size > 0:
                data_trace=trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            in_task = shim_dma_single_bd_task(
                "of_in1", inTensor, sizes=[1, 1, 1, tensorSize], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                "of_out",
                outTensor,
                sizes=[1, 1, 1, tensorSize],
                issue_token=True,
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)
            dma_free_task(in_task)
            trace_utils.gen_trace_done_aie2(ShimTile)


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1_1col
    elif device_name == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 512 if (len(sys.argv) != 4) else int(sys.argv[2])
    height = 9 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    conv_to_negativeAIE2(dev, width, height)
    print(ctx.module)
