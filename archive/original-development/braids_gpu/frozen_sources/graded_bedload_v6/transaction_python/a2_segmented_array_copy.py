"""Single-launch device snapshot/restore for reachable A2 CuPy arrays.

Validation-only transaction utility. It copies logical bytes, including
strided views, into one resident byte buffer. It does not alter numerical
fields, kernels or CUDA graph internals.
"""
from __future__ import annotations

import numpy as np


COPY_CUDA=r'''
extern "C" __global__ void a2_segmented_array_copy(
    const unsigned long long* pointers,
    const long long* offsets,
    const long long* byte_lengths,
    const int* dimensions,
    const int* item_sizes,
    const long long* shapes,
    const long long* strides,
    unsigned char* packed,
    const int segment_count,
    const int restore)
{
    const int segment=blockIdx.x;
    if (segment>=segment_count) return;
    unsigned char* array=reinterpret_cast<unsigned char*>(pointers[segment]);
    const long long nbytes=byte_lengths[segment];
    const long long packed_offset=offsets[segment];
    const int ndim=dimensions[segment];
    const int itemsize=item_sizes[segment];
    for (long long byte_index=threadIdx.x;byte_index<nbytes;byte_index+=blockDim.x) {
        long long array_offset=byte_index;
        if (ndim<0) {
            long long element=byte_index/itemsize;
            array_offset=byte_index%itemsize;
            for (int axis=-ndim-1;axis>=0;--axis) {
                const long long width=shapes[segment*8+axis];
                const long long coordinate=element%width;
                element/=width;
                array_offset+=coordinate*strides[segment*8+axis];
            }
        }
        if (restore) array[array_offset]=packed[packed_offset+byte_index];
        else packed[packed_offset+byte_index]=array[array_offset];
    }
}
'''


def signature(arrays):
    return tuple((name,int(array.data.ptr),array.shape,array.dtype.str,array.strides)
                 for name,array in arrays.items())


class SegmentedCopyPlan:
    def __init__(self,cp,arrays):
        self.cp=cp
        self.signature=signature(arrays)
        self.count=len(arrays)
        if not self.count:raise ValueError('Empty A2 GPU array inventory')
        if any(array.ndim>8 for array in arrays.values()):
            raise ValueError('A2 transaction supports at most eight dimensions')
        lengths=np.asarray([array.nbytes for array in arrays.values()],dtype=np.int64)
        offsets=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(lengths[:-1])))
        shape=np.ones((self.count,8),dtype=np.int64)
        strides=np.zeros((self.count,8),dtype=np.int64)
        dimensions=[]
        for i,array in enumerate(arrays.values()):
            shape[i,:array.ndim]=array.shape
            strides[i,:array.ndim]=array.strides
            dimensions.append(array.ndim if array.flags.c_contiguous else -array.ndim)
        self.pointers=cp.asarray(np.asarray([int(array.data.ptr) for array in arrays.values()],
                                          dtype=np.uint64))
        self.offsets=cp.asarray(offsets)
        self.lengths=cp.asarray(lengths)
        self.dimensions=cp.asarray(np.asarray(dimensions,dtype=np.int32))
        self.item_sizes=cp.asarray(np.asarray([array.dtype.itemsize for array in arrays.values()],
                                              dtype=np.int32))
        self.shapes=cp.asarray(shape)
        self.strides=cp.asarray(strides)
        self.total_bytes=int(lengths.sum())
        self.packed=cp.empty(self.total_bytes,dtype=cp.uint8)
        self.kernel=cp.RawKernel(COPY_CUDA,'a2_segmented_array_copy',
                                 options=('--std=c++11',))

    def _launch(self,restore):
        self.kernel((self.count,),(128,),
            (self.pointers,self.offsets,self.lengths,self.dimensions,self.item_sizes,
             self.shapes,self.strides,self.packed,np.int32(self.count),np.int32(restore)))

    def snapshot(self):self._launch(False)

    def restore(self):self._launch(True)
