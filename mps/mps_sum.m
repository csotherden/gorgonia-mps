// mps_sum.m
// Minimal Objective-C helper that uses a custom Metal compute kernel to
// perform row-wise summation over a float32 matrix using the shared
// engine context (device + command queue).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "mps_sum.h"

// Re-declare the engine context ObjC class so we can downcast the
// opaque MPSEngineContext handle back to a usable object. The actual
// implementation lives in mps_engine_ctx.m.
@interface MPSEngineContextObj : NSObject
@property(nonatomic, readonly) id<MTLDevice> device;
@property(nonatomic, readonly) id<MTLCommandQueue> queue;
@property(nonatomic, readonly) id<MTLComputePipelineState> rowSumPSO;
@end

int mpsRowSumFloat32(MPSEngineContext ctx,
                     const float *x,
                     float *y,
                     int rows,
                     int cols) {
    @autoreleasepool {
        if (ctx == NULL) {
            return -1;
        }

        MPSEngineContextObj *obj = (__bridge MPSEngineContextObj *)ctx;
        id<MTLDevice> device = obj.device;
        id<MTLCommandQueue> queue = obj.queue;
        id<MTLComputePipelineState> pso = obj.rowSumPSO;

        if (device == nil || queue == nil || pso == nil) {
            return -1;
        }

        if (x == NULL || y == NULL) {
            return -2;
        }

        const NSUInteger uRows = (NSUInteger)rows;
        const NSUInteger uCols = (NSUInteger)cols;
        const NSUInteger bytesX = uRows * uCols * sizeof(float);
        const NSUInteger bytesY = uRows * sizeof(float);

        id<MTLBuffer> bufX =
            [device newBufferWithBytesNoCopy:(void *)x
                                      length:bytesX
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        id<MTLBuffer> bufY =
            [device newBufferWithBytesNoCopy:(void *)y
                                      length:bytesY
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        if (bufX == nil || bufY == nil) {
            return -6;
        }

        typedef struct {
            uint rows;
            uint cols;
        } RowSumShape;

        RowSumShape shape = { (uint)rows, (uint)cols };
        id<MTLBuffer> bufShape =
            [device newBufferWithBytes:&shape
                                length:sizeof(shape)
                               options:MTLResourceStorageModeShared];
        if (bufShape == nil) {
            return -7;
        }

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        if (cmdBuf == nil) {
            return -8;
        }

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        if (enc == nil) {
            return -9;
        }

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufX offset:0 atIndex:0];
        [enc setBuffer:bufY offset:0 atIndex:1];
        [enc setBuffer:bufShape offset:0 atIndex:2];

        // Launch one threadgroup per row, with multiple threads per row
        // cooperating via threadgroup memory.
        NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
        if (maxThreads == 0) {
            maxThreads = 1;
        }
        const NSUInteger maxPerRow = 256;
        // Don't launch more threads per row than we have columns.
        NSUInteger threadsPerThreadgroup = MIN(maxThreads, MIN(uCols, maxPerRow));

        MTLSize numThreadgroups = MTLSizeMake(uRows, 1, 1);
        MTLSize tgSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        [enc dispatchThreadgroups:numThreadgroups
            threadsPerThreadgroup:tgSize];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(y, [bufY contents], bytesY);

        return 0;
    }
}

