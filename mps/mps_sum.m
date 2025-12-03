// mps_sum.m
// Minimal Objective-C helper that uses a custom Metal compute kernel to
// perform row-wise summation over a float32 matrix using the shared
// engine context (device + command queue).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "mps_engine_ctx.h"
#import "mps_sum.h"

// Re-declare the engine context ObjC class so we can downcast the
// opaque MPSEngineContext handle back to a usable object. The actual
// implementation lives in mps_engine_ctx.m.
@interface MPSEngineContextObj : NSObject
@property(nonatomic, readonly) id<MTLDevice> device;
@property(nonatomic, readonly) id<MTLCommandQueue> queue;
@end

static NSString * const kRowSumKernelSource =
@"#include <metal_stdlib>\n"
 "using namespace metal;\n"
 "\n"
 "kernel void row_sum(\n"
 "    const device float *X      [[buffer(0)]],\n"
 "    device float *Y            [[buffer(1)]],\n"
 "    constant uint2 &shape      [[buffer(2)]],\n"
 "    uint gid                   [[thread_position_in_grid]]) {\n"
 "  uint rows = shape.x;\n"
 "  uint cols = shape.y;\n"
 "  if (gid >= rows) { return; }\n"
 "  float acc = 0.0f;\n"
 "  uint base = gid * cols;\n"
 "  for (uint j = 0; j < cols; ++j) {\n"
 "    acc += X[base + j];\n"
 "  }\n"
 "  Y[gid] = acc;\n"
 "}\n";

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

        if (device == nil || queue == nil) {
            return -1;
        }

        if (x == NULL || y == NULL) {
            return -2;
        }

        const NSUInteger uRows = (NSUInteger)rows;
        const NSUInteger uCols = (NSUInteger)cols;
        const NSUInteger bytesX = uRows * uCols * sizeof(float);
        const NSUInteger bytesY = uRows * sizeof(float);

        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kRowSumKernelSource
                                                 options:nil
                                                   error:&err];
        if (lib == nil) {
            return -3;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"row_sum"];
        if (fn == nil) {
            return -4;
        }

        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) {
            return -5;
        }

        id<MTLBuffer> bufX =
            [device newBufferWithBytes:x
                                length:bytesX
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY =
            [device newBufferWithLength:bytesY
                                 options:MTLResourceStorageModeShared];
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

        // Launch one thread per row.
        NSUInteger threadsPerGrid = uRows;
        NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
        if (maxThreads == 0) {
            maxThreads = 1;
        }
        NSUInteger threadsPerThreadgroup = MIN(maxThreads, threadsPerGrid);

        MTLSize gridSize = MTLSizeMake(threadsPerGrid, 1, 1);
        MTLSize tgSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:gridSize
         threadsPerThreadgroup:tgSize];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(y, [bufY contents], bytesY);

        return 0;
    }
}

