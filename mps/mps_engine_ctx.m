// mps_engine_ctx.m
// Objective-C implementation of the engine-level Metal/MPS context.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "mps_engine_ctx.h"

@interface MPSEngineContextObj : NSObject
@property(nonatomic, readonly) id<MTLDevice> device;
@property(nonatomic, readonly) id<MTLCommandQueue> queue;
@property(nonatomic, readonly) id<MTLComputePipelineState> rowSumPSO;
@end

@implementation MPSEngineContextObj {
    id<MTLComputePipelineState> _rowSumPSO;
}

// Metal compute kernel source for row-wise sum. Each threadgroup
// processes one row, with multiple threads per row accumulating partial
// sums in threadgroup memory and then reducing to a single value.
static NSString * const kRowSumKernelSource =
@"#include <metal_stdlib>\n"
 "using namespace metal;\n"
 "\n"
 "kernel void row_sum(\n"
 "    const device float *X      [[buffer(0)]],\n"
 "    device float *Y            [[buffer(1)]],\n"
 "    constant uint2 &shape      [[buffer(2)]],\n"
 "    uint  tid                  [[thread_index_in_threadgroup]],\n"
 "    uint3 tgpig                [[threadgroup_position_in_grid]],\n"
 "    uint  tgSize               [[threads_per_threadgroup]]) {\n"
 "  uint rows = shape.x;\n"
 "  uint cols = shape.y;\n"
 "  uint row  = tgpig.x;\n"
 "  if (row >= rows) { return; }\n"
 "  threadgroup float partial[256];\n"
 "  float acc = 0.0f;\n"
 "  uint base = row * cols;\n"
 "  for (uint c = tid; c < cols; c += tgSize) {\n"
 "    acc += X[base + c];\n"
 "  }\n"
 "  partial[tid] = acc;\n"
 "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
 "  for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {\n"
 "    if (tid < stride && tid + stride < tgSize) {\n"
 "      partial[tid] += partial[tid + stride];\n"
 "    }\n"
 "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
 "  }\n"
 "  if (tid == 0) {\n"
 "    Y[row] = partial[0];\n"
 "  }\n"
 "}\n";

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            return nil;
        }
        _queue = [_device newCommandQueue];
        if (!_queue) {
            return nil;
        }

        // Compile and cache the row_sum pipeline once per engine context.
        NSError *err = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:kRowSumKernelSource
                                                   options:nil
                                                     error:&err];
        if (!lib) {
            return nil;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"row_sum"];
        if (!fn) {
            return nil;
        }
        _rowSumPSO = [_device newComputePipelineStateWithFunction:fn error:&err];
        if (!_rowSumPSO) {
            return nil;
        }
    }
    return self;
}

- (id<MTLComputePipelineState>)rowSumPSO {
    return _rowSumPSO;
}

@end

MPSEngineContext MPSEngineCreateContext(void) {
    @autoreleasepool {
        MPSEngineContextObj *ctx = [MPSEngineContextObj new];
        if (!ctx || !ctx.device || !ctx.queue) {
            return NULL;
        }
        return (__bridge_retained void *)ctx;
    }
}

void MPSEngineReleaseContext(MPSEngineContext ctx) {
    @autoreleasepool {
        if (!ctx) {
            return;
        }
        MPSEngineContextObj *obj = (__bridge_transfer MPSEngineContextObj *)ctx;
        obj = nil;
    }
}


