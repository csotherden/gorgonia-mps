// mps_matmul.m
// Minimal Objective-C helper that uses Metal Performance Shaders to
// perform a single-precision matrix multiplication using an engine-level
// context defined in mps_engine_ctx.{h,m}.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "mps_matmul.h"

// Re-declare the engine context ObjC class so we can downcast the
// opaque MPSEngineContext handle back to a usable object. The actual
// implementation lives in mps_engine_ctx.m.
@interface MPSEngineContextObj : NSObject
@property(nonatomic, readonly) id<MTLDevice> device;
@property(nonatomic, readonly) id<MTLCommandQueue> queue;
@end

int mpsMatMulFloat32(MPSEngineContext ctx,
                     const float *a,
                     const float *b,
                     float *c,
                     int m,
                     int n,
                     int k) {
    @autoreleasepool {
        if (ctx == NULL) {
            return -1;
        }
        MPSEngineContextObj *obj = (__bridge MPSEngineContextObj *)ctx;
        id<MTLDevice> g_mpsDevice = obj.device;
        id<MTLCommandQueue> g_mpsQueue = obj.queue;

        if (g_mpsDevice == nil || g_mpsQueue == nil) {
            // No usable Metal device; caller should fall back to CPU.
            return -1;
        }

        if (a == NULL || b == NULL || c == NULL) {
            return -2;
        }

        const NSUInteger rowsA = (NSUInteger)m;
        const NSUInteger colsA = (NSUInteger)k;
        const NSUInteger rowsB = (NSUInteger)k;
        const NSUInteger colsB = (NSUInteger)n;
        const NSUInteger rowsC = (NSUInteger)m;
        const NSUInteger colsC = (NSUInteger)n;

        const NSUInteger bytesA = rowsA * colsA * sizeof(float);
        const NSUInteger bytesB = rowsB * colsB * sizeof(float);
        const NSUInteger bytesC = rowsC * colsC * sizeof(float);

        id<MTLBuffer> bufA =
            [g_mpsDevice newBufferWithBytes:a
                                     length:bytesA
                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB =
            [g_mpsDevice newBufferWithBytes:b
                                     length:bytesB
                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC =
            [g_mpsDevice newBufferWithLength:bytesC
                                     options:MTLResourceStorageModeShared];

        if (bufA == nil || bufB == nil || bufC == nil) {
            return -3;
        }

        // Row-major layout: rowBytes is (number of columns * sizeof(float)).
        MPSMatrixDescriptor *descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:rowsA
                                                  columns:colsA
                                                 rowBytes:colsA * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:rowsB
                                                  columns:colsB
                                                 rowBytes:colsB * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:rowsC
                                                  columns:colsC
                                                 rowBytes:colsC * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        if (descA == nil || descB == nil || descC == nil) {
            return -4;
        }

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        if (matA == nil || matB == nil || matC == nil) {
            return -5;
        }

        // alpha=1, beta=0 gives C = A*B without blending with existing C.
        MPSMatrixMultiplication *mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:g_mpsDevice
                                             transposeLeft:NO
                                            transposeRight:NO
                                               resultRows:rowsC
                                             resultColumns:colsC
                                          interiorColumns:colsA
                                                    alpha:1.0f
                                                     beta:0.0f];

        if (mm == nil) {
            return -6;
        }

        id<MTLCommandBuffer> cmdBuf = [g_mpsQueue commandBuffer];
        if (cmdBuf == nil) {
            return -7;
        }

        [mm encodeToCommandBuffer:cmdBuf
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Copy results back into the caller-provided CPU buffer.
        memcpy(c, [bufC contents], bytesC);

        return 0;
    }
}
