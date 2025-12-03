// mps_engine_ctx.m
// Objective-C implementation of the engine-level Metal/MPS context.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "mps_engine_ctx.h"

@interface MPSEngineContextObj : NSObject
@property(nonatomic, readonly) id<MTLDevice> device;
@property(nonatomic, readonly) id<MTLCommandQueue> queue;
@end

@implementation MPSEngineContextObj

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
    }
    return self;
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


