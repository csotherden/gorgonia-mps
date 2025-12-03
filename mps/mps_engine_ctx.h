// mps_engine_ctx.h
// Engine-level Metal/Metal Performance Shaders context used by MPSEng.
//
// This defines an opaque MPSEngineContext handle that owns a Metal
// device and command queue. Individual GPU-backed ops (matmul, sum,
// etc.) take this context as an argument rather than creating their own
// global state.

#pragma once

typedef void *MPSEngineContext;

#ifdef __cplusplus
extern "C" {
#endif

// MPSEngineCreateContext creates a new engine context. Returns NULL on
// failure.
MPSEngineContext MPSEngineCreateContext(void);

// MPSEngineReleaseContext releases a previously created context.
void MPSEngineReleaseContext(MPSEngineContext ctx);

#ifdef __cplusplus
}
#endif


