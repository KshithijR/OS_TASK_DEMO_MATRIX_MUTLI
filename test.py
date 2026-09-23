"""
Raw OptiX Ray Tracer for RTX 5050 (no Blender / no external 3D software)
--------------------------------------------------------------------------
Uses NVIDIA's OptiX API directly through the official PyOptiX bindings.
This hits the RT cores directly for BVH traversal and ray/triangle
intersection -- there is no rasterizer involved at any point.

REQUIRED SETUP (one-time, local machine):
  1. Install the CUDA Toolkit matching your driver.
  2. Download & install the NVIDIA OptiX SDK (developer.nvidia.com/designworks/optix).
  3. cd into <OptiX_SDK>/SDK/pyoptix and run: pip install .
  4. pip install cupy-cuda12x numpy pillow   (match cupy version to your CUDA)

This renders a single hard-coded triangle to prove the pipeline works end
to end on real RT cores. Swap in your own geometry / BVH build once this
runs -- the scaffolding (context, module, pipeline, SBT, launch) is the
part that's fiddly to get right, not the scene content.
"""

import cupy as cp
import numpy as np
import optix
from PIL import Image

WIDTH, HEIGHT = 1024, 768

# ---------------------------------------------------------------
# 1. CUDA device-side program (compiled at runtime via NVRTC)
# ---------------------------------------------------------------
# This is the actual GPU code: ray generation, a miss shader (background
# color), and a closest-hit shader (triangle color). OptiX schedules
# these onto the RT cores / SM cores as rays are traced.
CUDA_SOURCE = r"""
#include <optix.h>
#include "params.h"

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float u = (float(idx.x) + 0.5f) / float(dim.x);
    float v = (float(idx.y) + 0.5f) / float(dim.y);

    float3 origin = make_float3(0.0f, 0.0f, 2.0f);
    float3 dir = make_float3((u - 0.5f) * 2.0f, (v - 0.5f) * 2.0f, -1.0f);

    unsigned int p0, p1, p2;
    optixTrace(
        params.handle, origin, dir,
        0.0f, 1e16f, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1, p2);

    float3 result = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    params.image[idx.y * dim.x + idx.x] = make_uchar4(
        (unsigned char)(result.x * 255), (unsigned char)(result.y * 255),
        (unsigned char)(result.z * 255), 255);
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(__float_as_uint(0.1f));
    optixSetPayload_1(__float_as_uint(0.1f));
    optixSetPayload_2(__float_as_uint(0.15f));
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(__float_as_uint(0.9f));
    optixSetPayload_1(__float_as_uint(0.3f));
    optixSetPayload_2(__float_as_uint(0.2f));
}
"""

# params.h describes the struct shared between host and device
PARAMS_HEADER = r"""
struct Params {
    uchar4* image;
    OptixTraversableHandle handle;
};
"""

# ---------------------------------------------------------------
# 2. OptiX context setup
# ---------------------------------------------------------------
def log_callback(level, tag, msg, data):
    print(f"[OptiX][{tag}] {msg}")

cp.cuda.Device(0).use()
optix.init()

ctx_options = optix.DeviceContextOptions(
    logCallbackFunction=log_callback,
    logCallbackLevel=4
)
cu_ctx = 0  # use current CUDA context
optix_ctx = optix.deviceContextCreate(cu_ctx, ctx_options)

# ---------------------------------------------------------------
# 3. Build geometry (one triangle) + BVH acceleration structure
#    -- THIS is the step the RT cores accelerate.
# ---------------------------------------------------------------
vertices = cp.array([
    [-0.5, -0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [ 0.0,  0.5, 0.0],
], dtype=cp.float32)

build_input = optix.BuildInputTriangleArray()
build_input.vertexBuffers = [vertices.data.ptr]
build_input.numVertices = vertices.shape[0]
build_input.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
build_input.flags = [optix.GEOMETRY_FLAG_NONE]
build_input.numSbtRecords = 1

accel_options = optix.AccelBuildOptions(
    buildFlags=optix.BUILD_FLAG_ALLOW_COMPACTION,
    operation=optix.BUILD_OPERATION_BUILD
)

gas_buffer_sizes = optix_ctx.accelComputeMemoryUsage([accel_options], [build_input])
d_temp = cp.cuda.alloc(gas_buffer_sizes.tempSizeInBytes)
d_output = cp.cuda.alloc(gas_buffer_sizes.outputSizeInBytes)

gas_handle, _ = optix_ctx.accelBuild(
    0, [accel_options], [build_input],
    d_temp.ptr, gas_buffer_sizes.tempSizeInBytes,
    d_output.ptr, gas_buffer_sizes.outputSizeInBytes,
    []
)

# ---------------------------------------------------------------
# 4. Compile module + build pipeline + shader binding table
# ---------------------------------------------------------------
module_options = optix.ModuleCompileOptions(
    maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
    optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
    debugLevel=optix.COMPILE_DEBUG_LEVEL_NONE
)
pipeline_options = optix.PipelineCompileOptions(
    usesMotionBlur=False,
    traversableGraphFlags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
    numPayloadValues=3,
    numAttributeValues=2,
    exceptionFlags=optix.EXCEPTION_FLAG_NONE,
    pipelineLaunchParamsVariableName="params"
)

ptx = optix.compileCudaSource(CUDA_SOURCE, header=PARAMS_HEADER)  # wraps NVRTC compile
module = optix_ctx.moduleCreate(module_options, pipeline_options, ptx)

raygen_desc = optix.ProgramGroupDesc(raygenModule=module, raygenEntryFunctionName="__raygen__rg")
miss_desc = optix.ProgramGroupDesc(missModule=module, missEntryFunctionName="__miss__ms")
hit_desc = optix.ProgramGroupDesc(hitgroupModuleCH=module, hitgroupEntryFunctionNameCH="__closesthit__ch")

raygen_pg = optix_ctx.programGroupCreate([raygen_desc])[0]
miss_pg = optix_ctx.programGroupCreate([miss_desc])[0]
hit_pg = optix_ctx.programGroupCreate([hit_desc])[0]

pipeline_link_options = optix.PipelineLinkOptions(maxTraceDepth=1)
pipeline = optix_ctx.pipelineCreate(
    pipeline_options, pipeline_link_options,
    [raygen_pg, miss_pg, hit_pg]
)

sbt = optix.ShaderBindingTable(
    raygenRecord=optix.sbtRecordPack(raygen_pg),
    missRecordBase=optix.sbtRecordPack(miss_pg),
    missRecordStrideInBytes=optix.sbtRecordSize(),
    missRecordCount=1,
    hitgroupRecordBase=optix.sbtRecordPack(hit_pg),
    hitgroupRecordStrideInBytes=optix.sbtRecordSize(),
    hitgroupRecordCount=1
)

# ---------------------------------------------------------------
# 5. Launch rays -- this is where RT cores do BVH traversal +
#    ray/triangle intersection for every pixel in parallel.
# ---------------------------------------------------------------
d_image = cp.zeros((HEIGHT * WIDTH, 4), dtype=cp.uint8)

params = np.zeros(1, dtype=[('image', np.uint64), ('handle', np.uint64)])
params['image'] = d_image.data.ptr
params['handle'] = gas_handle
d_params = cp.asarray(params.view(np.uint8))

optix_ctx.launch(
    pipeline, 0,
    d_params.data.ptr, d_params.nbytes,
    sbt, WIDTH, HEIGHT, 1
)
cp.cuda.Stream.null.synchronize()

# ---------------------------------------------------------------
# 6. Save result
# ---------------------------------------------------------------
image_host = cp.asnumpy(d_image).reshape(HEIGHT, WIDTH, 4)
Image.fromarray(image_host, 'RGBA').save("optix_render.png")
print("Saved optix_render.png")