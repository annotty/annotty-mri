#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

/// Maximum number of annotation classes (1-8)
#define MAX_CLASSES 8

/// Uniforms passed to canvas shaders
struct CanvasUniforms {
    /// Canvas transform matrix (pan/zoom/rotate)
    matrix_float3x3 transform;
    /// Inverse of transform matrix
    matrix_float3x3 inverseTransform;
    /// Image contrast (0.0 - 2.0, 1.0 = normal)
    float imageContrast;
    /// Image brightness (-1.0 to 1.0, 0.0 = normal)
    float imageBrightness;
    /// Mask fill opacity (0.0 - 1.0, affects interior fill)
    float maskFillAlpha;
    /// Mask edge opacity (0.0 - 1.0, affects edge/outline)
    float maskEdgeAlpha;
    /// Viewport size in pixels
    simd_float2 canvasSize;
    /// Source image size in pixels
    simd_float2 imageSize;
    /// Internal mask size in pixels (2x image with 4096 max clamp)
    simd_float2 maskSize;
    /// Scale factor from image to mask coordinates
    float maskScaleFactor;
    /// Padding for alignment
    float _padding2;
    /// Class colors (index 0 unused, 1-8 = class colors)
    simd_float4 classColors[MAX_CLASSES + 1];
};

/// Parameters for brush stamp compute shader
struct BrushParams {
    /// Stamp center in mask coordinates
    simd_float2 center;
    /// Brush radius in mask pixels
    float radius;
    /// Paint value: 1 for paint, 0 for erase
    uint8_t paintValue;
    /// Padding for alignment
    uint8_t _padding[3];
};

/// Vertex output for canvas rendering
struct VertexOut {
    simd_float4 position;
    simd_float2 texCoord;
};

/// Buffer indices for shader bindings
enum BufferIndex {
    BufferIndexUniforms = 0,
    BufferIndexVertices = 1,
    BufferIndexBrushParams = 0
};

/// Texture indices for shader bindings
enum TextureIndex {
    TextureIndexImage = 0,
    TextureIndexMask = 1,
    TextureIndexOutput = 0
};

#endif /* ShaderTypes_h */
