#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

// MARK: - Vertex Shader

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct FragmentIn {
    float4 position [[position]];
    float2 texCoord;
};

/// Full-screen quad vertex shader for canvas rendering
vertex FragmentIn canvasVertex(uint vertexID [[vertex_id]],
                                constant CanvasUniforms &uniforms [[buffer(BufferIndexUniforms)]]) {
    // Full-screen triangle vertices
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };

    float2 texCoords[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };

    FragmentIn out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];

    return out;
}

// MARK: - Fragment Shader

/// Transform UV coordinates using the inverse canvas transform
float2 transformUV(float2 uv, float2 canvasSize, float2 imageSize, matrix_float3x3 inverseTransform) {
    // Convert UV to screen coordinates
    float2 screenPos = uv * canvasSize;

    // Apply inverse transform to get image coordinates
    float3 transformed = inverseTransform * float3(screenPos, 1.0);
    float2 imagePos = transformed.xy;

    // Convert to UV in image space
    return imagePos / imageSize;
}

/// Check if pixel is on the edge of a mask region
/// Returns true if any of the 8 neighbors has a different class ID
bool isEdgePixel(texture2d<uint, access::read> maskTexture, uint2 coord, uint classID) {
    uint width = maskTexture.get_width();
    uint height = maskTexture.get_height();

    // Check 8 neighbors
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = int(coord.x) + dx;
            int ny = int(coord.y) + dy;

            // Boundary check
            if (nx < 0 || nx >= int(width) || ny < 0 || ny >= int(height)) {
                // Edge of texture counts as edge
                return true;
            }

            uint neighborClass = maskTexture.read(uint2(nx, ny)).r;
            if (neighborClass != classID) {
                return true;
            }
        }
    }
    return false;
}

/// Apply contrast and brightness to a color
float4 applyContrastBrightness(float4 color, float contrast, float brightness) {
    // Contrast: scale around 0.5 (mid-gray)
    float3 adjusted = (color.rgb - 0.5) * contrast + 0.5;
    // Brightness: simple offset
    adjusted = adjusted + brightness;
    // Clamp to valid range
    adjusted = clamp(adjusted, 0.0, 1.0);
    return float4(adjusted, color.a);
}

/// Canvas compositing fragment shader
/// Blends source image with mask overlay using class colors
/// Mask values: 0 = no mask, 1-8 = class ID (matches currentClassID)
/// Edge pixels are always opaque, fill pixels use maskFillAlpha
fragment float4 canvasFragment(FragmentIn in [[stage_in]],
                                texture2d<float, access::sample> imageTexture [[texture(TextureIndexImage)]],
                                texture2d<uint, access::read> maskTexture [[texture(TextureIndexMask)]],
                                constant CanvasUniforms &uniforms [[buffer(BufferIndexUniforms)]]) {

    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    // Transform UV to image space
    float2 imageUV = transformUV(in.texCoord, uniforms.canvasSize, uniforms.imageSize, uniforms.inverseTransform);

    // Check if UV is within image bounds
    if (imageUV.x < 0.0 || imageUV.x > 1.0 || imageUV.y < 0.0 || imageUV.y > 1.0) {
        // Background color (dark gray)
        return float4(0.2, 0.2, 0.2, 1.0);
    }

    // Sample source image with bilinear filtering
    float4 imageColor = imageTexture.sample(linearSampler, imageUV);

    // Apply contrast and brightness to image
    imageColor = applyContrastBrightness(imageColor, uniforms.imageContrast, uniforms.imageBrightness);

    // Calculate mask UV (scaled by maskScaleFactor)
    float2 maskUV = imageUV * float2(uniforms.maskSize);
    uint2 maskCoord = uint2(maskUV);

    // Clamp to mask bounds
    maskCoord = min(maskCoord, uint2(uniforms.maskSize) - 1);

    // Sample mask - value is class ID (0 = none, 1-8 = class)
    uint classID = maskTexture.read(maskCoord).r;

    // Start with adjusted image
    float4 result = imageColor;
    result.a = 1.0;

    // If there's a class at this pixel, blend its color
    // classColor.a == 0 means the class is temporarily hidden
    if (classID > 0 && classID <= MAX_CLASSES) {
        float4 classColor = uniforms.classColors[classID];

        if (classColor.a > 0) {
            // Check if this is an edge pixel
            bool isEdge = isEdgePixel(maskTexture, maskCoord, classID);

            // Edge pixels use maskEdgeAlpha, fill pixels use maskFillAlpha
            float alpha = isEdge ? uniforms.maskEdgeAlpha : uniforms.maskFillAlpha;

            float4 maskOverlay = float4(classColor.rgb, alpha);
            result = mix(result, maskOverlay, maskOverlay.a);
        }
    }

    return result;
}

// MARK: - Compute Shaders

/// Brush stamp compute shader
/// Applies a circular stamp to the mask texture
kernel void brushStamp(texture2d<uint, access::read_write> mask [[texture(TextureIndexOutput)]],
                       constant BrushParams &params [[buffer(BufferIndexBrushParams)]],
                       uint2 gid [[thread_position_in_grid]]) {

    // Check bounds
    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) {
        return;
    }

    // Calculate distance from stamp center
    float2 pos = float2(gid);
    float dist = distance(pos, params.center);

    // Apply stamp if within radius
    if (dist <= params.radius) {
        mask.write(uint4(params.paintValue, 0, 0, 0), gid);
    }
}

/// Clear mask compute shader
/// Fills the entire mask with a value
kernel void clearMask(texture2d<uint, access::write> mask [[texture(TextureIndexOutput)]],
                      constant uint &clearValue [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) {
        return;
    }

    mask.write(uint4(clearValue, 0, 0, 0), gid);
}
