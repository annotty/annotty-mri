import Foundation
import Metal
import MetalKit
import simd
import Combine

/// Main Metal rendering coordinator
/// Manages render pipelines, compute pipelines, and GPU operations
class MetalRenderer: NSObject, ObservableObject {
    // MARK: - Metal Objects

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let textureManager: TextureManager

    private var renderPipelineState: MTLRenderPipelineState?
    private var brushStampPipelineState: MTLComputePipelineState?
    private var clearMaskPipelineState: MTLComputePipelineState?

    // MARK: - Uniforms

    /// Image contrast (0.0 - 2.0, 1.0 = normal)
    @Published var imageContrast: Float = 1.0
    /// Image brightness (-1.0 to 1.0, 0.0 = normal)
    @Published var imageBrightness: Float = 0.0
    /// Mask fill opacity (0.0 - 1.0, affects interior fill)
    @Published var maskFillAlpha: Float = 0.5
    /// Mask edge opacity (0.0 - 1.0, affects edge/outline)
    @Published var maskEdgeAlpha: Float = 1.0

    /// Class colors for rendering (index 0 unused, 1-8 = class colors)
    /// Default: red, orange, yellow, green, cyan, blue, purple, pink
    var classColors: [simd_float4] = [
        simd_float4(0, 0, 0, 0),        // 0: unused
        simd_float4(1, 0, 0, 1),        // 1: red
        simd_float4(1, 0.5, 0, 1),      // 2: orange
        simd_float4(1, 1, 0, 1),        // 3: yellow
        simd_float4(0, 1, 0, 1),        // 4: green
        simd_float4(0, 1, 1, 1),        // 5: cyan
        simd_float4(0, 0, 1, 1),        // 6: blue
        simd_float4(0.5, 0, 1, 1),      // 7: purple
        simd_float4(1, 0.4, 0.7, 1)     // 8: pink
    ]

    // MARK: - State

    /// Current class ID for drawing (1-8, 0 = eraser)
    @Published var currentClassID: Int = 1
    /// Class IDs whose annotations are temporarily hidden during rendering
    var hiddenClassIDs: Set<Int> = []
    var canvasTransform = CanvasTransform()
    private(set) var viewportSize: CGSize = .zero
    private(set) var contentScaleFactor: CGFloat = 1.0

    // MARK: - State Flags

    private(set) var isPipelineReady: Bool = false

    // MARK: - Initialization

    override init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }

        self.device = device
        self.commandQueue = commandQueue
        self.textureManager = TextureManager(device: device)

        super.init()

        // Debug: Print struct sizes for Metal alignment verification
        print("📐 CanvasUniforms size: \(MemoryLayout<CanvasUniforms>.size) bytes")
        print("📐 CanvasUniforms stride: \(MemoryLayout<CanvasUniforms>.stride) bytes")
        print("📐 CanvasUniforms alignment: \(MemoryLayout<CanvasUniforms>.alignment) bytes")
        print("📐 simd_float3x3 size: \(MemoryLayout<simd_float3x3>.size) bytes")
        print("📐 simd_float3x3 stride: \(MemoryLayout<simd_float3x3>.stride) bytes")

        do {
            try setupPipelines()
            isPipelineReady = true
        } catch {
            print("⚠️ Metal pipeline setup failed: \(error)")
            print("⚠️ Make sure Shaders.metal is added to Xcode's Compile Sources")
            isPipelineReady = false
        }
    }

    // MARK: - Pipeline Setup

    private func setupPipelines() throws {
        guard let library = device.makeDefaultLibrary() else {
            print("❌ device.makeDefaultLibrary() returned nil")
            print("❌ Shaders.metal must be added to the Xcode project's Compile Sources")
            throw RendererError.libraryCreationFailed
        }

        // Render pipeline for canvas
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.vertexFunction = library.makeFunction(name: "canvasVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "canvasFragment")
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        // Enable blending
        renderDescriptor.colorAttachments[0].isBlendingEnabled = true
        renderDescriptor.colorAttachments[0].rgbBlendOperation = .add
        renderDescriptor.colorAttachments[0].alphaBlendOperation = .add
        renderDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        renderDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        renderDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        renderPipelineState = try device.makeRenderPipelineState(descriptor: renderDescriptor)

        // Compute pipeline for brush stamp
        guard let brushStampFunction = library.makeFunction(name: "brushStamp") else {
            throw RendererError.functionNotFound("brushStamp")
        }
        brushStampPipelineState = try device.makeComputePipelineState(function: brushStampFunction)

        // Compute pipeline for clear mask
        guard let clearMaskFunction = library.makeFunction(name: "clearMask") else {
            throw RendererError.functionNotFound("clearMask")
        }
        clearMaskPipelineState = try device.makeComputePipelineState(function: clearMaskFunction)
    }

    // MARK: - Rendering

    /// Main render function called by MTKView
    func render(to drawable: CAMetalDrawable, with renderPassDescriptor: MTLRenderPassDescriptor) {
        guard let renderPipelineState = renderPipelineState,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        // Skip full rendering if no image is loaded - just clear to background
        guard textureManager.imageTexture != nil else {
            renderEncoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
            return
        }

        renderEncoder.setRenderPipelineState(renderPipelineState)

        // Set uniforms - use stride for proper Metal alignment
        var uniforms = createUniforms()
        let uniformsSize = MemoryLayout<CanvasUniforms>.stride
        renderEncoder.setVertexBytes(&uniforms, length: uniformsSize, index: 0)
        renderEncoder.setFragmentBytes(&uniforms, length: uniformsSize, index: 0)

        // Set textures
        if let imageTexture = textureManager.imageTexture {
            renderEncoder.setFragmentTexture(imageTexture, index: 0)
        }

        if let maskTexture = textureManager.maskTexture {
            renderEncoder.setFragmentTexture(maskTexture, index: 1)
        }

        // Draw full-screen triangle
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    /// Create uniforms structure for shaders
    private func createUniforms() -> CanvasUniforms {
        // Build class colors tuple (9 elements: 0 unused, 1-8 = class colors)
        // Hidden classes get alpha = 0 so the shader skips rendering them
        var visibleColors = classColors
        for classID in hiddenClassIDs {
            if classID >= 0 && classID < visibleColors.count {
                visibleColors[classID].w = 0  // Set alpha to 0
            }
        }
        let colors = (
            visibleColors[0], visibleColors[1], visibleColors[2], visibleColors[3],
            visibleColors[4], visibleColors[5], visibleColors[6], visibleColors[7],
            visibleColors[8]
        )

        return CanvasUniforms(
            transform: canvasTransform.toSimdMatrix(),
            inverseTransform: canvasTransform.toInverseSimdMatrix(),
            imageContrast: imageContrast,
            imageBrightness: imageBrightness,
            maskFillAlpha: maskFillAlpha,
            maskEdgeAlpha: maskEdgeAlpha,
            canvasSize: simd_float2(Float(viewportSize.width), Float(viewportSize.height)),
            imageSize: simd_float2(Float(textureManager.imageSize.width), Float(textureManager.imageSize.height)),
            maskSize: simd_float2(Float(textureManager.maskSize.width), Float(textureManager.maskSize.height)),
            maskScaleFactor: textureManager.maskScaleFactor,
            _padding2: 0,
            classColors: colors
        )
    }

    // MARK: - Brush Operations

    /// Apply a brush stamp at the given position (point is in UIKit points)
    /// When painting, writes currentClassID to mask; when erasing, writes 0
    func applyStamp(at point: CGPoint, radius: Float, isPainting: Bool) {
        guard let computePipeline = brushStampPipelineState,
              let maskTexture = try? textureManager.getMaskTexture(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        // Convert touch point (in UIKit points) to screen coordinates (in pixels)
        let screenPoint = convertTouchToScreen(point)
        // Convert screen point to mask coordinates
        let maskPoint = canvasTransform.screenToMask(screenPoint)

        // Brush radius in mask coordinates
        // UI radius is in original image pixels, multiply by maskScaleFactor to get mask pixels
        // Also multiply by contentScaleFactor to match the pixel-based coordinate system
        // (touch points are converted to pixels, so radius must be too)
        let adjustedRadius = radius * canvasTransform.maskScaleFactor * Float(contentScaleFactor)

        // Paint value: currentClassID when painting, 0 when erasing
        let paintValue: UInt8 = isPainting ? UInt8(currentClassID) : 0

        var params = BrushParams(
            center: simd_float2(Float(maskPoint.x), Float(maskPoint.y)),
            radius: adjustedRadius,
            paintValue: paintValue,
            _padding: (0, 0, 0)
        )

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(maskTexture, index: 0)
        computeEncoder.setBytes(&params, length: MemoryLayout<BrushParams>.size, index: 0)

        // Calculate thread groups
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (maskTexture.width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (maskTexture.height + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        // Don't wait - let Metal handle synchronization implicitly
        // The render pass will wait for compute to finish on shared texture
        commandBuffer.commit()
    }

    /// Apply multiple stamps in a single command buffer (optimized for strokes)
    /// When painting, writes currentClassID to mask; when erasing, writes 0
    func applyStamps(at points: [CGPoint], radius: Float, isPainting: Bool) {
        guard points.count > 0,
              let computePipeline = brushStampPipelineState,
              let maskTexture = try? textureManager.getMaskTexture(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(maskTexture, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (maskTexture.width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (maskTexture.height + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )

        // Brush radius in mask coordinates (UI "1" = 1 original image pixel)
        // Multiply by contentScaleFactor to match pixel-based coordinate system
        let adjustedRadius = radius * canvasTransform.maskScaleFactor * Float(contentScaleFactor)

        // Paint value: currentClassID when painting, 0 when erasing
        let paintValue: UInt8 = isPainting ? UInt8(currentClassID) : 0

        // Encode all stamps in a single command buffer
        for point in points {
            let screenPoint = convertTouchToScreen(point)
            let maskPoint = canvasTransform.screenToMask(screenPoint)

            var params = BrushParams(
                center: simd_float2(Float(maskPoint.x), Float(maskPoint.y)),
                radius: adjustedRadius,
                paintValue: paintValue,
                _padding: (0, 0, 0)
            )

            computeEncoder.setBytes(&params, length: MemoryLayout<BrushParams>.size, index: 0)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
    }

    /// Apply multiple stamps along a stroke path (legacy, calls individual stamps)
    func applyStroke(points: [CGPoint], radius: Float, isPainting: Bool) {
        applyStamps(at: points, radius: radius, isPainting: isPainting)
    }

    /// Clear the entire mask (all classes)
    func clearMask() {
        guard let computePipeline = clearMaskPipelineState,
              let maskTexture = textureManager.maskTexture,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        var clearValue: UInt32 = 0

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(maskTexture, index: 0)
        computeEncoder.setBytes(&clearValue, length: MemoryLayout<UInt32>.size, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (maskTexture.width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (maskTexture.height + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    // MARK: - Image Loading

    /// Load image from URL
    func loadImage(from url: URL) throws {
        try textureManager.loadImage(from: url)
        canvasTransform.maskScaleFactor = textureManager.maskScaleFactor
    }

    /// Load image from CGImage
    func loadImage(_ cgImage: CGImage) throws {
        try textureManager.loadImage(cgImage)
        canvasTransform.maskScaleFactor = textureManager.maskScaleFactor
    }

    /// Apply preloaded image data (fast path — CGImage already decoded off main thread)
    func applyPreloadedImage(_ preloaded: TextureManager.PreloadedImageData) throws {
        try textureManager.applyPreloadedImage(preloaded)
        canvasTransform.maskScaleFactor = textureManager.maskScaleFactor
    }

    // MARK: - Viewport

    /// Update viewport size and scale factor
    func updateViewportSize(_ size: CGSize, scaleFactor: CGFloat = 1.0) {
        viewportSize = size
        contentScaleFactor = scaleFactor
    }

    /// Convert touch point (in points) to screen coordinates (in pixels for shader)
    func convertTouchToScreen(_ point: CGPoint) -> CGPoint {
        return CGPoint(
            x: point.x * contentScaleFactor,
            y: point.y * contentScaleFactor
        )
    }

    // MARK: - Class Management

    /// Set current class for drawing (1-8)
    func setCurrentClass(_ classID: Int) {
        guard classID >= 1 && classID <= TextureManager.maxClasses else { return }
        currentClassID = classID
    }
}

// MARK: - CanvasUniforms Bridge

/// Bridge structure matching ShaderTypes.h
/// Must match memory layout exactly with Metal shader
struct CanvasUniforms {
    var transform: simd_float3x3
    var inverseTransform: simd_float3x3
    /// Image contrast (0.0 - 2.0, 1.0 = normal)
    var imageContrast: Float
    /// Image brightness (-1.0 to 1.0, 0.0 = normal)
    var imageBrightness: Float
    /// Mask fill opacity (0.0 - 1.0, affects interior fill)
    var maskFillAlpha: Float
    /// Mask edge opacity (0.0 - 1.0, affects edge/outline)
    var maskEdgeAlpha: Float
    var canvasSize: simd_float2
    var imageSize: simd_float2
    var maskSize: simd_float2
    var maskScaleFactor: Float
    var _padding2: Float
    /// Class colors (index 0 unused, 1-8 = class colors)
    var classColors: (simd_float4, simd_float4, simd_float4, simd_float4,
                      simd_float4, simd_float4, simd_float4, simd_float4,
                      simd_float4)  // 9 elements to match MAX_CLASSES + 1
}

/// Bridge structure matching ShaderTypes.h
struct BrushParams {
    var center: simd_float2
    var radius: Float
    var paintValue: UInt8
    var _padding: (UInt8, UInt8, UInt8)
}

// MARK: - Errors

enum RendererError: Error, LocalizedError {
    case libraryCreationFailed
    case functionNotFound(String)
    case pipelineCreationFailed

    var errorDescription: String? {
        switch self {
        case .libraryCreationFailed:
            return "Failed to create Metal library"
        case .functionNotFound(let name):
            return "Metal function not found: \(name)"
        case .pipelineCreationFailed:
            return "Failed to create pipeline state"
        }
    }
}

