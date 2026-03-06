import Foundation
import Metal
import MetalKit
import CoreGraphics
import UIKit

/// Manages Metal textures for images and masks
/// Uses single mask texture with class IDs encoded as pixel values:
/// - 0 = no mask (background)
/// - 1-8 = class ID (matches currentClassID directly)
class TextureManager {
    private let device: MTLDevice
    private let textureLoader: MTKTextureLoader

    /// Currently loaded source image texture
    private(set) var imageTexture: MTLTexture?

    /// Single mask texture with class IDs (0=none, 1-8=classID)
    private(set) var maskTexture: MTLTexture?

    /// Source image size
    private(set) var imageSize: CGSize = .zero

    /// Internal mask size (2x with 4096 clamp)
    private(set) var maskSize: CGSize = .zero

    /// Scale factor from image to mask
    private(set) var maskScaleFactor: Float = 2.0

    /// Maximum mask dimension
    nonisolated static let maxMaskDimension = 4096

    /// Maximum number of classes (1-8)
    static let maxClasses = 8

    init(device: MTLDevice) {
        self.device = device
        self.textureLoader = MTKTextureLoader(device: device)
    }

    // MARK: - Preloaded Image Data (for background loading)

    /// Holds decoded image data prepared on a background thread.
    /// CGImage is immutable and thread-safe, so it can be created off-main
    /// and consumed on the main thread for GPU texture creation.
    struct PreloadedImageData {
        let cgImage: CGImage
        let imageWidth: Int
        let imageHeight: Int
        let maskWidth: Int
        let maskHeight: Int
        let maskScaleFactor: Float
    }

    /// Decode image from URL on any thread (no Metal dependency).
    /// Returns PreloadedImageData ready for applyPreloadedImage() on main thread.
    nonisolated static func prepareImageData(from url: URL) throws -> PreloadedImageData {
        guard let data = try? Data(contentsOf: url),
              let uiImage = UIImage(data: data),
              let cgImage = uiImage.cgImage else {
            throw TextureError.invalidImage
        }

        let width = cgImage.width
        let height = cgImage.height

        let maxEdge = max(width, height)
        let scaleFactor = min(2.0, Float(maxMaskDimension) / Float(maxEdge))
        let mw = Int(Float(width) * scaleFactor)
        let mh = Int(Float(height) * scaleFactor)

        return PreloadedImageData(
            cgImage: cgImage,
            imageWidth: width,
            imageHeight: height,
            maskWidth: mw,
            maskHeight: mh,
            maskScaleFactor: scaleFactor
        )
    }

    /// Apply preloaded image data to GPU textures (must be called on main thread).
    /// Creates image texture from CGImage and initializes a fresh mask texture.
    func applyPreloadedImage(_ preloaded: PreloadedImageData) throws {
        imageSize = CGSize(width: preloaded.imageWidth, height: preloaded.imageHeight)
        maskScaleFactor = preloaded.maskScaleFactor
        maskSize = CGSize(width: preloaded.maskWidth, height: preloaded.maskHeight)

        let options: [MTKTextureLoader.Option: Any] = [
            .textureUsage: MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode: MTLStorageMode.shared.rawValue,
            .SRGB: false
        ]

        let texture = try textureLoader.newTexture(cgImage: preloaded.cgImage, options: options)
        imageTexture = texture

        maskTexture = try createMaskTexture()
    }

    // MARK: - Image Loading

    /// Load source image from URL using MTKTextureLoader (GPU-accelerated)
    func loadImage(from url: URL) throws {
        // Use MTKTextureLoader for fast GPU-accelerated loading
        let options: [MTKTextureLoader.Option: Any] = [
            .textureUsage: MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode: MTLStorageMode.shared.rawValue,
            .SRGB: false
        ]

        let texture = try textureLoader.newTexture(URL: url, options: options)
        imageTexture = texture

        let width = texture.width
        let height = texture.height
        imageSize = CGSize(width: width, height: height)

        // Calculate mask dimensions with 4096 max clamp
        let maxEdge = max(width, height)
        maskScaleFactor = min(2.0, Float(Self.maxMaskDimension) / Float(maxEdge))

        let maskWidth = Int(Float(width) * maskScaleFactor)
        let maskHeight = Int(Float(height) * maskScaleFactor)
        maskSize = CGSize(width: maskWidth, height: maskHeight)

        // Create new mask texture for this image
        maskTexture = try createMaskTexture()
    }

    /// Load source image from CGImage (fallback for in-memory images)
    func loadImage(_ cgImage: CGImage) throws {
        let width = cgImage.width
        let height = cgImage.height
        imageSize = CGSize(width: width, height: height)

        // Calculate mask dimensions with 4096 max clamp
        let maxEdge = max(width, height)
        maskScaleFactor = min(2.0, Float(Self.maxMaskDimension) / Float(maxEdge))

        let maskWidth = Int(Float(width) * maskScaleFactor)
        let maskHeight = Int(Float(height) * maskScaleFactor)
        maskSize = CGSize(width: maskWidth, height: maskHeight)

        // Use MTKTextureLoader for CGImage too
        let options: [MTKTextureLoader.Option: Any] = [
            .textureUsage: MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode: MTLStorageMode.shared.rawValue,
            .SRGB: false
        ]

        let texture = try textureLoader.newTexture(cgImage: cgImage, options: options)
        imageTexture = texture

        // Create new mask texture for this image
        maskTexture = try createMaskTexture()
    }

    // MARK: - Mask Texture

    /// Get the mask texture (creates if needed)
    func getMaskTexture() throws -> MTLTexture {
        if let existing = maskTexture {
            return existing
        }

        // Ensure image is loaded before creating mask
        guard imageTexture != nil else {
            throw TextureError.noImageLoaded
        }

        let texture = try createMaskTexture()
        maskTexture = texture
        return texture
    }

    /// Create a new empty mask texture
    private func createMaskTexture() throws -> MTLTexture {
        // Guard against zero-size textures (no image loaded yet)
        guard maskSize.width > 0 && maskSize.height > 0 else {
            throw TextureError.noImageLoaded
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Uint,
            width: Int(maskSize.width),
            height: Int(maskSize.height),
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw TextureError.textureCreationFailed
        }

        // Initialize to zero
        let byteCount = Int(maskSize.width) * Int(maskSize.height)
        var zeros = [UInt8](repeating: 0, count: byteCount)
        texture.replace(
            region: MTLRegion(
                origin: MTLOrigin(x: 0, y: 0, z: 0),
                size: MTLSize(width: Int(maskSize.width), height: Int(maskSize.height), depth: 1)
            ),
            mipmapLevel: 0,
            withBytes: &zeros,
            bytesPerRow: Int(maskSize.width)
        )

        return texture
    }

    /// Upload mask data to texture
    func uploadMask(_ data: [UInt8]) throws {
        let texture = try getMaskTexture()

        guard data.count == Int(maskSize.width) * Int(maskSize.height) else {
            throw TextureError.invalidMaskData
        }

        var mutableData = data
        texture.replace(
            region: MTLRegion(
                origin: MTLOrigin(x: 0, y: 0, z: 0),
                size: MTLSize(width: Int(maskSize.width), height: Int(maskSize.height), depth: 1)
            ),
            mipmapLevel: 0,
            withBytes: &mutableData,
            bytesPerRow: Int(maskSize.width)
        )
    }

    /// Read mask data from texture
    func readMask() -> [UInt8]? {
        guard let texture = maskTexture else { return nil }

        let width = texture.width
        let height = texture.height
        var data = [UInt8](repeating: 0, count: width * height)

        texture.getBytes(
            &data,
            bytesPerRow: width,
            from: MTLRegion(
                origin: MTLOrigin(x: 0, y: 0, z: 0),
                size: MTLSize(width: width, height: height, depth: 1)
            ),
            mipmapLevel: 0
        )

        return data
    }

    /// Read a region from mask texture (for undo patches)
    func readMaskRegion(bbox: CGRect) -> Data? {
        guard let texture = maskTexture else { return nil }

        // Use floor for min and ceil for max to ensure we capture all affected pixels
        let minX = max(0, Int(floor(bbox.minX)))
        let minY = max(0, Int(floor(bbox.minY)))
        let maxX = min(texture.width, Int(ceil(bbox.maxX)))
        let maxY = min(texture.height, Int(ceil(bbox.maxY)))

        let regionWidth = maxX - minX
        let regionHeight = maxY - minY

        guard regionWidth > 0 && regionHeight > 0 else { return nil }

        var data = [UInt8](repeating: 0, count: regionWidth * regionHeight)

        texture.getBytes(
            &data,
            bytesPerRow: regionWidth,
            from: MTLRegion(
                origin: MTLOrigin(x: minX, y: minY, z: 0),
                size: MTLSize(width: regionWidth, height: regionHeight, depth: 1)
            ),
            mipmapLevel: 0
        )

        return Data(data)
    }

    /// Write a region to mask texture (for undo restore)
    func writeMaskRegion(bbox: CGRect, data: Data) {
        guard let texture = maskTexture else { return }

        // Use floor for min and ceil for max to match readMaskRegion
        let minX = max(0, Int(floor(bbox.minX)))
        let minY = max(0, Int(floor(bbox.minY)))
        let maxX = min(texture.width, Int(ceil(bbox.maxX)))
        let maxY = min(texture.height, Int(ceil(bbox.maxY)))

        let regionWidth = maxX - minX
        let regionHeight = maxY - minY

        guard regionWidth > 0 && regionHeight > 0 else { return }

        data.withUnsafeBytes { buffer in
            guard let ptr = buffer.baseAddress else { return }
            texture.replace(
                region: MTLRegion(
                    origin: MTLOrigin(x: minX, y: minY, z: 0),
                    size: MTLSize(width: regionWidth, height: regionHeight, depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: ptr,
                bytesPerRow: regionWidth
            )
        }
    }

    /// Clear the mask texture
    func clearMask() {
        guard let texture = maskTexture else { return }

        let byteCount = texture.width * texture.height
        var zeros = [UInt8](repeating: 0, count: byteCount)
        texture.replace(
            region: MTLRegion(
                origin: MTLOrigin(x: 0, y: 0, z: 0),
                size: MTLSize(width: texture.width, height: texture.height, depth: 1)
            ),
            mipmapLevel: 0,
            withBytes: &zeros,
            bytesPerRow: texture.width
        )
    }

    /// Clear all textures
    func clear() {
        imageTexture = nil
        maskTexture = nil
        imageSize = .zero
        maskSize = .zero
    }
}

// MARK: - Errors

enum TextureError: Error, LocalizedError {
    case invalidImage
    case textureCreationFailed
    case invalidMaskData
    case noImageLoaded

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Failed to load image"
        case .textureCreationFailed:
            return "Failed to create Metal texture"
        case .invalidMaskData:
            return "Invalid mask data size"
        case .noImageLoaded:
            return "No image loaded - cannot create mask texture"
        }
    }
}
