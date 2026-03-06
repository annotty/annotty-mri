import Foundation
import CoreGraphics
import Metal

/// Internal mask representation at high resolution
/// Values are strictly 0 or 1 (UInt8), Boolean arrays must not be used
struct InternalMask {
    /// Mask data stored as UInt8 buffer (0 or 1 only)
    private(set) var data: [UInt8]

    /// Width of the mask in pixels
    let width: Int

    /// Height of the mask in pixels
    let height: Int

    /// Class ID this mask belongs to
    let classID: Int

    /// Scale factor from original image to this mask
    /// Typically 2.0, but clamped if original image > 2048px
    let scaleFactor: Float

    /// Maximum mask dimension (4096px)
    static let maxDimension = 4096

    /// Initialize an empty mask
    init(width: Int, height: Int, classID: Int, scaleFactor: Float) {
        self.width = width
        self.height = height
        self.classID = classID
        self.scaleFactor = scaleFactor
        self.data = [UInt8](repeating: 0, count: width * height)
    }

    /// Initialize from existing data
    init(data: [UInt8], width: Int, height: Int, classID: Int, scaleFactor: Float) {
        precondition(data.count == width * height, "Data size must match width * height")
        self.data = data
        self.width = width
        self.height = height
        self.classID = classID
        self.scaleFactor = scaleFactor
    }

    /// Calculate mask dimensions for a given image size
    /// - Parameter imageSize: Original image size
    /// - Returns: Tuple of (maskWidth, maskHeight, scaleFactor)
    static func calculateDimensions(for imageSize: CGSize) -> (width: Int, height: Int, scaleFactor: Float) {
        let maxEdge = max(imageSize.width, imageSize.height)
        let scaleFactor = min(2.0, Float(maxDimension) / Float(maxEdge))

        let maskWidth = Int(imageSize.width * CGFloat(scaleFactor))
        let maskHeight = Int(imageSize.height * CGFloat(scaleFactor))

        return (maskWidth, maskHeight, scaleFactor)
    }

    /// Get value at pixel coordinate
    func getValue(at x: Int, y: Int) -> UInt8 {
        guard x >= 0, x < width, y >= 0, y < height else { return 0 }
        return data[y * width + x]
    }

    /// Set value at pixel coordinate (0 or 1 only)
    mutating func setValue(_ value: UInt8, at x: Int, y: Int) {
        guard x >= 0, x < width, y >= 0, y < height else { return }
        data[y * width + x] = value > 0 ? 1 : 0
    }

    /// Extract a rectangular region as raw Data (for undo patches)
    func extractRegion(bbox: CGRect) -> Data {
        let minX = max(0, Int(bbox.minX))
        let minY = max(0, Int(bbox.minY))
        let maxX = min(width, Int(bbox.maxX))
        let maxY = min(height, Int(bbox.maxY))

        let regionWidth = maxX - minX
        let regionHeight = maxY - minY

        var regionData = [UInt8](repeating: 0, count: regionWidth * regionHeight)

        for y in minY..<maxY {
            for x in minX..<maxX {
                let srcIndex = y * width + x
                let dstIndex = (y - minY) * regionWidth + (x - minX)
                regionData[dstIndex] = data[srcIndex]
            }
        }

        return Data(regionData)
    }

    /// Restore a rectangular region from raw Data (for undo)
    mutating func restoreRegion(bbox: CGRect, from regionData: Data) {
        let minX = max(0, Int(bbox.minX))
        let minY = max(0, Int(bbox.minY))
        let maxX = min(width, Int(bbox.maxX))
        let maxY = min(height, Int(bbox.maxY))

        let regionWidth = maxX - minX

        regionData.withUnsafeBytes { buffer in
            guard let ptr = buffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }

            for y in minY..<maxY {
                for x in minX..<maxX {
                    let srcIndex = (y - minY) * regionWidth + (x - minX)
                    let dstIndex = y * width + x
                    data[dstIndex] = ptr[srcIndex]
                }
            }
        }
    }

    /// Apply a circular stamp at the given center point
    mutating func applyStamp(center: CGPoint, radius: Float, value: UInt8) {
        let centerX = Int(center.x)
        let centerY = Int(center.y)
        let intRadius = Int(ceil(radius))

        let minX = max(0, centerX - intRadius)
        let maxX = min(width - 1, centerX + intRadius)
        let minY = max(0, centerY - intRadius)
        let maxY = min(height - 1, centerY + intRadius)

        let radiusSquared = radius * radius
        let paintValue: UInt8 = value > 0 ? 1 : 0

        for y in minY...maxY {
            for x in minX...maxX {
                let dx = Float(x - centerX)
                let dy = Float(y - centerY)
                if dx * dx + dy * dy <= radiusSquared {
                    data[y * width + x] = paintValue
                }
            }
        }
    }
}

