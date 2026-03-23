import Foundation
import CoreGraphics
import UIKit

/// Result of parsing a color annotation PNG
struct ParsedAnnotation {
    /// Detected classes with their original colors
    var classes: [MaskClass]

    /// Mask data per class (1x resolution, will be scaled to 2x)
    var masks: [Int: [UInt8]]

    /// Original image dimensions
    var width: Int
    var height: Int
}

/// Parses color annotation PNGs into class-separated binary masks
/// - White (#FFFFFF) is treated as background
/// - Each unique color becomes a separate class
/// - Anti-aliased pixels are snapped to nearest solid color
class ColorMaskParser {
    /// Background color (white)
    static let backgroundColor: UInt32 = 0xFFFFFF

    /// Distance threshold for color snapping (for anti-aliased pixels)
    static let colorSnapThreshold: Int = 30

    /// Parse a color annotation image
    /// - Parameter image: The CGImage to parse
    /// - Returns: ParsedAnnotation with classes and masks
    func parse(image: CGImage) -> ParsedAnnotation {
        let width = image.width
        let height = image.height

        // Get pixel data
        guard let pixelData = getPixelData(from: image, width: width, height: height) else {
            return ParsedAnnotation(classes: [], masks: [:], width: width, height: height)
        }

        // First pass: collect all unique colors (excluding background)
        var colorSet = Set<UInt32>()
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let r = pixelData[offset]
                let g = pixelData[offset + 1]
                let b = pixelData[offset + 2]

                let colorKey = UInt32(r) << 16 | UInt32(g) << 8 | UInt32(b)

                // Skip background (white)
                if colorKey == Self.backgroundColor {
                    continue
                }

                // Skip near-white colors (likely anti-aliased background)
                if isNearWhite(r: r, g: g, b: b) {
                    continue
                }

                colorSet.insert(colorKey)
            }
        }

        // Build color to class mapping
        var colorToClassID: [UInt32: Int] = [:]
        var classes: [MaskClass] = []
        var nextClassID = 0

        // Sort colors for consistent ordering
        let sortedColors = colorSet.sorted()

        // Check class limit
        let maxClasses = min(sortedColors.count, MaskClass.maxClasses)

        for i in 0..<maxClasses {
            let colorKey = sortedColors[i]
            let r = UInt8((colorKey >> 16) & 0xFF)
            let g = UInt8((colorKey >> 8) & 0xFF)
            let b = UInt8(colorKey & 0xFF)

            colorToClassID[colorKey] = nextClassID
            classes.append(MaskClass(id: nextClassID, r: r, g: g, b: b))
            nextClassID += 1
        }

        // Second pass: create masks
        var masks: [Int: [UInt8]] = [:]
        for classID in 0..<nextClassID {
            masks[classID] = [UInt8](repeating: 0, count: width * height)
        }

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let r = pixelData[offset]
                let g = pixelData[offset + 1]
                let b = pixelData[offset + 2]

                let colorKey = UInt32(r) << 16 | UInt32(g) << 8 | UInt32(b)

                // Try exact match first
                if let classID = colorToClassID[colorKey] {
                    masks[classID]![y * width + x] = 1
                    continue
                }

                // Skip background
                if colorKey == Self.backgroundColor || isNearWhite(r: r, g: g, b: b) {
                    continue
                }

                // Snap to nearest color (for anti-aliased pixels)
                if let nearestClassID = findNearestClass(
                    r: r, g: g, b: b,
                    colorToClassID: colorToClassID
                ) {
                    masks[nearestClassID]![y * width + x] = 1
                }
            }
        }

        return ParsedAnnotation(
            classes: classes,
            masks: masks,
            width: width,
            height: height
        )
    }

    /// Extract pixel data from CGImage
    private func getPixelData(from image: CGImage, width: Int, height: Int) -> [UInt8]? {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelData
    }

    /// Check if color is near white (for background detection)
    private func isNearWhite(r: UInt8, g: UInt8, b: UInt8) -> Bool {
        let threshold: UInt8 = 250
        return r >= threshold && g >= threshold && b >= threshold
    }

    /// Find nearest class for anti-aliased pixel
    private func findNearestClass(
        r: UInt8, g: UInt8, b: UInt8,
        colorToClassID: [UInt32: Int]
    ) -> Int? {
        var nearestClassID: Int?
        var minDistance = Int.max

        for (colorKey, classID) in colorToClassID {
            let cr = UInt8((colorKey >> 16) & 0xFF)
            let cg = UInt8((colorKey >> 8) & 0xFF)
            let cb = UInt8(colorKey & 0xFF)

            let distance = colorDistance(r1: r, g1: g, b1: b, r2: cr, g2: cg, b2: cb)

            if distance < minDistance && distance < Self.colorSnapThreshold {
                minDistance = distance
                nearestClassID = classID
            }
        }

        return nearestClassID
    }

    /// Calculate color distance (simple Euclidean in RGB space)
    private func colorDistance(
        r1: UInt8, g1: UInt8, b1: UInt8,
        r2: UInt8, g2: UInt8, b2: UInt8
    ) -> Int {
        let dr = Int(r1) - Int(r2)
        let dg = Int(g1) - Int(g2)
        let db = Int(b1) - Int(b2)
        return Int(sqrt(Double(dr * dr + dg * dg + db * db)))
    }
}
