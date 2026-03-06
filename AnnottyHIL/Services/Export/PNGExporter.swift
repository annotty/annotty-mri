import Foundation
import CoreGraphics
import UIKit
import SwiftUI

/// Export mode for PNG masks
enum PNGExportMode {
    /// All classes in one image with colors
    case colorMask
    /// Separate binary PNG per class
    case classSeparate
}

/// Exports masks as PNG images
class PNGExporter {
    /// Export masks to PNG
    /// - Parameters:
    ///   - masks: Dictionary of class ID to mask data
    ///   - classes: Class definitions with colors
    ///   - imageSize: Original image size
    ///   - mode: Export mode (color mask or separate)
    ///   - scaleFactor: Scale factor from image to mask
    /// - Returns: Dictionary of filenames to PNG data
    func export(
        masks: [Int: InternalMask],
        classes: [MaskClass],
        imageSize: CGSize,
        mode: PNGExportMode,
        scaleFactor: Float
    ) -> [String: Data] {
        switch mode {
        case .colorMask:
            if let data = exportColorMask(masks: masks, classes: classes, imageSize: imageSize, scaleFactor: scaleFactor) {
                return ["mask.png": data]
            }
            return [:]

        case .classSeparate:
            return exportSeparateMasks(masks: masks, classes: classes, imageSize: imageSize, scaleFactor: scaleFactor)
        }
    }

    /// Export all classes as a single color mask
    private func exportColorMask(
        masks: [Int: InternalMask],
        classes: [MaskClass],
        imageSize: CGSize,
        scaleFactor: Float
    ) -> Data? {
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)

        // Create pixel buffer (RGBA)
        var pixels = [UInt8](repeating: 255, count: width * height * 4)

        // Build class color lookup
        var classColors: [Int: (r: UInt8, g: UInt8, b: UInt8)] = [:]
        for maskClass in classes {
            let uiColor = UIColor(maskClass.originalColor)
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0
            uiColor.getRed(&r, green: &g, blue: &b, alpha: nil)
            classColors[maskClass.id] = (
                r: UInt8(r * 255),
                g: UInt8(g * 255),
                b: UInt8(b * 255)
            )
        }

        // Fill pixels
        for (classID, mask) in masks {
            guard let color = classColors[classID] else { continue }

            let invScale = 1.0 / scaleFactor

            for y in 0..<height {
                for x in 0..<width {
                    // Map to mask coordinates
                    let maskX = Int(Float(x) * scaleFactor)
                    let maskY = Int(Float(y) * scaleFactor)

                    guard maskX < mask.width && maskY < mask.height else { continue }

                    if mask.getValue(at: maskX, y: maskY) == 1 {
                        let pixelIndex = (y * width + x) * 4
                        pixels[pixelIndex] = color.r
                        pixels[pixelIndex + 1] = color.g
                        pixels[pixelIndex + 2] = color.b
                        pixels[pixelIndex + 3] = 255
                    }
                }
            }
        }

        return createPNG(from: pixels, width: width, height: height)
    }

    /// Export each class as a separate binary mask
    private func exportSeparateMasks(
        masks: [Int: InternalMask],
        classes: [MaskClass],
        imageSize: CGSize,
        scaleFactor: Float
    ) -> [String: Data] {
        var result: [String: Data] = [:]
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)

        for (classID, mask) in masks {
            let className = classes.first { $0.id == classID }?.name ?? "class\(classID)"
            let filename = "mask_\(className).png"

            // Create binary mask pixels (grayscale)
            var pixels = [UInt8](repeating: 0, count: width * height * 4)

            for y in 0..<height {
                for x in 0..<width {
                    let maskX = Int(Float(x) * scaleFactor)
                    let maskY = Int(Float(y) * scaleFactor)

                    guard maskX < mask.width && maskY < mask.height else { continue }

                    let value: UInt8 = mask.getValue(at: maskX, y: maskY) == 1 ? 255 : 0
                    let pixelIndex = (y * width + x) * 4
                    pixels[pixelIndex] = value
                    pixels[pixelIndex + 1] = value
                    pixels[pixelIndex + 2] = value
                    pixels[pixelIndex + 3] = 255
                }
            }

            if let data = createPNG(from: pixels, width: width, height: height) {
                result[filename] = data
            }
        }

        return result
    }

    /// Create PNG data from pixel buffer
    private func createPNG(from pixels: [UInt8], width: Int, height: Int) -> Data? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        var mutablePixels = pixels

        guard let context = CGContext(
            data: &mutablePixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ),
              let cgImage = context.makeImage() else {
            return nil
        }

        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.pngData()
    }
}
