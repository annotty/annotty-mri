import Foundation
import CoreGraphics
import UIKit

/// Loads and processes annotation files
/// Handles:
/// - Auto-detection of matching annotations
/// - Color mask parsing
/// - Scaling to internal mask resolution
class AnnotationLoader {
    // MARK: - Dependencies

    private let colorParser = ColorMaskParser()
    private let projectService = ProjectFileService.shared

    // MARK: - Load Annotation

    /// Load annotation for an image if it exists
    /// - Parameters:
    ///   - imageURL: URL of the source image
    ///   - imageSize: Size of the source image (for scaling calculation)
    /// - Returns: Loaded annotation with masks, or nil if not found
    func loadAnnotation(for imageURL: URL, imageSize: CGSize) -> LoadedAnnotation? {
        // Check if annotation exists
        guard let annotationURL = projectService.getAnnotationURL(for: imageURL),
              projectService.annotationExists(for: imageURL) else {
            return nil
        }

        // Load annotation image
        guard let annotationImage = loadImage(from: annotationURL) else {
            print("Failed to load annotation image: \(annotationURL)")
            return nil
        }

        // Parse color annotation
        let parsed = colorParser.parse(image: annotationImage)

        // Calculate mask dimensions with 4096 max clamp
        let (maskWidth, maskHeight, scaleFactor) = InternalMask.calculateDimensions(for: imageSize)

        // Scale masks to internal resolution
        var scaledMasks: [Int: InternalMask] = [:]

        for (classID, maskData) in parsed.masks {
            let scaledData = scaleMask(
                data: maskData,
                fromWidth: parsed.width,
                fromHeight: parsed.height,
                toWidth: maskWidth,
                toHeight: maskHeight
            )

            scaledMasks[classID] = InternalMask(
                data: scaledData,
                width: maskWidth,
                height: maskHeight,
                classID: classID,
                scaleFactor: scaleFactor
            )
        }

        return LoadedAnnotation(
            classes: parsed.classes,
            masks: scaledMasks,
            scaleFactor: scaleFactor
        )
    }

    // MARK: - Image Loading

    private func loadImage(from url: URL) -> CGImage? {
        guard let uiImage = UIImage(contentsOfFile: url.path),
              let cgImage = uiImage.cgImage else {
            return nil
        }
        return cgImage
    }

    // MARK: - Mask Scaling

    /// Scale mask data using nearest neighbor interpolation
    private func scaleMask(
        data: [UInt8],
        fromWidth: Int,
        fromHeight: Int,
        toWidth: Int,
        toHeight: Int
    ) -> [UInt8] {
        var scaledData = [UInt8](repeating: 0, count: toWidth * toHeight)

        let scaleX = Float(fromWidth) / Float(toWidth)
        let scaleY = Float(fromHeight) / Float(toHeight)

        for y in 0..<toHeight {
            for x in 0..<toWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)

                let srcIndex = min(srcY * fromWidth + srcX, data.count - 1)
                let dstIndex = y * toWidth + x

                scaledData[dstIndex] = data[srcIndex]
            }
        }

        return scaledData
    }
}

// MARK: - Result Types

/// Result of loading an annotation
struct LoadedAnnotation {
    /// Detected classes
    let classes: [MaskClass]

    /// Masks per class (at internal resolution)
    let masks: [Int: InternalMask]

    /// Scale factor from image to mask
    let scaleFactor: Float
}
