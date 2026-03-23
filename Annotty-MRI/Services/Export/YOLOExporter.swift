import Foundation
import CoreGraphics

/// Exports masks in YOLO-seg format
/// Format: class_id x1 y1 x2 y2 ... xn yn (normalized 0-1)
class YOLOExporter {
    /// Export masks to YOLO-seg format
    /// - Parameters:
    ///   - masks: Dictionary of class ID to mask
    ///   - classes: Class definitions
    ///   - imageSize: Original image size
    ///   - scaleFactor: Scale factor from image to mask
    /// - Returns: Text content as Data
    func export(
        masks: [Int: InternalMask],
        classes: [MaskClass],
        imageSize: CGSize,
        scaleFactor: Float
    ) -> Data? {
        var lines: [String] = []

        let imageWidth = Double(imageSize.width)
        let imageHeight = Double(imageSize.height)

        for (classID, mask) in masks {
            // Extract contours
            let contours = ContourExtractor.extractContours(
                from: mask.data,
                width: mask.width,
                height: mask.height
            )

            for contour in contours {
                // Simplify contour
                let simplified = ContourExtractor.simplifyContour(contour, epsilon: 2.0)

                guard simplified.count >= 3 else { continue }

                // Scale to image coordinates
                let scaled = ContourExtractor.scaleContour(simplified, scaleFactor: scaleFactor)

                // Normalize to 0-1 range
                let normalized = scaled.map { point -> String in
                    let x = Double(point.x) / imageWidth
                    let y = Double(point.y) / imageHeight
                    // Clamp to valid range
                    let clampedX = max(0, min(1, x))
                    let clampedY = max(0, min(1, y))
                    return String(format: "%.6f %.6f", clampedX, clampedY)
                }

                // Format: class_id x1 y1 x2 y2 ... xn yn
                let line = "\(classID) " + normalized.joined(separator: " ")
                lines.append(line)
            }
        }

        let content = lines.joined(separator: "\n")
        return content.data(using: .utf8)
    }
}
