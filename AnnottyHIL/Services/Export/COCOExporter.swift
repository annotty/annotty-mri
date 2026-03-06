import Foundation
import CoreGraphics

/// COCO format annotation structures
struct COCOAnnotation: Codable {
    let images: [COCOImage]
    let annotations: [COCOSegmentation]
    let categories: [COCOCategory]
}

struct COCOImage: Codable {
    let id: Int
    let file_name: String
    let width: Int
    let height: Int
}

struct COCOSegmentation: Codable {
    let id: Int
    let image_id: Int
    let category_id: Int
    let segmentation: [[Double]]
    let area: Double
    let bbox: [Double]
    let iscrowd: Int
}

struct COCOCategory: Codable {
    let id: Int
    let name: String
    let supercategory: String
}

/// Exports masks in COCO JSON format
class COCOExporter {
    /// Export masks to COCO JSON format
    /// - Parameters:
    ///   - masks: Dictionary of class ID to mask
    ///   - classes: Class definitions
    ///   - imageName: Name of the source image
    ///   - imageSize: Original image size
    ///   - scaleFactor: Scale factor from image to mask
    /// - Returns: JSON data
    func export(
        masks: [Int: InternalMask],
        classes: [MaskClass],
        imageName: String,
        imageSize: CGSize,
        scaleFactor: Float
    ) -> Data? {
        let imageWidth = Int(imageSize.width)
        let imageHeight = Int(imageSize.height)

        // Create image entry
        let image = COCOImage(
            id: 1,
            file_name: imageName,
            width: imageWidth,
            height: imageHeight
        )

        // Create categories
        let categories = classes.map { maskClass in
            COCOCategory(
                id: maskClass.id,
                name: maskClass.name,
                supercategory: "annotation"
            )
        }

        // Create segmentation annotations
        var annotations: [COCOSegmentation] = []
        var annotationID = 1

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

                // Convert to flat array of doubles
                let segmentation = scaled.flatMap { [Double($0.x), Double($0.y)] }

                // Calculate bounding box
                let bbox = calculateBbox(from: scaled)

                // Calculate area
                let area = calculateArea(from: scaled)

                let annotation = COCOSegmentation(
                    id: annotationID,
                    image_id: 1,
                    category_id: classID,
                    segmentation: [segmentation],
                    area: area,
                    bbox: bbox,
                    iscrowd: 0
                )

                annotations.append(annotation)
                annotationID += 1
            }
        }

        // Create COCO annotation structure
        let cocoAnnotation = COCOAnnotation(
            images: [image],
            annotations: annotations,
            categories: categories
        )

        // Encode to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        return try? encoder.encode(cocoAnnotation)
    }

    /// Calculate bounding box [x, y, width, height]
    private func calculateBbox(from points: [CGPoint]) -> [Double] {
        guard !points.isEmpty else { return [0, 0, 0, 0] }

        let minX = points.map { $0.x }.min() ?? 0
        let minY = points.map { $0.y }.min() ?? 0
        let maxX = points.map { $0.x }.max() ?? 0
        let maxY = points.map { $0.y }.max() ?? 0

        return [Double(minX), Double(minY), Double(maxX - minX), Double(maxY - minY)]
    }

    /// Calculate polygon area using shoelace formula
    private func calculateArea(from points: [CGPoint]) -> Double {
        guard points.count >= 3 else { return 0 }

        var area: Double = 0

        for i in 0..<points.count {
            let j = (i + 1) % points.count
            area += Double(points[i].x * points[j].y)
            area -= Double(points[j].x * points[i].y)
        }

        return abs(area) / 2.0
    }
}
