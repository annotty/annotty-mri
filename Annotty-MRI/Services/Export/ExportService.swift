import Foundation
import CoreGraphics

/// Coordinates export operations for all formats
class ExportService {
    // MARK: - Singleton

    static let shared = ExportService()

    // MARK: - Exporters

    private let pngExporter = PNGExporter()
    private let cocoExporter = COCOExporter()
    private let yoloExporter = YOLOExporter()

    // MARK: - Dependencies

    private let projectService = ProjectFileService.shared

    // MARK: - Initialization

    private init() {}

    // MARK: - Export

    /// Export all selected formats
    /// - Parameters:
    ///   - masks: Dictionary of class ID to mask
    ///   - classes: Class definitions
    ///   - imageURL: Source image URL
    ///   - imageSize: Original image size
    ///   - scaleFactor: Scale factor from image to mask
    ///   - formats: Formats to export
    /// - Returns: URLs of exported files
    func export(
        masks: [Int: InternalMask],
        classes: [MaskClass],
        imageURL: URL,
        imageSize: CGSize,
        scaleFactor: Float,
        formats: Set<ExportFormat>
    ) throws -> [URL] {
        var exportedURLs: [URL] = []
        let imageName = imageURL.lastPathComponent

        // Export PNG
        if formats.contains(.png) {
            let pngData = pngExporter.export(
                masks: masks,
                classes: classes,
                imageSize: imageSize,
                mode: .colorMask,
                scaleFactor: scaleFactor
            )

            for (filename, data) in pngData {
                if let url = projectService.getLabelURL(for: imageURL, format: .png) {
                    try data.write(to: url)
                    exportedURLs.append(url)
                }
            }
        }

        // Export COCO
        if formats.contains(.coco) {
            if let cocoData = cocoExporter.export(
                masks: masks,
                classes: classes,
                imageName: imageName,
                imageSize: imageSize,
                scaleFactor: scaleFactor
            ) {
                if let url = projectService.getLabelURL(for: imageURL, format: .coco) {
                    try cocoData.write(to: url)
                    exportedURLs.append(url)
                }
            }
        }

        // Export YOLO
        if formats.contains(.yolo) {
            if let yoloData = yoloExporter.export(
                masks: masks,
                classes: classes,
                imageSize: imageSize,
                scaleFactor: scaleFactor
            ) {
                if let url = projectService.getLabelURL(for: imageURL, format: .yolo) {
                    try yoloData.write(to: url)
                    exportedURLs.append(url)
                }
            }
        }

        return exportedURLs
    }
}
