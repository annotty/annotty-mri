import Foundation
import CoreGraphics
import UIKit
import Combine

/// Represents a single image in the project with its metadata
struct ImageItem: Identifiable, Equatable {
    /// Unique identifier
    let id: UUID

    /// File URL of the source image
    let url: URL

    /// Base name without extension (used for matching annotations)
    var baseName: String {
        url.deletingPathExtension().lastPathComponent
    }

    /// File extension
    var fileExtension: String {
        url.pathExtension.lowercased()
    }

    /// Original image size (set after loading)
    var imageSize: CGSize?

    /// Whether an annotation file exists for this image
    var hasAnnotation: Bool = false

    /// Path to the corresponding annotation file (if exists)
    var annotationURL: URL?

    init(url: URL) {
        self.id = UUID()
        self.url = url
    }

    /// Check if this is a supported image format
    var isSupportedFormat: Bool {
        ["png", "jpg", "jpeg"].contains(fileExtension)
    }

    static func == (lhs: ImageItem, rhs: ImageItem) -> Bool {
        lhs.id == rhs.id
    }
}

/// Manages the list of images in the project
class ImageItemManager: ObservableObject {
    @Published private(set) var items: [ImageItem] = []
    @Published var currentIndex: Int = 0

    var currentItem: ImageItem? {
        guard currentIndex >= 0 && currentIndex < items.count else { return nil }
        return items[currentIndex]
    }

    var totalCount: Int {
        items.count
    }

    var currentPosition: Int {
        currentIndex + 1
    }

    /// Load images from a directory
    func loadImages(from directory: URL) {
        let fileManager = FileManager.default

        guard let contents = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        items = contents
            .map { ImageItem(url: $0) }
            .filter { $0.isSupportedFormat }
            .sorted { $0.baseName.localizedStandardCompare($1.baseName) == .orderedAscending }

        currentIndex = items.isEmpty ? -1 : 0
    }

    /// Set images from URL array (from ProjectFileService)
    func setImages(_ urls: [URL]) {
        items = urls.map { url in
            var item = ImageItem(url: url)
            if let annotationURL = ProjectFileService.shared.getAnnotationURL(for: url),
               FileManager.default.fileExists(atPath: annotationURL.path) {
                item.hasAnnotation = true
                item.annotationURL = annotationURL
            }
            return item
        }
        currentIndex = items.isEmpty ? 0 : 0
    }

    /// Check for existing annotations
    func checkAnnotations(in annotationsDirectory: URL) {
        let fileManager = FileManager.default

        for i in 0..<items.count {
            let annotationPath = annotationsDirectory
                .appendingPathComponent(items[i].baseName)
                .appendingPathExtension("png")

            if fileManager.fileExists(atPath: annotationPath.path) {
                items[i].hasAnnotation = true
                items[i].annotationURL = annotationPath
            }
        }
    }

    /// Navigate to previous image
    func previous() {
        guard currentIndex > 0 else { return }
        currentIndex -= 1
    }

    /// Navigate to next image
    func next() {
        guard currentIndex < items.count - 1 else { return }
        currentIndex += 1
    }

    /// Navigate to specific index
    func goTo(index: Int) {
        guard index >= 0 && index < items.count else { return }
        currentIndex = index
    }
}

