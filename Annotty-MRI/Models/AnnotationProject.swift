import Foundation
import Combine

/// Represents an annotation project with its folder structure and state
class AnnotationProject: ObservableObject {
    // MARK: - Properties

    /// Project root directory
    let rootURL: URL

    /// Image items in the project
    @Published private(set) var images: [ImageItem] = []

    /// Current image index
    @Published var currentIndex: Int = 0

    /// Classes detected in the project
    @Published private(set) var classes: [MaskClass] = []

    // MARK: - Services

    private let projectService = ProjectFileService.shared
    private let annotationLoader = AnnotationLoader()

    // MARK: - Computed Properties

    var currentImage: ImageItem? {
        guard currentIndex >= 0 && currentIndex < images.count else { return nil }
        return images[currentIndex]
    }

    var totalImages: Int { images.count }
    var displayIndex: Int { images.isEmpty ? 0 : currentIndex + 1 }

    // MARK: - Initialization

    init(rootURL: URL) throws {
        self.rootURL = rootURL
        try projectService.initializeProject(at: rootURL)
        loadImages()
    }

    // MARK: - Image Management

    private func loadImages() {
        let imageURLs = projectService.getImageURLs()
        images = imageURLs.map { ImageItem(url: $0) }

        // Check for existing annotations
        for i in 0..<images.count {
            images[i].hasAnnotation = projectService.annotationExists(for: images[i].url)
            if images[i].hasAnnotation {
                images[i].annotationURL = projectService.getAnnotationURL(for: images[i].url)
            }
        }
    }

    // MARK: - Navigation

    func goToPrevious() {
        guard currentIndex > 0 else { return }
        currentIndex -= 1
    }

    func goToNext() {
        guard currentIndex < images.count - 1 else { return }
        currentIndex += 1
    }

    func goTo(index: Int) {
        guard index >= 0 && index < images.count else { return }
        currentIndex = index
    }

    // MARK: - Class Management

    /// Add a new class (if under limit)
    func addClass(_ newClass: MaskClass) -> Bool {
        guard classes.count < MaskClass.maxClasses else {
            return false
        }
        classes.append(newClass)
        return true
    }

    /// Update classes from loaded annotation
    func updateClasses(_ newClasses: [MaskClass]) {
        // Merge with existing classes, respecting the max limit
        for newClass in newClasses {
            if !classes.contains(where: { $0.id == newClass.id }) {
                if classes.count < MaskClass.maxClasses {
                    classes.append(newClass)
                }
            }
        }
    }

    /// Clear all classes
    func clearClasses() {
        classes.removeAll()
    }
}

