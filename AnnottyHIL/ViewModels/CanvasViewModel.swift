import SwiftUI
import Combine
import simd

nonisolated(unsafe) private let classRGBColors: [(UInt8, UInt8, UInt8)] = [
    (255, 0, 0),       // 1: red
    (255, 128, 0),     // 2: orange
    (255, 255, 0),     // 3: yellow
    (0, 255, 0),       // 4: green
    (0, 255, 255),     // 5: cyan
    (0, 0, 255),       // 6: blue
    (128, 0, 255),     // 7: purple
    (255, 102, 178)    // 8: pink
]

nonisolated(unsafe) private let exactColorLookup: [UInt32: UInt8] = {
    var table = [UInt32: UInt8]()
    for (index, (r, g, b)) in classRGBColors.enumerated() {
        let key = UInt32(r) << 16 | UInt32(g) << 8 | UInt32(b)
        table[key] = UInt8(index + 1)
    }
    return table
}()

/// Main state coordinator for the canvas
/// Manages drawing state, image navigation, and coordinates with Metal renderer
class CanvasViewModel: ObservableObject {
    // MARK: - Renderer

    @Published private(set) var renderer: MetalRenderer?

    // MARK: - Gesture Coordinator

    let gestureCoordinator = GestureCoordinator()

    // MARK: - Drawing State

    @Published var brushRadius: Float = 20
    @Published var isPainting: Bool = true
    @Published var isDrawing: Bool = false
    @Published var isFillMode: Bool = false {
        didSet {
            gestureCoordinator.isFillMode = isFillMode
            // Disable other modes when fill mode is enabled
            if isFillMode {
                if isSmoothMode { isSmoothMode = false }
            }
        }
    }

    /// Smooth mode for boundary smoothing
    @Published var isSmoothMode: Bool = false {
        didSet {
            gestureCoordinator.isSmoothMode = isSmoothMode
            // Disable other modes when smooth mode is enabled
            if isSmoothMode {
                if isFillMode { isFillMode = false }
            }
        }
    }

    /// U-Net processing state (during inference)
    @Published private(set) var isUNetProcessing: Bool = false

    /// U-Net model loading state (initial model load)
    @Published private(set) var isUNetLoading: Bool = false

    /// U-Net ensemble service for segmentation
    let unetService = UNetService()
    @Published var lastDrawPoint: CGPoint = .zero
    @Published var currentScale: CGFloat = 1.0  // Current zoom level for UI updates

    // MARK: - Class Management

    /// Preset colors mapped to class IDs (index+1 = classID)
    /// These must match the colors in MetalRenderer.classColors exactly
    /// Class 1=red, 2=orange, 3=yellow, 4=green, 5=cyan, 6=blue, 7=purple, 8=pink
    static let classColors: [Color] = [
        Color(red: 1, green: 0, blue: 0),        // 1: red
        Color(red: 1, green: 0.5, blue: 0),      // 2: orange
        Color(red: 1, green: 1, blue: 0),        // 3: yellow
        Color(red: 0, green: 1, blue: 0),        // 4: green
        Color(red: 0, green: 1, blue: 1),        // 5: cyan
        Color(red: 0, green: 0, blue: 1),        // 6: blue
        Color(red: 0.5, green: 0, blue: 1),      // 7: purple
        Color(red: 1, green: 0.4, blue: 0.7)     // 8: pink
    ]

    /// Current active class ID (1-8, 0 = eraser/background)
    @Published private(set) var currentClassID: Int = 1

    /// Custom class names (index 0-7 = class 1-8)
    /// Empty string means unnamed class
    @Published var classNames: [String] = Array(repeating: "", count: 8) {
        didSet {
            saveClassNames()
        }
    }

    /// Class IDs whose annotations are temporarily hidden (not rendered)
    @Published var hiddenClassIDs: Set<Int> = [] {
        didSet {
            renderer?.hiddenClassIDs = hiddenClassIDs
        }
    }

    /// UserDefaults key for class names
    private static let classNamesKey = "annotty.classNames"

    /// UserDefaults key for last viewed image
    private static let lastImageNameKey = "annotty.lastImageName"

    // MARK: - UI Preferences

    // MARK: - Display Settings

    @Published var annotationColor: Color = Color(red: 1, green: 0, blue: 0) {
        didSet {
            // Update currentClassID based on selected color (index + 1)
            if let index = Self.classColors.firstIndex(of: annotationColor) {
                currentClassID = index + 1
                renderer?.setCurrentClass(currentClassID)
            }
        }
    }

    /// Image contrast (0.0 - 2.0, 1.0 = normal, 0% - 200%)
    @Published var imageContrast: Float = 1.0 {
        didSet {
            renderer?.imageContrast = imageContrast
        }
    }

    /// Image brightness (-1.0 to 1.0, 0.0 = normal)
    @Published var imageBrightness: Float = 0.0 {
        didSet {
            renderer?.imageBrightness = imageBrightness
        }
    }

    /// Mask fill opacity (0.0 - 1.0, affects interior fill)
    @Published var maskFillAlpha: Float = 0.5 {
        didSet {
            renderer?.maskFillAlpha = maskFillAlpha
        }
    }

    /// Mask edge opacity (0.0 - 1.0, affects edge/outline)
    @Published var maskEdgeAlpha: Float = 1.0 {
        didSet {
            renderer?.maskEdgeAlpha = maskEdgeAlpha
        }
    }

    /// Smooth kernel size for boundary smoothing (odd number, 7-31)
    /// Larger values smooth larger wavelength undulations
    @Published var smoothKernelSize: Int = 21

    // MARK: - Image Navigation

    @Published private(set) var currentImageIndex: Int = 0
    @Published private(set) var totalImageCount: Int = 0

    /// Filename of the currently displayed image (e.g. "image001.png"), nil if no image loaded
    var currentImageFileName: String? {
        imageManager.currentItem?.url.lastPathComponent
    }

    /// Set of all local image filenames (for checking which server images are already imported)
    var localImageFileNames: Set<String> {
        Set(imageManager.items.map { $0.url.lastPathComponent })
    }

    // MARK: - Loading/Saving State

    @Published private(set) var isLoading: Bool = false
    @Published private(set) var isSaving: Bool = false

    // MARK: - Undo Manager

    private let undoManager = AnnotationUndoManager()

    // MARK: - Image Manager

    private let imageManager = ImageItemManager()

    // MARK: - View State

    private var viewSize: CGSize = .zero
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Stroke Tracking

    private var strokePoints: [CGPoint] = []
    private var strokeBbox: CGRect = .null
    private var strokeStartPatch: Data?

    /// Flag to track if mask has been modified since last save
    private var maskModified: Bool = false

    /// Generation counter for stale load prevention during rapid navigation
    private var loadGeneration: Int = 0

    /// LRU cache for preloaded image + annotation data
    private let imageLoadCache = ImageLoadCache()

    /// Original bbox at stroke start (for proper patch expansion)
    private var originalStrokeBbox: CGRect = .null
    /// Original patch at stroke start (for proper patch expansion)
    private var originalStrokePatch: Data?

    // MARK: - QuickLine (hold to straighten)

    /// Timer for detecting stationary pen (QuickLine feature)
    private var quickLineTimer: Timer?
    /// Threshold time in seconds to trigger line straightening
    private let quickLineDelay: TimeInterval = 1.0
    /// Distance threshold to consider pen as stationary (in screen points)
    private let quickLineStationaryThreshold: CGFloat = 5.0
    /// Last position for stationary detection
    private var quickLineLastPoint: CGPoint?
    /// First point of stroke (for straight line)
    private var strokeStartPoint: CGPoint?

    // MARK: - Smooth Stroke Tracking

    /// Published smooth stroke points for overlay visualization
    @Published var smoothStrokePoints: [CGPoint] = []
    private var smoothStrokeBbox: CGRect = .null

    // MARK: - Initialization

    init() {
        print("🚀 CanvasViewModel init")
        setupRenderer()
        setupBindings()
        setupGestureCallbacks()
        loadClassNames()
        initializeProjectFolder()
        observeImportNotification()
    }

    /// Subscribe to import notifications from ImportCoordinator
    private func observeImportNotification() {
        NotificationCenter.default.publisher(for: .didImportImages)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.reloadImagesFromProject()
            }
            .store(in: &cancellables)
    }

    /// Initialize the project folder structure and load existing images
    private func initializeProjectFolder() {
        guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("[Project] Failed to get Documents directory")
            return
        }

        // Use Documents directly (no subfolder)
        print("[Project] Root: \(documentsURL.path)")

        do {
            try ProjectFileService.shared.initializeProject(at: documentsURL)
            print("[Project] Folder structure: images/, annotations/, labels/")
            ProjectFileService.shared.cleanInbox()
            reloadImagesFromProject()
        } catch {
            print("[Project] Failed to initialize: \(error)")
        }
    }

    /// Open a different project folder
    func openProject(at folderURL: URL) {
        let didStartAccessing = folderURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }

        // Clear cache on project switch
        Task { await imageLoadCache.clear() }

        do {
            try ProjectFileService.shared.initializeProject(at: folderURL)
            print("[Project] Opened project: \(folderURL.lastPathComponent)")
            reloadImagesFromProject()
        } catch {
            print("[Project] Failed to open: \(error)")
        }
    }

    /// Reload images from the project's images folder
    func reloadImagesFromProject() {
        let imageURLs = ProjectFileService.shared.getImageURLs()
        print("[Project] Found \(imageURLs.count) images")

        imageManager.setImages(imageURLs)

        // Resume from last viewed image if it still exists
        if let lastImageName = UserDefaults.standard.string(forKey: Self.lastImageNameKey),
           let index = imageManager.items.firstIndex(where: { $0.baseName == lastImageName }) {
            imageManager.goTo(index: index)
            print("[Project] Resuming from: \(lastImageName)")
        }

        // Load current image with its annotation (if exists)
        if imageManager.currentItem != nil {
            loadCurrentImage()
        }
    }

    /// Navigate to an existing image by filename. Returns true if found.
    func navigateToImage(named filename: String) -> Bool {
        if let index = imageManager.items.firstIndex(where: { $0.url.lastPathComponent == filename }) {
            imageManager.goTo(index: index)
            loadCurrentImage()
            return true
        }
        return false
    }

    /// Import a single image to the project
    func importImage(from sourceURL: URL) {
        do {
            let destinationURL = try ProjectFileService.shared.copyImageToProject(sourceURL)
            print("[Import] Copied: \(destinationURL.lastPathComponent)")

            // Refresh image list without loading any image (avoids
            // reloadImagesFromProject restoring the previous lastImageName)
            let imageURLs = ProjectFileService.shared.getImageURLs()
            imageManager.setImages(imageURLs)

            if let index = imageManager.items.firstIndex(where: { $0.url == destinationURL }) {
                imageManager.goTo(index: index)
                loadCurrentImage()
            }
        } catch {
            print("[Import] Failed: \(error)")
        }
    }

    /// Import multiple images to the project (batch)
    func importImages(from urls: [URL]) {
        var importedCount = 0
        for url in urls {
            do {
                _ = try ProjectFileService.shared.copyImageToProject(url)
                importedCount += 1
            } catch {
                print("[Import] Failed to copy \(url.lastPathComponent): \(error)")
            }
        }
        print("[Import] Batch imported \(importedCount)/\(urls.count) images")
        reloadImagesFromProject()
    }

    /// Import all images from a folder
    func importImagesFromFolder(_ folderURL: URL) {
        let fileManager = FileManager.default
        let supportedExtensions = ["png", "jpg", "jpeg"]

        let didStartAccessing = folderURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }

        do {
            let contents = try fileManager.contentsOfDirectory(
                at: folderURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )

            let imageFiles = contents.filter { supportedExtensions.contains($0.pathExtension.lowercased()) }
            print("[Import] Found \(imageFiles.count) images in folder")

            for imageURL in imageFiles {
                _ = try? ProjectFileService.shared.copyImageToProject(imageURL)
            }

            reloadImagesFromProject()
        } catch {
            print("[Import] Folder read failed: \(error)")
        }
    }

    /// Handle image providers from drag & drop
    func handleDroppedProviders(_ providers: [NSItemProvider]) {
        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier("public.image") {
                provider.loadFileRepresentation(forTypeIdentifier: "public.image") { [weak self] url, error in
                    guard let url = url else {
                        print("[Drop] Failed to load file: \(error?.localizedDescription ?? "unknown")")
                        return
                    }

                    // Copy to a temporary location (the provided URL is deleted after this callback)
                    let tempDir = FileManager.default.temporaryDirectory
                    let tempURL = tempDir.appendingPathComponent(url.lastPathComponent)
                    try? FileManager.default.removeItem(at: tempURL)
                    try? FileManager.default.copyItem(at: url, to: tempURL)

                    DispatchQueue.main.async {
                        guard let self = self else { return }
                        do {
                            _ = try ProjectFileService.shared.copyImageToProject(tempURL)
                            try? FileManager.default.removeItem(at: tempURL)
                            self.reloadImagesFromProject()
                            print("[Drop] Imported: \(url.lastPathComponent)")
                        } catch {
                            print("[Drop] Import failed: \(error)")
                        }
                    }
                }
            }
        }
    }

    private func setupRenderer() {
        renderer = MetalRenderer()
        // Set initial class ID (1 = red)
        renderer?.setCurrentClass(currentClassID)
    }

    private func setupGestureCallbacks() {
        // Drawing callbacks
        gestureCoordinator.onStrokeBegin = { [weak self] point in
            self?.beginStroke(at: point)
        }

        gestureCoordinator.onStrokeContinue = { [weak self] point in
            self?.continueStroke(to: point)
        }

        gestureCoordinator.onStrokeEnd = { [weak self] in
            self?.endStroke()
        }

        gestureCoordinator.onStrokeCancel = { [weak self] in
            self?.cancelStroke()
        }

        // Navigation callbacks
        gestureCoordinator.onPan = { [weak self] translation in
            self?.handlePan(translation: translation)
        }

        gestureCoordinator.onPinch = { [weak self] scale, center in
            self?.handlePinchDelta(scale: scale, at: center)
        }

        gestureCoordinator.onRotation = { [weak self] rotation, center in
            self?.handleRotationDelta(angle: rotation, at: center)
        }

        // Undo/Redo callbacks
        gestureCoordinator.onUndo = { [weak self] in
            self?.undo()
        }

        gestureCoordinator.onRedo = { [weak self] in
            self?.redo()
        }

        // Fill callback
        gestureCoordinator.onFillTap = { [weak self] point in
            self?.floodFill(at: point)
            // Auto-disable fill mode after fill
            self?.isFillMode = false
        }

        // Smooth stroke callbacks
        gestureCoordinator.onSmoothStrokeBegin = { [weak self] point in
            self?.beginSmoothStroke(at: point)
        }

        gestureCoordinator.onSmoothStrokeContinue = { [weak self] point in
            self?.continueSmoothStroke(to: point)
        }

        gestureCoordinator.onSmoothStrokeEnd = { [weak self] in
            self?.finishSmoothStroke()
        }
    }

    private func setupBindings() {
        // Sync image manager with published properties
        imageManager.$currentIndex
            .receive(on: DispatchQueue.main)
            .sink { [weak self] index in
                self?.currentImageIndex = index
            }
            .store(in: &cancellables)

        imageManager.$items
            .receive(on: DispatchQueue.main)
            .sink { [weak self] items in
                self?.totalImageCount = items.count
            }
            .store(in: &cancellables)

        // Forward undoManager changes so SwiftUI updates canUndo/canRedo
        undoManager.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &cancellables)
    }

    // MARK: - View Size

    func updateViewSize(_ size: CGSize) {
        viewSize = size
        renderer?.updateViewportSize(size)
    }

    // MARK: - Image Loading

    func loadImage(from url: URL) {
        do {
            try renderer?.loadImage(from: url)
        } catch {
            print("Failed to load image: \(error)")
        }
    }

    func loadImagesFromFolder(_ folderURL: URL) {
        let imagesURL = folderURL.appendingPathComponent("images")
        let annotationsURL = folderURL.appendingPathComponent("annotations")

        imageManager.loadImages(from: imagesURL)
        imageManager.checkAnnotations(in: annotationsURL)

        if let firstImage = imageManager.currentItem {
            loadImage(from: firstImage.url)
        }
    }

    // MARK: - Image Navigation

    func previousImage() {
        saveAndNavigate { [weak self] in
            self?.imageManager.previous()
        }
    }

    func nextImage() {
        saveAndNavigate { [weak self] in
            self?.imageManager.next()
        }
    }

    func goToImage(index: Int) {
        guard index != currentImageIndex else { return }
        saveAndNavigate { [weak self] in
            self?.imageManager.goTo(index: index)
        }
    }

    /// Save current annotation in background, navigate immediately
    private func saveAndNavigate(navigation: @escaping () -> Void) {
        // Capture data needed for background save before navigation
        let shouldSave = maskModified
        let imageItem = imageManager.currentItem
        let textureManager = renderer?.textureManager
        let maskData: [UInt8]? = shouldSave ? textureManager?.readMask() : nil
        let maskWidth = textureManager.map { Int($0.maskSize.width) } ?? 0
        let maskHeight = textureManager.map { Int($0.maskSize.height) } ?? 0
        let color = annotationColor

        // Navigate immediately (no waiting for save)
        navigation()
        loadCurrentImage()

        // Save in background if needed
        guard shouldSave,
              let imageURL = imageItem?.url,
              let data = maskData,
              data.contains(where: { $0 != 0 }) else {
            return
        }

        // Invalidate cache for saved image (annotation changed on disk)
        Task { await imageLoadCache.invalidate(for: imageURL) }

        // Reset modified flag and show saving indicator
        maskModified = false
        isSaving = true

        // PNG generation and file write on background thread
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let pngData = self?.createColoredPNG(
                from: data,
                width: maskWidth,
                height: maskHeight,
                color: color
            ) else {
                DispatchQueue.main.async {
                    self?.isSaving = false
                }
                return
            }

            // Save to file
            do {
                try ProjectFileService.shared.saveAnnotation(pngData, for: imageURL)
                print("[Save] Saved annotation (background)")
            } catch {
                print("[Save] Failed: \(error)")
            }

            DispatchQueue.main.async {
                self?.isSaving = false
            }
        }
    }

    private func loadCurrentImage() {
        guard let item = imageManager.currentItem else { return }

        // Phase A: immediate main-thread setup
        isLoading = true
        maskModified = false
        undoManager.clear()

        loadGeneration += 1
        let generation = loadGeneration
        let imageURL = item.url
        let annotationURL: URL? = {
            if let url = ProjectFileService.shared.getAnnotationURL(for: imageURL),
               FileManager.default.fileExists(atPath: url.path) {
                return url
            }
            return nil
        }()

        // Phase B: heavy work on background thread
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }

            // Check cache first
            let cached = await self.imageLoadCache.get(for: imageURL)

            let imageData: TextureManager.PreloadedImageData
            let maskData: [UInt8]?

            if let cached = cached {
                imageData = cached.imageData
                maskData = cached.maskData
            } else {
                // Cache miss — decode from disk
                guard let decoded = try? TextureManager.prepareImageData(from: imageURL) else {
                    await MainActor.run { self.isLoading = false }
                    return
                }
                imageData = decoded

                // Parse annotation if exists
                if let annotationURL = annotationURL {
                    maskData = CanvasViewModel.parseAnnotationInBackground(
                        annotationURL: annotationURL,
                        targetMaskWidth: decoded.maskWidth,
                        targetMaskHeight: decoded.maskHeight
                    )
                } else {
                    maskData = nil
                }

                // Store in cache
                await self.imageLoadCache.put(
                    url: imageURL,
                    entry: ImageLoadCache.Entry(imageData: imageData, maskData: maskData)
                )
            }

            // Phase C: apply to GPU on main thread
            await MainActor.run {
                // Generation check — discard stale loads from rapid navigation
                guard self.loadGeneration == generation else { return }

                do {
                    try self.renderer?.applyPreloadedImage(imageData)

                    if let maskData = maskData {
                        try self.renderer?.textureManager.uploadMask(maskData)
                        print("[Load] Loaded annotation (\(imageData.maskWidth)x\(imageData.maskHeight))")
                    }
                } catch {
                    print("[Load] Failed: \(error)")
                }

                self.resetView()
                UserDefaults.standard.set(item.baseName, forKey: Self.lastImageNameKey)
                self.isLoading = false

                // Prefetch adjacent images
                self.prefetchAdjacentImages()
            }
        }
    }

    /// Prefetch previous and next images into cache on a low-priority background thread
    private func prefetchAdjacentImages() {
        let items = imageManager.items
        let currentIndex = imageManager.currentIndex
        guard items.count > 1 else { return }

        var adjacentURLs: [(URL, URL?)] = []
        if currentIndex + 1 < items.count {
            let url = items[currentIndex + 1].url
            let annURL = ProjectFileService.shared.getAnnotationURL(for: url)
            adjacentURLs.append((url, annURL))
        }
        if currentIndex > 0 {
            let url = items[currentIndex - 1].url
            let annURL = ProjectFileService.shared.getAnnotationURL(for: url)
            adjacentURLs.append((url, annURL))
        }

        Task.detached(priority: .utility) { [weak self] in
            guard let self = self else { return }
            for (url, annURL) in adjacentURLs {
                // Skip if already cached
                if await self.imageLoadCache.get(for: url) != nil { continue }

                guard let decoded = try? TextureManager.prepareImageData(from: url) else { continue }

                var maskData: [UInt8]? = nil
                if let annURL = annURL,
                   FileManager.default.fileExists(atPath: annURL.path) {
                    maskData = CanvasViewModel.parseAnnotationInBackground(
                        annotationURL: annURL,
                        targetMaskWidth: decoded.maskWidth,
                        targetMaskHeight: decoded.maskHeight
                    )
                }

                await self.imageLoadCache.put(
                    url: url,
                    entry: ImageLoadCache.Entry(imageData: decoded, maskData: maskData)
                )
            }
        }
    }

    // MARK: - Drawing

    /// Brush preview size for UI (in screen coordinates, reflects actual drawn size)
    var brushPreviewSize: CGFloat {
        CGFloat(brushRadius) * 2 * currentScale
    }

    /// Counter for throttling bbox expansion checks
    private var strokePointCounter: Int = 0

    func beginStroke(at point: CGPoint) {
        isDrawing = true
        lastDrawPoint = point
        strokePoints = [point]
        strokePointCounter = 0

        // QuickLine: save start point and initialize
        strokeStartPoint = point
        quickLineLastPoint = point
        quickLineTimer?.invalidate()
        quickLineTimer = nil

        // Convert touch point to screen pixels, then to mask coordinates
        let screenPoint = renderer?.convertTouchToScreen(point) ?? point
        let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? point

        // Guard against invalid coordinates (NaN or infinite from degenerate transforms)
        guard maskPoint.x.isFinite && maskPoint.y.isFinite else {
            print("[Stroke] Invalid maskPoint at begin: \(maskPoint), skipping stroke setup")
            isDrawing = false
            return
        }

        // Ensure mask texture exists before capturing patch
        _ = try? renderer?.textureManager.getMaskTexture()

        // Use very large initial bbox to minimize expensive expandStrokePatch calls
        // For most strokes, this single upfront allocation is much faster than repeated expansions
        if let textureManager = renderer?.textureManager {
            let maxWidth = textureManager.maskSize.width
            let maxHeight = textureManager.maskSize.height

            // Use minimum 2000x2000 or full texture (whichever is smaller)
            // This covers most strokes without needing expansion
            let initialSize: CGFloat = min(2000, min(maxWidth, maxHeight))
            let halfSize = initialSize / 2

            strokeBbox = CGRect(
                x: maskPoint.x - halfSize,
                y: maskPoint.y - halfSize,
                width: initialSize,
                height: initialSize
            )

            // Clamp to texture bounds
            strokeBbox = strokeBbox.intersection(CGRect(x: 0, y: 0, width: maxWidth, height: maxHeight))
        }

        // Capture patch before stroke (larger area upfront)
        strokeStartPatch = renderer?.textureManager.readMaskRegion(bbox: strokeBbox)

        // Store original bbox and patch for proper expansion
        originalStrokeBbox = strokeBbox
        originalStrokePatch = strokeStartPatch

        // Apply first stamp
        renderer?.applyStamp(at: point, radius: brushRadius, isPainting: isPainting)
    }

    func continueStroke(to point: CGPoint) {
        guard isDrawing else { return }

        lastDrawPoint = point

        // Interpolate points for smooth continuous strokes
        if let lastPoint = strokePoints.last {
            let distance = hypot(point.x - lastPoint.x, point.y - lastPoint.y)
            // Use smaller step interval (30% of brush radius) for overlapping stamps
            // This ensures continuous lines even when pen moves quickly
            let stepInterval = max(1.0, CGFloat(brushRadius) * 0.3)
            let steps = max(1, Int(ceil(distance / stepInterval)))

            // Collect all interpolated points for batch processing
            var interpolatedPoints: [CGPoint] = []

            // Track if we need to expand bbox (throttled check)
            var needsBboxExpansion = false
            var expandedBbox = strokeBbox

            for i in 1...steps {
                let t = CGFloat(i) / CGFloat(steps)
                let interpolatedPoint = CGPoint(
                    x: lastPoint.x + (point.x - lastPoint.x) * t,
                    y: lastPoint.y + (point.y - lastPoint.y) * t
                )

                interpolatedPoints.append(interpolatedPoint)
                strokePointCounter += 1

                // Only check bbox expansion every 20 points to reduce overhead
                // The large initial bbox should cover most cases
                if strokePointCounter % 20 == 0 {
                    let screenPoint = renderer?.convertTouchToScreen(interpolatedPoint) ?? interpolatedPoint
                    let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? interpolatedPoint

                    // Skip if coordinates are invalid (NaN or infinite)
                    guard maskPoint.x.isFinite && maskPoint.y.isFinite else { continue }

                    let maskScaleFactor = renderer?.canvasTransform.maskScaleFactor ?? 2.0
                    // Radius in mask coordinates (UI "1" = 1 original image pixel)
                    let radius = CGFloat(brushRadius * maskScaleFactor) + 1.0
                    let stampRect = CGRect(
                        x: maskPoint.x - radius,
                        y: maskPoint.y - radius,
                        width: radius * 2,
                        height: radius * 2
                    )

                    if !expandedBbox.isNull {
                        let newBbox = expandedBbox.union(stampRect)
                        if newBbox != expandedBbox {
                            expandedBbox = newBbox
                            needsBboxExpansion = true
                        }
                    }
                }
            }

            // Apply all stamps in a single GPU batch (this is fast)
            renderer?.applyStamps(at: interpolatedPoints, radius: brushRadius, isPainting: isPainting)

            // Track expanded bbox for later (defer GPU read to endStroke to avoid freeze)
            if needsBboxExpansion && expandedBbox != strokeBbox {
                strokeBbox = expandedBbox
            }

            // Only store the final point (not all interpolated points)
            strokePoints.append(point)

            // QuickLine: detect stationary pen
            if let lastPoint = quickLineLastPoint {
                let distance = hypot(point.x - lastPoint.x, point.y - lastPoint.y)
                if distance < quickLineStationaryThreshold {
                    // Pen is stationary - start timer if not already running
                    if quickLineTimer == nil {
                        quickLineTimer = Timer.scheduledTimer(withTimeInterval: quickLineDelay, repeats: false) { [weak self] _ in
                            self?.snapToStraightLine()
                        }
                    }
                } else {
                    // Pen moved - cancel timer and update last point
                    quickLineTimer?.invalidate()
                    quickLineTimer = nil
                    quickLineLastPoint = point
                }
            }
        }
    }

    func endStroke() {
        guard isDrawing else { return }

        isDrawing = false

        // Cancel QuickLine timer
        quickLineTimer?.invalidate()
        quickLineTimer = nil

        // If bbox was expanded during stroke, read the full patch now (deferred from continueStroke)
        if strokeBbox != originalStrokeBbox && !strokeBbox.isNull {
            strokeStartPatch = renderer?.textureManager.readMaskRegion(bbox: strokeBbox)
        }

        // Create undo action
        if let patch = strokeStartPatch, !strokeBbox.isNull {
            let action = UndoAction(
                classID: currentClassID,
                bbox: strokeBbox,
                previousPatch: patch
            )
            undoManager.pushUndo(action)
            maskModified = true
        }

        // Reset stroke state
        strokePoints.removeAll()
        strokeBbox = .null
        strokeStartPatch = nil
        originalStrokeBbox = .null
        originalStrokePatch = nil
        strokeStartPoint = nil
        quickLineLastPoint = nil

        // Note: Auto-save removed for performance
        // Saving happens on image navigation or app background
    }

    /// QuickLine: Replace freehand stroke with a straight line
    private func snapToStraightLine() {
        guard isDrawing,
              let startPoint = strokeStartPoint,
              let endPoint = strokePoints.last,
              let originalPatch = originalStrokePatch,
              !originalStrokeBbox.isNull else {
            return
        }

        // Cancel timer
        quickLineTimer?.invalidate()
        quickLineTimer = nil

        // Restore original state (undo freehand drawing)
        renderer?.textureManager.writeMaskRegion(bbox: originalStrokeBbox, data: originalPatch)

        // Calculate straight line points
        let distance = hypot(endPoint.x - startPoint.x, endPoint.y - startPoint.y)
        let stepInterval = max(1.0, CGFloat(brushRadius) * 0.3)
        let steps = max(1, Int(ceil(distance / stepInterval)))

        var linePoints: [CGPoint] = []
        for i in 0...steps {
            let t = CGFloat(i) / CGFloat(steps)
            let x = startPoint.x + (endPoint.x - startPoint.x) * t
            let y = startPoint.y + (endPoint.y - startPoint.y) * t
            linePoints.append(CGPoint(x: x, y: y))
        }

        // Draw straight line
        renderer?.applyStamps(at: linePoints, radius: brushRadius, isPainting: isPainting)

        // Update stroke points to reflect the straight line
        strokePoints = [startPoint, endPoint]

        print("[QuickLine] Snapped to straight line: \(Int(distance))px")
    }

    private func expandStrokePatch(to newBbox: CGRect) {
        // We need to properly composite: the previous correct patch + new regions
        // The texture now has painted data in the previous region, so we can't just re-read
        guard let previousPatch = originalStrokePatch,
              !originalStrokeBbox.isNull,
              let textureManager = renderer?.textureManager,
              let texture = textureManager.maskTexture else { return }

        // Guard against invalid bbox values (NaN or infinite)
        guard newBbox.minX.isFinite && newBbox.minY.isFinite &&
              newBbox.maxX.isFinite && newBbox.maxY.isFinite else {
            print("[Stroke] Invalid bbox values, skipping expansion")
            return
        }

        // Calculate integer bounds for new bbox
        // Use floor for min and ceil for max to ensure we capture all affected pixels
        let newMinX = max(0, Int(floor(newBbox.minX)))
        let newMinY = max(0, Int(floor(newBbox.minY)))
        let newMaxX = min(texture.width, Int(ceil(newBbox.maxX)))
        let newMaxY = min(texture.height, Int(ceil(newBbox.maxY)))
        let newWidth = newMaxX - newMinX
        let newHeight = newMaxY - newMinY

        guard newWidth > 0 && newHeight > 0 else { return }

        // Read the new bbox from texture (has painted data in previous region, original data in new regions)
        var newPatchData = [UInt8](repeating: 0, count: newWidth * newHeight)
        texture.getBytes(
            &newPatchData,
            bytesPerRow: newWidth,
            from: MTLRegion(
                origin: MTLOrigin(x: newMinX, y: newMinY, z: 0),
                size: MTLSize(width: newWidth, height: newHeight, depth: 1)
            ),
            mipmapLevel: 0
        )

        // Calculate integer bounds for previous bbox
        // Guard against invalid values
        guard originalStrokeBbox.minX.isFinite && originalStrokeBbox.minY.isFinite &&
              originalStrokeBbox.maxX.isFinite && originalStrokeBbox.maxY.isFinite else {
            print("[Stroke] Invalid originalStrokeBbox values, skipping expansion")
            return
        }
        let prevMinX = max(0, Int(originalStrokeBbox.minX))
        let prevMinY = max(0, Int(originalStrokeBbox.minY))
        let prevMaxX = min(texture.width, Int(originalStrokeBbox.maxX))
        let prevMaxY = min(texture.height, Int(originalStrokeBbox.maxY))
        let prevWidth = prevMaxX - prevMinX
        let prevHeight = prevMaxY - prevMinY

        // Calculate offset of previous region within new region
        let offsetX = prevMinX - newMinX
        let offsetY = prevMinY - newMinY

        // Copy previous patch data into the new patch at the correct offset
        // This restores the "before stroke" state for the previously covered region
        // Use row-by-row memcpy for performance (much faster than pixel-by-pixel)
        previousPatch.withUnsafeBytes { prevBuffer in
            guard let prevPtr = prevBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }

            for y in 0..<prevHeight {
                let prevRowStart = y * prevWidth
                let newRowStart = (y + offsetY) * newWidth + offsetX

                // Bounds check for safety
                guard prevRowStart + prevWidth <= previousPatch.count,
                      newRowStart + prevWidth <= newPatchData.count else { continue }

                // Copy entire row at once using memcpy
                memcpy(&newPatchData[newRowStart], prevPtr.advanced(by: prevRowStart), prevWidth)
            }
        }

        let compositedPatch = Data(newPatchData)
        strokeStartPatch = compositedPatch

        // Update original tracking to the composited result
        // So next expansion uses the correct "before" data for the entire area so far
        originalStrokeBbox = CGRect(x: newMinX, y: newMinY, width: newWidth, height: newHeight)
        originalStrokePatch = compositedPatch
    }

    // MARK: - Stroke Cancellation

    /// Cancel current stroke and restore previous state (called when 2+ fingers detected)
    func cancelStroke() {
        guard isDrawing else { return }

        isDrawing = false

        // Restore the mask to pre-stroke state
        if let patch = strokeStartPatch, !strokeBbox.isNull {
            renderer?.textureManager.writeMaskRegion(bbox: strokeBbox, data: patch)
        }

        // Reset stroke state
        strokePoints.removeAll()
        strokeBbox = .null
        strokeStartPatch = nil
        originalStrokeBbox = .null
        originalStrokePatch = nil

        print("🚫 Stroke cancelled - restored previous state")
    }

    // MARK: - Navigation Gesture Handling (UIKit)

    /// Handle pan gesture (2-finger drag)
    func handlePan(translation: CGPoint) {
        // Convert translation from points to pixels (same as touch coordinates)
        let scaledTranslation = renderer?.convertTouchToScreen(translation) ?? translation
        renderer?.canvasTransform.applyPan(delta: scaledTranslation)
    }

    /// Handle pinch gesture with delta scale and center point
    func handlePinchDelta(scale: CGFloat, at center: CGPoint) {
        let screenCenter = renderer?.convertTouchToScreen(center) ?? center
        renderer?.canvasTransform.applyPinch(scaleFactor: scale, center: screenCenter)
        // Update published scale for UI
        currentScale = renderer?.canvasTransform.scale ?? 1.0
    }

    /// Handle rotation gesture with delta angle and center point
    func handleRotationDelta(angle: CGFloat, at center: CGPoint) {
        let screenCenter = renderer?.convertTouchToScreen(center) ?? center
        renderer?.canvasTransform.applyRotation(angleDelta: angle, center: screenCenter)
    }

    /// Fit image to view (aspect fit, centered, original orientation)
    func resetView() {
        guard let renderer = renderer else { return }

        // Get image size and viewport size (both in pixels for Metal)
        let imageSize = renderer.textureManager.imageSize
        let viewportSize = renderer.viewportSize

        // Fit image to viewport
        renderer.canvasTransform.fitToView(imageSize: imageSize, viewSize: viewportSize)

        // Update published scale for UI
        currentScale = renderer.canvasTransform.scale
        print("[View] Fit to view: scale=\(String(format: "%.2f", currentScale))")
    }

    // MARK: - Undo/Redo

    var canUndo: Bool { undoManager.canUndo }
    var canRedo: Bool { undoManager.canRedo }

    func undo() {
        guard let action = undoManager.undo() else { return }

        // Capture current state before restoring (for redo)
        if let currentPatch = renderer?.textureManager.readMaskRegion(bbox: action.bbox) {
            undoManager.setNewPatchOnLastRedo(currentPatch)
        }

        // Restore previous patch
        renderer?.textureManager.writeMaskRegion(bbox: action.bbox, data: action.previousPatch)
        maskModified = true
    }

    func redo() {
        guard let action = undoManager.redo() else { return }
        guard let newPatch = action.newPatch else { return }

        // Write the post-stroke state
        renderer?.textureManager.writeMaskRegion(bbox: action.bbox, data: newPatch)
        maskModified = true
    }

    /// Clear all annotations for current image (undoable with 2-finger tap)
    func clearAllAnnotations() {
        guard let textureManager = renderer?.textureManager else {
            print("[Clear] No texture manager")
            return
        }

        // Read current mask data for undo
        guard let currentMaskData = textureManager.readMask() else {
            print("[Clear] No mask data to backup")
            return
        }

        // Check if there's anything to clear
        let hasData = currentMaskData.contains { $0 != 0 }
        guard hasData else {
            print("[Clear] Mask is already empty")
            return
        }

        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        // Create undo action with full mask
        let bbox = CGRect(x: 0, y: 0, width: CGFloat(maskWidth), height: CGFloat(maskHeight))
        let previousPatch = Data(currentMaskData)

        let action = UndoAction(
            classID: 0,  // 0 indicates clear action
            bbox: bbox,
            previousPatch: previousPatch
        )
        undoManager.pushUndo(action)

        // Clear the mask
        renderer?.clearMask()
        maskModified = true
        print("[Clear] Cleared all annotations (undoable)")
    }

    // MARK: - Save

    /// Save current annotation (called on image navigation and app background)
    func saveBeforeBackground() {
        saveCurrentAnnotation()
        print("[Save] Background save completed")
    }

    private func saveCurrentAnnotation() {
        guard let imageItem = imageManager.currentItem,
              let textureManager = renderer?.textureManager else {
            print("[Save] No image or texture manager")
            return
        }

        // Read mask data from GPU
        guard let maskData = textureManager.readMask() else {
            print("[Save] No mask data to save")
            return
        }

        // Check if mask has any data (skip save if empty)
        let hasData = maskData.contains { $0 != 0 }
        guard hasData else {
            print("[Save] Mask is empty, skipping save")
            return
        }

        // Convert binary mask to color PNG
        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        guard let pngData = createColoredPNG(
            from: maskData,
            width: maskWidth,
            height: maskHeight,
            color: annotationColor
        ) else {
            print("[Save] Failed to create PNG")
            return
        }

        // Save to annotations folder
        do {
            try ProjectFileService.shared.saveAnnotation(pngData, for: imageItem.url)
            print("[Save] Saved annotation for \(imageItem.baseName)")
        } catch {
            print("[Save] Failed: \(error)")
        }
    }

    private func loadAnnotation(from url: URL) {
        guard let textureManager = renderer?.textureManager,
              let data = try? Data(contentsOf: url),
              let image = UIImage(data: data),
              let cgImage = image.cgImage else {
            print("[Load] Failed to load annotation image")
            return
        }

        let width = cgImage.width
        let height = cgImage.height

        // Read pixel data from PNG
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
            print("[Load] Failed to create context")
            return
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert colored PNG to class ID mask
        // Each pixel color is snapped to nearest class color
        var maskData = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            let offset = i * bytesPerPixel
            let r = pixelData[offset]
            let g = pixelData[offset + 1]
            let b = pixelData[offset + 2]
            let a = pixelData[offset + 3]

            // If transparent or white, it's background (classID = 0)
            let isBackground = a < 128 || (r > 250 && g > 250 && b > 250)
            if isBackground {
                maskData[i] = 0
            } else {
                // Find nearest class color
                maskData[i] = UInt8(findNearestClassID(r: r, g: g, b: b))
            }
        }

        // Check if we need to resize (annotation size might differ from mask size)
        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        if width == maskWidth && height == maskHeight {
            // Same size, upload directly
            do {
                try textureManager.uploadMask(maskData)
                print("[Load] Loaded annotation (\(width)x\(height))")
            } catch {
                print("[Load] Upload failed: \(error)")
            }
        } else {
            // Need to resize - use nearest neighbor for binary mask
            let resizedMask = resizeMask(
                maskData,
                fromWidth: width, fromHeight: height,
                toWidth: maskWidth, toHeight: maskHeight
            )
            do {
                try textureManager.uploadMask(resizedMask)
                print("[Load] Loaded and resized annotation (\(width)x\(height) → \(maskWidth)x\(maskHeight))")
            } catch {
                print("[Load] Upload failed: \(error)")
            }
        }
    }

    /// Create a multi-class colored PNG from mask data
    /// Mask values: 0=background(white), 1-8=class colors
    private func createColoredPNG(from maskData: [UInt8], width: Int, height: Int, color: Color) -> Data? {
        // Create RGBA pixel data
        let bytesPerPixel = 4
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        for i in 0..<(width * height) {
            let offset = i * bytesPerPixel
            let classID = Int(maskData[i])

            if classID > 0 && classID <= classRGBColors.count {
                // Masked pixel: use corresponding class color
                let (r, g, b) = classRGBColors[classID - 1]
                pixelData[offset] = r
                pixelData[offset + 1] = g
                pixelData[offset + 2] = b
                pixelData[offset + 3] = 255
            } else {
                // Background: white with full alpha
                pixelData[offset] = 255
                pixelData[offset + 1] = 255
                pixelData[offset + 2] = 255
                pixelData[offset + 3] = 255
            }
        }

        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * bytesPerPixel,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
              let cgImage = context.makeImage() else {
            return nil
        }

        // Convert to PNG data
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.pngData()
    }

    /// Find the nearest class ID for a given RGB color
    /// Returns 1-8 for matching class colors, or 1 as fallback
    private func findNearestClassID(r: UInt8, g: UInt8, b: UInt8) -> Int {
        Self.findNearestClassIDStatic(r: r, g: g, b: b)
    }

    /// Thread-safe static version: O(1) exact match, fallback to nearest neighbor
    private nonisolated static func findNearestClassIDStatic(r: UInt8, g: UInt8, b: UInt8) -> Int {
        // O(1) exact match (covers the vast majority of pixels)
        let key = UInt32(r) << 16 | UInt32(g) << 8 | UInt32(b)
        if let classID = exactColorLookup[key] {
            return Int(classID)
        }

        // Fallback: nearest neighbor for anti-aliased pixels
        var minDistance = Int.max
        var nearestClassID = 1
        for (index, (cr, cg, cb)) in classRGBColors.enumerated() {
            let distance = abs(Int(r) - Int(cr)) + abs(Int(g) - Int(cg)) + abs(Int(b) - Int(cb))
            if distance < minDistance {
                minDistance = distance
                nearestClassID = index + 1
            }
        }
        return nearestClassID
    }

    /// Resize binary mask using nearest neighbor interpolation
    private func resizeMask(_ source: [UInt8], fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        Self.resizeMaskStatic(source, fromWidth: fromWidth, fromHeight: fromHeight, toWidth: toWidth, toHeight: toHeight)
    }

    /// Thread-safe static resize with nearest neighbor interpolation
    private nonisolated static func resizeMaskStatic(_ source: [UInt8], fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: toWidth * toHeight)

        let scaleX = Float(fromWidth) / Float(toWidth)
        let scaleY = Float(fromHeight) / Float(toHeight)

        for y in 0..<toHeight {
            let srcY = Int(Float(y) * scaleY)
            for x in 0..<toWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcIndex = srcY * fromWidth + srcX
                let dstIndex = y * toWidth + x

                if srcIndex < source.count {
                    result[dstIndex] = source[srcIndex]
                }
            }
        }

        return result
    }

    /// Parse annotation PNG to class ID mask on a background thread.
    /// All operations (file I/O, decode, pixel parsing) are thread-safe.
    nonisolated static func parseAnnotationInBackground(
        annotationURL: URL,
        targetMaskWidth: Int,
        targetMaskHeight: Int
    ) -> [UInt8]? {
        guard let data = try? Data(contentsOf: annotationURL),
              let image = UIImage(data: data),
              let cgImage = image.cgImage else {
            return nil
        }

        let width = cgImage.width
        let height = cgImage.height
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

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert colored PNG to class ID mask with pointer-based access
        let pixelCount = width * height
        var maskData = [UInt8](repeating: 0, count: pixelCount)

        pixelData.withUnsafeBufferPointer { pixelBuf in
            maskData.withUnsafeMutableBufferPointer { maskBuf in
                guard let pixelPtr = pixelBuf.baseAddress,
                      let maskPtr = maskBuf.baseAddress else { return }

                for i in 0..<pixelCount {
                    let offset = i * bytesPerPixel
                    let r = pixelPtr[offset]
                    let g = pixelPtr[offset + 1]
                    let b = pixelPtr[offset + 2]
                    let a = pixelPtr[offset + 3]

                    // Background check: transparent or white
                    if a < 128 || (r > 250 && g > 250 && b > 250) {
                        maskPtr[i] = 0
                    } else {
                        maskPtr[i] = UInt8(findNearestClassIDStatic(r: r, g: g, b: b))
                    }
                }
            }
        }

        // Resize if needed
        if width == targetMaskWidth && height == targetMaskHeight {
            return maskData
        } else {
            return resizeMaskStatic(
                maskData,
                fromWidth: width, fromHeight: height,
                toWidth: targetMaskWidth, toHeight: targetMaskHeight
            )
        }
    }

    // MARK: - Flood Fill

    /// Perform flood fill at the given touch location
    func floodFill(at touchPoint: CGPoint) {
        guard let renderer = renderer else {
            print("[FloodFill] No renderer or texture manager")
            return
        }
        let textureManager = renderer.textureManager

        // Convert touch point to screen pixels, then to mask coordinates
        let screenPoint = renderer.convertTouchToScreen(touchPoint)
        let maskPoint = renderer.canvasTransform.screenToMask(screenPoint)

        // Guard against invalid coordinates (NaN or infinite from degenerate transforms)
        guard maskPoint.x.isFinite && maskPoint.y.isFinite else {
            print("[FloodFill] Invalid maskPoint: \(maskPoint), skipping fill")
            return
        }

        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        // Ensure the point is within bounds
        let startX = Int(maskPoint.x)
        let startY = Int(maskPoint.y)

        guard startX >= 0 && startX < maskWidth && startY >= 0 && startY < maskHeight else {
            print("[FloodFill] Point out of bounds: (\(startX), \(startY))")
            return
        }

        // Get current mask data
        guard var maskData = textureManager.readMask() else {
            // Create new mask if it doesn't exist
            _ = try? textureManager.getMaskTexture()
            guard var newMaskData = textureManager.readMask() else {
                print("[FloodFill] Failed to get mask data")
                return
            }
            performFloodFill(on: &newMaskData, width: maskWidth, height: maskHeight, startX: startX, startY: startY)
            return
        }

        // Check what's at the starting point
        let startIndex = startY * maskWidth + startX
        let targetValue = maskData[startIndex]

        // If tapping on the same class as currently selected, do nothing
        if targetValue == UInt8(currentClassID) {
            print("[FloodFill] Same class selected, skipping")
            return
        }

        // Capture undo patch before flood fill
        let bbox = CGRect(x: 0, y: 0, width: CGFloat(maskWidth), height: CGFloat(maskHeight))
        let previousPatch = Data(maskData)

        // Perform flood fill
        performFloodFill(on: &maskData, width: maskWidth, height: maskHeight, startX: startX, startY: startY)

        // Upload the modified mask
        do {
            try textureManager.uploadMask(maskData)
            print("[FloodFill] Fill completed")

            // Create undo action
            let action = UndoAction(
                classID: currentClassID,
                bbox: bbox,
                previousPatch: previousPatch
            )
            undoManager.pushUndo(action)
            maskModified = true
        } catch {
            print("[FloodFill] Failed to upload mask: \(error)")
        }
    }

    /// Flood fill algorithm using BFS (Breadth-First Search)
    /// Replaces contiguous region of targetValue with currentClassID (1-8)
    /// Works for both empty regions (0) and existing class regions (1-8)
    private func performFloodFill(on maskData: inout [UInt8], width: Int, height: Int, startX: Int, startY: Int) {
        let startIndex = startY * width + startX
        let targetValue = maskData[startIndex]
        let fillValue = UInt8(currentClassID)

        // If target is same as fill value, nothing to do
        if targetValue == fillValue {
            return
        }

        // BFS queue
        var queue: [(Int, Int)] = [(startX, startY)]
        var visited = Set<Int>()
        visited.insert(startIndex)

        // Direction offsets: up, down, left, right
        let dx = [0, 0, -1, 1]
        let dy = [-1, 1, 0, 0]

        var fillCount = 0

        while !queue.isEmpty {
            let (x, y) = queue.removeFirst()
            let index = y * width + x

            // Fill this pixel with current class ID
            maskData[index] = fillValue
            fillCount += 1

            // Check all 4 neighbors
            for i in 0..<4 {
                let nx = x + dx[i]
                let ny = y + dy[i]

                // Bounds check
                guard nx >= 0 && nx < width && ny >= 0 && ny < height else { continue }

                let neighborIndex = ny * width + nx

                // Skip if already visited
                guard !visited.contains(neighborIndex) else { continue }

                // Fill if same value as target (contiguous region)
                if maskData[neighborIndex] == targetValue {
                    visited.insert(neighborIndex)
                    queue.append((nx, ny))
                }
            }
        }

        let action = targetValue == 0 ? "Filled" : "Replaced class \(targetValue) with"
        print("[FloodFill] \(action) \(fillCount) pixels → class \(currentClassID)")
    }

    // MARK: - Class Names Persistence

    /// Save class names to UserDefaults
    private func saveClassNames() {
        UserDefaults.standard.set(classNames, forKey: Self.classNamesKey)
        print("[ClassNames] Saved: \(classNames.filter { !$0.isEmpty })")
    }

    /// Load class names from UserDefaults
    private func loadClassNames() {
        if let saved = UserDefaults.standard.stringArray(forKey: Self.classNamesKey) {
            // Ensure we always have exactly 8 elements
            if saved.count == 8 {
                classNames = saved
            } else {
                // Pad or truncate to 8 elements
                var adjusted = saved
                while adjusted.count < 8 { adjusted.append("") }
                classNames = Array(adjusted.prefix(8))
            }
            print("[ClassNames] Loaded: \(classNames.filter { !$0.isEmpty })")
        }
    }

    /// Clear all class names
    func clearClassNames() {
        classNames = Array(repeating: "", count: 8)
        print("[ClassNames] Cleared")
    }

    /// Delete all project files (images, annotations, labels) and reset canvas state
    func deleteAllFiles() {
        // Save nothing — we're deleting everything
        maskModified = false

        // Delete all files from disk
        let count = ProjectFileService.shared.deleteAllProjectFiles()

        // Clear in-memory state (image + mask textures)
        renderer?.textureManager.clear()
        undoManager.clear()
        imageManager.setImages([])
        currentImageIndex = 0
        totalImageCount = 0

        // Remove last-image bookmark so next launch starts fresh
        UserDefaults.standard.removeObject(forKey: Self.lastImageNameKey)

        print("[Delete] All project files removed (\(count) files), canvas reset")
    }

    /// Delete the current image and its annotation from the project
    func deleteCurrentImage() {
        guard let currentItem = imageManager.currentItem else { return }

        // Don't save the mask for the image we're about to delete
        maskModified = false

        // Invalidate cache for deleted image
        Task { await imageLoadCache.invalidate(for: currentItem.url) }

        // Delete from disk
        let deleted = ProjectFileService.shared.deleteImage(at: currentItem.url)
        guard deleted else { return }

        print("[Delete] Removed image: \(currentItem.baseName)")

        // Clear undo history (no longer relevant)
        undoManager.clear()

        // Reload image list and navigate
        let imageURLs = ProjectFileService.shared.getImageURLs()
        imageManager.setImages(imageURLs)

        if imageManager.currentItem != nil {
            loadCurrentImage()
        } else {
            // No images left — clear canvas completely (image + mask)
            renderer?.textureManager.clear()
            UserDefaults.standard.removeObject(forKey: Self.lastImageNameKey)
        }
    }

    /// Toggle visibility of a specific class's annotations
    func toggleClassVisibility(_ classID: Int) {
        if hiddenClassIDs.contains(classID) {
            hiddenClassIDs.remove(classID)
        } else {
            hiddenClassIDs.insert(classID)
        }
    }

    // MARK: - U-Net Integration

    /// Run U-Net ensemble prediction (one-shot, no prompt needed)
    func runUNetPrediction() {
        Task { @MainActor in
            do {
                // Load models if not ready
                if !unetService.isReady {
                    isUNetLoading = true
                    print("[UNet] Loading models...")
                    try await unetService.loadModels()
                    isUNetLoading = false
                }

                // Get current image
                guard let imageItem = imageManager.currentItem,
                      let imageData = try? Data(contentsOf: imageItem.url),
                      let uiImage = UIImage(data: imageData),
                      let cgImage = uiImage.cgImage else {
                    print("[UNet] No image to predict on")
                    return
                }

                // Run prediction
                isUNetProcessing = true
                let result = try await unetService.predict(cgImage: cgImage)
                isUNetProcessing = false

                // Apply mask
                applyUNetMask(result.mask, size: result.size)

                print("[UNet] Prediction applied successfully")
            } catch {
                isUNetLoading = false
                isUNetProcessing = false
                print("[UNet] Failed: \(error)")
            }
        }
    }

    /// Apply U-Net mask result to the current mask texture
    func applyUNetMask(_ unetMask: [UInt8], size: CGSize) {
        guard let textureManager = renderer?.textureManager else {
            print("[UNet] No texture manager")
            return
        }

        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)
        let srcWidth = Int(size.width)
        let srcHeight = Int(size.height)

        // Get current mask data for undo
        guard let currentMaskData = textureManager.readMask() else {
            print("[UNet] Failed to read current mask")
            return
        }

        // Create undo action
        let bbox = CGRect(x: 0, y: 0, width: CGFloat(maskWidth), height: CGFloat(maskHeight))
        let previousPatch = Data(currentMaskData)
        let action = UndoAction(classID: currentClassID, bbox: bbox, previousPatch: previousPatch)
        undoManager.pushUndo(action)

        // Scale U-Net mask to texture mask size
        var newMaskData = currentMaskData  // Start with current mask

        let scaleX = Float(srcWidth) / Float(maskWidth)
        let scaleY = Float(srcHeight) / Float(maskHeight)

        // Replace mask with U-Net inference result
        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)

                guard srcX < srcWidth && srcY < srcHeight else { continue }

                let srcIdx = srcY * srcWidth + srcX
                let dstIdx = y * maskWidth + x

                if unetMask[srcIdx] > 0 {
                    newMaskData[dstIdx] = UInt8(currentClassID)
                } else {
                    newMaskData[dstIdx] = 0
                }
            }
        }

        // Upload to texture
        do {
            try textureManager.uploadMask(newMaskData)
            maskModified = true
            print("[UNet] Mask applied with class \(currentClassID)")
        } catch {
            print("[UNet] Failed to upload mask: \(error)")
        }
    }

    // MARK: - HIL Server Integration

    /// Apply a server prediction mask to the canvas
    /// Thin wrapper around applyUNetMask for external callers
    func applyServerPredictionMask(_ binaryMask: [UInt8], sourceWidth: Int, sourceHeight: Int) {
        applyUNetMask(binaryMask, size: CGSize(width: sourceWidth, height: sourceHeight))
    }

    /// Export current mask as 512×512 RGBA PNG for server submission
    /// classID > 0 → red (255,0,0,255), classID == 0 → transparent (0,0,0,0)
    func exportMaskForServer() -> Data? {
        guard let textureManager = renderer?.textureManager,
              let maskData = textureManager.readMask() else {
            print("[HIL] No mask data to export")
            return nil
        }

        let srcWidth = Int(textureManager.maskSize.width)
        let srcHeight = Int(textureManager.maskSize.height)
        let dstWidth = 512
        let dstHeight = 512

        // Create RGBA buffer
        var rgba = [UInt8](repeating: 0, count: dstWidth * dstHeight * 4)

        let scaleX = Float(srcWidth) / Float(dstWidth)
        let scaleY = Float(srcHeight) / Float(dstHeight)

        for y in 0..<dstHeight {
            for x in 0..<dstWidth {
                let srcX = min(Int(Float(x) * scaleX), srcWidth - 1)
                let srcY = min(Int(Float(y) * scaleY), srcHeight - 1)
                let srcIdx = srcY * srcWidth + srcX
                let dstIdx = (y * dstWidth + x) * 4

                if maskData[srcIdx] > 0 {
                    rgba[dstIdx]     = 255  // R
                    rgba[dstIdx + 1] = 0    // G
                    rgba[dstIdx + 2] = 0    // B
                    rgba[dstIdx + 3] = 255  // A
                }
                // else: remains all zeros (transparent)
            }
        }

        // Convert to PNG via CGContext → CGImage → UIImage
        guard let context = CGContext(
            data: &rgba,
            width: dstWidth,
            height: dstHeight,
            bitsPerComponent: 8,
            bytesPerRow: dstWidth * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
              let cgImage = context.makeImage() else {
            print("[HIL] Failed to create PNG context")
            return nil
        }

        return UIImage(cgImage: cgImage).pngData()
    }

    // MARK: - Smooth Stroke Handling

    /// Begin smooth stroke at the given point
    func beginSmoothStroke(at point: CGPoint) {
        print("[Smooth] 🖌️ Stroke BEGIN at \(point)")
        smoothStrokePoints = [point]

        // Convert touch point to mask coordinates for bbox calculation
        let screenPoint = renderer?.convertTouchToScreen(point) ?? point
        let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? point

        // Guard against invalid coordinates
        guard maskPoint.x.isFinite && maskPoint.y.isFinite else {
            print("[Smooth] Invalid maskPoint at begin: \(maskPoint)")
            return
        }

        // Initialize bbox with brush radius padding
        let maskScaleFactor = renderer?.canvasTransform.maskScaleFactor ?? 2.0
        let padding = CGFloat(brushRadius * maskScaleFactor)
        smoothStrokeBbox = CGRect(
            x: maskPoint.x - padding,
            y: maskPoint.y - padding,
            width: padding * 2,
            height: padding * 2
        )
        print("[Smooth] Initial bbox: \(smoothStrokeBbox)")
    }

    /// Continue smooth stroke to the given point
    func continueSmoothStroke(to point: CGPoint) {
        smoothStrokePoints.append(point)

        // Expand bbox to include new point
        let screenPoint = renderer?.convertTouchToScreen(point) ?? point
        let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? point

        guard maskPoint.x.isFinite && maskPoint.y.isFinite else { return }

        let maskScaleFactor = renderer?.canvasTransform.maskScaleFactor ?? 2.0
        let padding = CGFloat(brushRadius * maskScaleFactor)
        let pointRect = CGRect(
            x: maskPoint.x - padding,
            y: maskPoint.y - padding,
            width: padding * 2,
            height: padding * 2
        )

        if smoothStrokeBbox.isNull {
            smoothStrokeBbox = pointRect
        } else {
            smoothStrokeBbox = smoothStrokeBbox.union(pointRect)
        }
    }

    /// Finish smooth stroke and apply morphological smoothing
    func finishSmoothStroke() {
        print("[Smooth] ✅ Stroke END - points: \(smoothStrokePoints.count), bbox: \(smoothStrokeBbox)")

        guard !smoothStrokeBbox.isNull else {
            print("[Smooth] No bbox to smooth")
            isSmoothMode = false
            return
        }

        // Perform morphological smoothing on the stroked region
        performMorphologicalSmoothing(in: smoothStrokeBbox)

        // Reset stroke tracking
        smoothStrokePoints.removeAll()
        smoothStrokeBbox = .null

        // Auto-disable smooth mode after one stroke
        isSmoothMode = false
    }

    /// Perform moving average smoothing on mask boundaries
    /// This smooths jagged edges by averaging nearby pixels
    private func performMorphologicalSmoothing(in bbox: CGRect) {
        guard let textureManager = renderer?.textureManager else {
            print("[Smooth] No texture manager")
            return
        }

        // 1. Read current mask data
        guard let previousData = textureManager.readMask() else {
            print("[Smooth] Failed to read mask")
            return
        }

        let width = Int(textureManager.maskSize.width)
        let height = Int(textureManager.maskSize.height)
        let fullBbox = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))

        // 2. Clamp bbox to texture bounds
        let clampedBbox = bbox.intersection(fullBbox)
        guard !clampedBbox.isNull else {
            print("[Smooth] Bbox out of bounds")
            return
        }

        // 3. Use configured kernel size for smoothing
        // Ensure odd number for symmetric window
        let kernelSize = smoothKernelSize | 1
        let halfKernel = kernelSize / 2

        // 4. Expand processing region by kernel size
        let expandedBbox = clampedBbox.insetBy(dx: CGFloat(-kernelSize), dy: CGFloat(-kernelSize))
        let processingBbox = expandedBbox.intersection(fullBbox)

        // 5. Convert bbox to integer bounds
        let minX = max(halfKernel, Int(processingBbox.minX))
        let minY = max(halfKernel, Int(processingBbox.minY))
        let maxX = min(width - halfKernel, Int(processingBbox.maxX))
        let maxY = min(height - halfKernel, Int(processingBbox.maxY))

        guard maxX > minX && maxY > minY else {
            print("[Smooth] Invalid processing bounds")
            return
        }

        // 6. Find which classes are present in the processing region
        var presentClasses = Set<UInt8>()
        for y in minY..<maxY {
            for x in minX..<maxX {
                let value = previousData[y * width + x]
                if value > 0 && value <= 8 {
                    presentClasses.insert(value)
                }
            }
        }

        guard !presentClasses.isEmpty else {
            print("[Smooth] No mask data in region")
            return
        }

        // 7. Apply competition-based smoothing
        // Instead of processing each class independently, have classes compete for boundary pixels
        // This prevents gaps between adjacent class boundaries
        // IMPORTANT: Only smooth pixels that were originally painted (not background)
        let numberOfPasses = 2
        var maskData = previousData

        for pass in 0..<numberOfPasses {
            var newMask = maskData

            for y in minY..<maxY {
                for x in minX..<maxX {
                    let idx = y * width + x
                    let currentValue = maskData[idx]
                    let originalValue = previousData[idx]

                    // Check if this pixel is near a boundary (has different neighbors)
                    var isBoundary = false
                    if x > 0 && maskData[idx - 1] != currentValue { isBoundary = true }
                    else if x < width - 1 && maskData[idx + 1] != currentValue { isBoundary = true }
                    else if y > 0 && maskData[idx - width] != currentValue { isBoundary = true }
                    else if y < height - 1 && maskData[idx + width] != currentValue { isBoundary = true }

                    // Only process boundary pixels
                    if !isBoundary { continue }

                    // For background pixels, only process if adjacent to painted pixels
                    // This fills gaps between classes but doesn't paint distant unpainted areas
                    if originalValue == 0 {
                        var hasPaintedNeighbor = false
                        if x > 0 && previousData[idx - 1] > 0 { hasPaintedNeighbor = true }
                        else if x < width - 1 && previousData[idx + 1] > 0 { hasPaintedNeighbor = true }
                        else if y > 0 && previousData[idx - width] > 0 { hasPaintedNeighbor = true }
                        else if y < height - 1 && previousData[idx + width] > 0 { hasPaintedNeighbor = true }

                        // Skip background pixels not adjacent to any painted pixel
                        if !hasPaintedNeighbor { continue }
                    }

                    // Calculate moving average for each class in kernel window
                    // Include background (0) as a class so boundaries with background also smooth
                    var classCounts: [UInt8: Float] = [:]

                    for ky in -halfKernel...halfKernel {
                        for kx in -halfKernel...halfKernel {
                            let nx = x + kx
                            let ny = y + ky
                            if nx >= 0 && nx < width && ny >= 0 && ny < height {
                                let neighborValue = maskData[ny * width + nx]
                                classCounts[neighborValue, default: 0] += 1
                            }
                        }
                    }

                    // Find the class with highest count (including background)
                    var bestClass: UInt8 = currentValue
                    var bestCount: Float = 0

                    for (classID, count) in classCounts {
                        if count > bestCount {
                            bestCount = count
                            bestClass = classID
                        }
                    }

                    // Assign pixel to winning class (can become background if background wins)
                    newMask[idx] = bestClass
                }
            }

            maskData = newMask
            print("[Smooth] Pass \(pass + 1)/\(numberOfPasses) completed")
        }

        // 8. Upload modified mask
        do {
            try textureManager.uploadMask(maskData)

            // 9. Create undo action
            let action = UndoAction(
                classID: 0,
                bbox: fullBbox,
                previousPatch: Data(previousData)
            )
            undoManager.pushUndo(action)

            print("[Smooth] Competition-based smoothing applied (kernel=\(kernelSize), classes=\(presentClasses))")
        } catch {
            print("[Smooth] Failed to upload mask: \(error)")
        }
    }

}

