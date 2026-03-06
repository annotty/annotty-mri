import SwiftUI
import Combine

/// Coordinates HIL server interactions and state
/// Does NOT own CanvasViewModel — receives it as a method parameter
@MainActor
class HILViewModel: ObservableObject {
    // MARK: - Published State

    @Published var isConnected = false
    @Published var isLoading = false
    @Published var serverInfo: HILServerClient.ServerInfo?
    @Published var imageList: [HILServerClient.ImageInfo] = []
    @Published var currentImageId: String?
    @Published var trainingStatus: HILServerClient.TrainingStatus?
    @Published var errorMessage: String?

    @Published var isHILSubmitting = false
    @Published var isSyncingModel = false
    @Published var syncError: String?

    // MARK: - Dependencies

    private let settings: HILSettings
    private let client: HILServerClient
    private let cache = CacheManager.shared

    private var pollingTask: Task<Void, Never>?

    init(settings: HILSettings) {
        self.settings = settings
        self.client = HILServerClient(baseURL: settings.serverURL)
    }

    // MARK: - Connection

    /// Connect to server and fetch image list
    func connect() async {
        guard settings.isConfigured else { return }
        await updateBaseURL()
        isLoading = true
        errorMessage = nil

        do {
            let info = try await client.getInfo()
            serverInfo = info
            isConnected = true

            let response = try await client.listImages()
            imageList = response.images
        } catch {
            isConnected = false
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }

    // MARK: - On-Device Prediction

    /// Run on-device CoreML prediction via CanvasViewModel
    func requestPrediction(canvasVM: CanvasViewModel) {
        canvasVM.runUNetPrediction()
    }

    // MARK: - Submit & Delete

    /// Submit current mask to server, then delete the image from the device.
    /// Derives imageId from the canvas's current filename (= server imageId).
    func submitAndDelete(canvasVM: CanvasViewModel) async {
        // Derive imageId from the currently displayed image filename
        let imageId: String
        if let fileName = canvasVM.currentImageFileName {
            imageId = fileName
            print("[HIL] Submit: resolved imageId from canvas filename: \(imageId)")
        } else if let fallback = currentImageId {
            imageId = fallback
            print("[HIL] Submit: canvas filename nil, using fallback currentImageId: \(imageId)")
        } else {
            errorMessage = "No image to submit"
            print("[HIL] Submit: both canvas filename and currentImageId are nil")
            print("[HIL] Debug: canvasVM.currentImageIndex=\(canvasVM.currentImageIndex), totalImageCount=\(canvasVM.totalImageCount)")
            return
        }

        // Verify the image exists on the server
        let knownOnServer = imageList.contains(where: { $0.id == imageId })
        print("[HIL] Submit: imageId=\(imageId), knownOnServer=\(knownOnServer), imageList.count=\(imageList.count)")

        await updateBaseURL()
        isHILSubmitting = true
        errorMessage = nil

        do {
            // Export mask from canvas
            guard let maskPNG = canvasVM.exportMaskForServer() else {
                print("[HIL] Submit: exportMaskForServer() returned nil")
                throw HILError.maskConversionFailed
            }
            print("[HIL] Submit: mask exported, size=\(maskPNG.count) bytes")

            // Submit to server
            let result = try await client.submitLabel(imageId: imageId, maskPNG: maskPNG)
            print("[HIL] Submit: server response status=\(result.status)")

            // Refresh image list & server info
            let response = try await client.listImages()
            imageList = response.images
            serverInfo = try await client.getInfo()
            print("[HIL] Submit: refreshed imageList (\(imageList.count) images), labeled=\(serverInfo?.labeledImages ?? -1)")

            // Delete current image from device (navigates to adjacent image automatically)
            canvasVM.deleteCurrentImage()
            currentImageId = nil
            print("[HIL] Submit: deleted \(imageId) from device")
        } catch {
            errorMessage = "Submit failed: \(error.localizedDescription)"
            print("[HIL] Submit: ERROR \(error)")
        }

        isHILSubmitting = false
    }

    // MARK: - Batch Import

    /// Import multiple images from the server into the project
    func importImages(imageIds: Set<String>, canvasVM: CanvasViewModel) async {
        print("[HIL] Import: starting batch import of \(imageIds.count) images: \(imageIds.sorted())")
        await updateBaseURL()
        isLoading = true
        errorMessage = nil

        var lastImportedId: String?

        for imageId in imageIds {
            do {
                // Skip if already in project
                if canvasVM.navigateToImage(named: imageId) {
                    print("[HIL] Import: \(imageId) already in project, skipped download")
                    lastImportedId = imageId
                    continue
                }

                // Download image (cache → server)
                let imageData: Data
                if let cached = cache.loadImage(imageId: imageId) {
                    imageData = cached
                } else {
                    imageData = try await client.downloadImage(imageId: imageId)
                    cache.saveImage(imageData, imageId: imageId)
                }

                // Save to temp and import
                let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(imageId)
                try imageData.write(to: tempURL)
                canvasVM.importImage(from: tempURL)

                // If the image has a label on the server, download and save as annotation
                if let imageInfo = imageList.first(where: { $0.id == imageId }), imageInfo.hasLabel {
                    let labelData = try await client.downloadLabel(imageId: imageId)
                    // Find the imported image URL in the project
                    let imageURLs = ProjectFileService.shared.getImageURLs()
                    if let imageURL = imageURLs.first(where: { $0.lastPathComponent == imageId }) {
                        try ProjectFileService.shared.saveAnnotation(labelData, for: imageURL)
                        print("[HIL] Import: saved label for \(imageId)")
                    }
                }

                lastImportedId = imageId
                print("[HIL] Import: \(imageId) imported")
            } catch {
                print("[HIL] Import: failed for \(imageId): \(error)")
            }
        }

        // Navigate to the last imported image
        if let lastId = lastImportedId {
            let navigated = canvasVM.navigateToImage(named: lastId)
            print("[HIL] Import: navigated to \(lastId): \(navigated)")
        }

        print("[HIL] Import: done. canvasVM.currentImageFileName=\(canvasVM.currentImageFileName ?? "nil"), totalImageCount=\(canvasVM.totalImageCount)")
        isLoading = false
    }

    // MARK: - Training

    /// Start training on the server and begin polling status
    func startTraining() async {
        await updateBaseURL()
        errorMessage = nil

        do {
            _ = try await client.startTraining()
            startPollingTrainingStatus()
        } catch {
            errorMessage = error.localizedDescription
            autoClearError()
        }
    }

    /// Cancel ongoing training
    func cancelTraining() async {
        await updateBaseURL()
        do {
            _ = try await client.cancelTraining()
            pollingTask?.cancel()
            trainingStatus = nil
        } catch {
            errorMessage = "Failed to cancel training: \(error.localizedDescription)"
        }
    }

    // MARK: - Next Sample

    /// Fetch the next recommended sample from active learning
    func fetchNextSample() async {
        await updateBaseURL()
        do {
            let response = try await client.getNextSample()
            currentImageId = response.imageId
        } catch {
            errorMessage = "Failed to get next sample: \(error.localizedDescription)"
        }
    }

    // MARK: - Image Loading

    /// Download image from server and load into canvas
    /// Skips download if the image already exists in the project folder.
    func loadImageIntoCanvas(imageId: String, canvasVM: CanvasViewModel) async {
        await updateBaseURL()
        isLoading = true
        currentImageId = imageId

        do {
            // Check if already in project — just navigate to it
            if canvasVM.navigateToImage(named: imageId) {
                print("[HIL] Image \(imageId) already in project, navigated")
            } else {
                // Download (or use cache)
                let imageData: Data
                if let cached = cache.loadImage(imageId: imageId) {
                    imageData = cached
                } else {
                    imageData = try await client.downloadImage(imageId: imageId)
                    cache.saveImage(imageData, imageId: imageId)
                }

                // Save to temp and import (imageId already includes extension)
                let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(imageId)
                try imageData.write(to: tempURL)
                canvasVM.importImage(from: tempURL)

                print("[HIL] Image \(imageId) downloaded and loaded")
            }
        } catch {
            errorMessage = "Failed to load image: \(error.localizedDescription)"
        }

        isLoading = false
    }

    // MARK: - Model Sync

    /// Download the latest server model and reload UNet for on-device inference.
    /// Simply hits GET /models/latest — server returns 200+ZIP or 404.
    func syncModel(canvasVM: CanvasViewModel) async {
        await updateBaseURL()
        isSyncingModel = true
        syncError = nil

        do {
            // 1. Download model ZIP (server returns 404 if not available)
            print("[HIL] Sync: downloading model...")
            let zipData = try await client.downloadLatestModel()
            print("[HIL] Sync: downloaded \(zipData.count) bytes")

            // 2. Save and extract
            let version = ISO8601DateFormatter().string(from: Date())
            try await ModelSyncManager.shared.saveModel(zipData: zipData, version: version)

            // 3. Reload UNet with new model
            try await canvasVM.unetService.reloadModels()
            print("[HIL] Sync: model reloaded successfully")

        } catch let hilError as HILError {
            if case .serverError(let code, _) = hilError, code == 404 {
                syncError = "No model available on server"
            } else {
                syncError = hilError.localizedDescription
            }
            print("[HIL] Sync: \(syncError ?? "")")
        } catch {
            syncError = error.localizedDescription
            print("[HIL] Sync: ERROR \(error)")
        }

        isSyncingModel = false
    }

    // MARK: - Private Helpers

    private func updateBaseURL() async {
        await client.updateSettings(baseURL: settings.serverURL, apiKey: settings.apiKey)
    }

    /// Clear errorMessage after 5 seconds
    private func autoClearError() {
        Task {
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            if errorMessage != nil {
                errorMessage = nil
            }
        }
    }

    // MARK: - Training Status Polling

    private func startPollingTrainingStatus() {
        pollingTask?.cancel()
        pollingTask = Task {
            while !Task.isCancelled {
                do {
                    let status = try await client.getTrainingStatus()
                    trainingStatus = status

                    if status.status == "completed" || status.status == "error" || status.status == "idle" {
                        break
                    }
                } catch {
                    break
                }

                try? await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds
            }
        }
    }
}
