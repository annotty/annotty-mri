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

    /// 症例一覧（サーバーから取得）
    @Published var cases: [HILServerClient.CaseInfo] = []
    /// 現在選択中の症例ID
    @Published var selectedCaseId: String?

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

    /// Connect to server and fetch case list + image list
    func connect(canvasVM: CanvasViewModel? = nil) async {
        guard settings.isConfigured else { return }
        await updateBaseURL()
        isLoading = true
        errorMessage = nil

        do {
            let info = try await client.getInfo()
            serverInfo = info
            isConnected = true

            // 症例一覧を取得
            let caseResponse = try await client.listCases()
            cases = caseResponse.cases

            // 症例が未選択なら最初の未ラベルありの症例を自動選択
            if selectedCaseId == nil, let firstUnlabeled = cases.first(where: { $0.unlabeledSlices > 0 }) {
                selectedCaseId = firstUnlabeled.caseId
            } else if selectedCaseId == nil, let first = cases.first {
                selectedCaseId = first.caseId
            }

            // 選択中の症例の画像一覧を取得
            if let caseId = selectedCaseId {
                await fetchCaseImages(caseId: caseId, canvasVM: canvasVM)
            } else {
                imageList = []
            }
        } catch {
            isConnected = false
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }

    /// 症例を切り替える
    func selectCase(caseId: String, canvasVM: CanvasViewModel? = nil) async {
        selectedCaseId = caseId
        isLoading = true
        errorMessage = nil

        await fetchCaseImages(caseId: caseId, canvasVM: canvasVM)

        isLoading = false
    }

    /// 症例の画像一覧を取得（内部用）
    private func fetchCaseImages(caseId: String, canvasVM: CanvasViewModel? = nil) async {
        do {
            let response = try await client.listCaseImages(caseId: caseId)
            imageList = response.images

            // label_configを取得して適用
            if let canvasVM = canvasVM {
                do {
                    let labelConfig = try await client.downloadCaseLabelConfig(caseId: caseId)
                    let entries = labelConfig.classes.map { cls in
                        ProjectFileService.LabelClassEntry(id: cls.id, name: cls.name, color: cls.color)
                    }
                    ProjectFileService.shared.saveLabelConfig(classes: entries)
                    canvasVM.loadLabelConfigFromProject()
                    print("[HIL] connect: applied \(entries.count) classes from case \(caseId)")
                } catch {
                    print("[HIL] connect: label_config not available for case \(caseId): \(error)")
                }
            }
        } catch {
            imageList = []
            errorMessage = error.localizedDescription
        }
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

        // Use case-based API if a case is selected, otherwise fall back to flat API
        let caseId = selectedCaseId
        let knownOnServer = imageList.contains(where: { $0.id == imageId })
        print("[HIL] Submit: imageId=\(imageId), caseId=\(caseId ?? "nil"), knownOnServer=\(knownOnServer)")

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

            // Submit to server (case-based or flat)
            if let caseId = caseId {
                let result = try await client.submitCaseLabel(caseId: caseId, imageId: imageId, maskPNG: maskPNG)
                print("[HIL] Submit: server response status=\(result.status)")

                // Refresh with case-based API
                let response = try await client.listCaseImages(caseId: caseId)
                imageList = response.images
            } else {
                let result = try await client.submitLabel(imageId: imageId, maskPNG: maskPNG)
                print("[HIL] Submit: server response status=\(result.status)")

                let response = try await client.listImages()
                imageList = response.images
            }
            serverInfo = try await client.getInfo()
            print("[HIL] Submit: refreshed imageList (\(imageList.count) images), labeled=\(serverInfo?.labeledImages ?? -1)")

            // Refresh case list to update progress counts
            if let _ = caseId {
                let caseResponse = try await client.listCases()
                cases = caseResponse.cases
            }

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
        guard let caseId = selectedCaseId else {
            errorMessage = "No case selected"
            return
        }

        print("[HIL] Import: starting batch import of \(imageIds.count) images from case \(caseId)")
        await updateBaseURL()
        isLoading = true

        // サーバーからlabel_configを取得してiPadに適用
        do {
            let labelConfig = try await client.downloadCaseLabelConfig(caseId: caseId)
            let entries = labelConfig.classes.map { cls in
                ProjectFileService.LabelClassEntry(id: cls.id, name: cls.name, color: cls.color)
            }
            ProjectFileService.shared.saveLabelConfig(classes: entries)
            canvasVM.loadLabelConfigFromProject()
            print("[HIL] Import: applied \(entries.count) classes from case \(caseId) label_config")
        } catch {
            print("[HIL] Import: label_config not available for case \(caseId): \(error)")
        }
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
                let cacheKey = "\(caseId)/\(imageId)"
                if let cached = cache.loadImage(imageId: cacheKey) {
                    imageData = cached
                } else {
                    imageData = try await client.downloadCaseImage(caseId: caseId, imageId: imageId)
                    cache.saveImage(imageData, imageId: cacheKey)
                }

                // Save to temp and import
                let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(imageId)
                try imageData.write(to: tempURL)
                canvasVM.importImage(from: tempURL)

                // If the image has a label on the server, download and save as annotation
                if let imageInfo = imageList.first(where: { $0.id == imageId }), imageInfo.hasLabel {
                    let labelData = try await client.downloadCaseLabel(caseId: caseId, imageId: imageId)
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

        let caseId = selectedCaseId

        // サーバーからlabel_configを取得してiPadに適用
        do {
            let labelConfig: HILServerClient.LabelConfigResponse
            if let caseId = caseId {
                labelConfig = try await client.downloadCaseLabelConfig(caseId: caseId)
            } else {
                labelConfig = try await client.downloadLabelConfig()
            }
            let entries = labelConfig.classes.map { cls in
                ProjectFileService.LabelClassEntry(id: cls.id, name: cls.name, color: cls.color)
            }
            ProjectFileService.shared.saveLabelConfig(classes: entries)
            canvasVM.loadLabelConfigFromProject()
            print("[HIL] loadImage: applied \(entries.count) classes from label_config")
        } catch {
            print("[HIL] loadImage: label_config not available: \(error)")
        }

        do {
            // Check if already in project — just navigate to it
            if canvasVM.navigateToImage(named: imageId) {
                print("[HIL] Image \(imageId) already in project, navigated")
            } else {
                // Download (or use cache)
                let imageData: Data
                let cacheKey = caseId.map { "\($0)/\(imageId)" } ?? imageId
                if let cached = cache.loadImage(imageId: cacheKey) {
                    imageData = cached
                } else {
                    if let caseId = caseId {
                        imageData = try await client.downloadCaseImage(caseId: caseId, imageId: imageId)
                    } else {
                        imageData = try await client.downloadImage(imageId: imageId)
                    }
                    cache.saveImage(imageData, imageId: cacheKey)
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
