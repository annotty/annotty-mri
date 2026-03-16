import Foundation
import CoreML
import CoreGraphics
import Combine
import Metal

/// Result of U-Net mask prediction
struct UNetMaskResult {
    /// Mask at original image resolution: class IDs (0=background, 1-9=class) for multiclass, or 0/1 for binary
    let mask: [UInt8]
    /// Size of the mask (matches original image)
    let size: CGSize
}

/// Source of the loaded model(s)
enum ModelSource: String {
    case bundled = "Built-in"
    case downloaded = "Server"
}

/// U-Net ensemble inference service (4-fold, prompt-free)
/// Loads 4 CoreML models and averages their sigmoid outputs for robust segmentation.
/// When a downloaded server model is available, uses that single model instead.
///
/// Model variants:
///   - Bundled: 4-fold ensemble, RGB 3ch, 512×512, binary output
///   - Downloaded (server): single model, grayscale 1ch, 256×256, 10-class output
@MainActor
final class UNetService: ObservableObject {
    // MARK: - Published State

    @Published private(set) var isReady: Bool = false
    @Published private(set) var isProcessing: Bool = false
    @Published private(set) var lastError: String?
    @Published private(set) var modelSource: ModelSource = .bundled

    // MARK: - Models

    private var models: [MLModel] = []

    /// Model input size — depends on model source
    static let bundledInputSize: Int = 512
    static let downloadedInputSize: Int = 256

    /// Number of output classes for downloaded server model
    static let numClasses: Int = 10

    /// Number of ensemble folds
    static let foldCount: Int = 4

    /// Effective input size for the currently loaded model
    var inputSize: Int {
        modelSource == .downloaded ? Self.downloadedInputSize : Self.bundledInputSize
    }

    // MARK: - ImageNet Normalization Constants
    // ResNet34 encoder was trained with these values
    private static let imagenetMean: [Float] = [0.485, 0.456, 0.406]  // RGB
    private static let imagenetStd: [Float] = [0.229, 0.224, 0.225]   // RGB

    // MARK: - Initialization

    init() {
        print("[UNet] Service initialized (models not loaded yet)")
    }

    /// Load models — prefers downloaded server model over bundled 4-fold ensemble
    func loadModels() async throws {
        if isReady {
            print("[UNet] Models already loaded")
            return
        }

        let config = MLModelConfiguration()
        config.computeUnits = Self.detectOptimalComputeUnits()

        // Try downloaded server model first
        let syncManager = ModelSyncManager.shared
        if await syncManager.hasDownloadedModel, let serverModelURL = await syncManager.modelURL {
            print("[UNet] Loading downloaded server model...")
            let compiledURL: URL
            if serverModelURL.pathExtension == "mlmodelc" {
                compiledURL = serverModelURL
            } else {
                compiledURL = try await MLModel.compileModel(at: serverModelURL)
            }
            let model = try MLModel(contentsOf: compiledURL, configuration: config)
            models = [model]
            modelSource = .downloaded
            isReady = true
            lastError = nil
            print("[UNet] Server model loaded (version: \(await syncManager.currentVersion ?? "unknown"))")
            return
        }

        // Fall back to bundled 4-fold ensemble
        print("[UNet] Loading \(Self.foldCount) bundled fold models...")

        var loadedModels: [MLModel] = []

        for i in 1...Self.foldCount {
            let modelName = "unet_fold\(i)"
            guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
                    ?? Self.compileModelIfNeeded(name: modelName) else {
                let error = "Model not found: \(modelName)"
                lastError = error
                print("[UNet] \(error)")
                throw UNetError.modelNotFound(modelName)
            }

            let model = try MLModel(contentsOf: modelURL, configuration: config)
            loadedModels.append(model)
            print("[UNet] Loaded fold \(i)")
        }

        models = loadedModels
        modelSource = .bundled
        isReady = true
        lastError = nil
        print("[UNet] All \(Self.foldCount) bundled models loaded successfully")
    }

    /// Reload models after a new server model is downloaded
    func reloadModels() async throws {
        print("[UNet] Reloading models...")
        models = []
        isReady = false
        lastError = nil
        try await loadModels()
    }

    /// Predict segmentation mask from a CGImage (prompt-free, full-image)
    func predict(cgImage: CGImage) async throws -> UNetMaskResult {
        guard isReady, let firstModel = models.first else {
            throw UNetError.modelNotLoaded
        }

        isProcessing = true
        defer { isProcessing = false }

        let originalSize = CGSize(width: cgImage.width, height: cgImage.height)

        // Determine model type from actual model input description,
        // not from modelSource flag (which can be stale due to race conditions)
        let isMulticlass = Self.isMulticlassModel(firstModel)
        let size = isMulticlass ? Self.downloadedInputSize : Self.bundledInputSize

        if isMulticlass {
            return try await predictMulticlass(cgImage: cgImage, originalSize: originalSize, size: size)
        } else {
            return try await predictBundledEnsemble(cgImage: cgImage, originalSize: originalSize, size: size)
        }
    }

    /// Check if a CoreML model expects single-channel (grayscale) input = multiclass server model
    /// Server model: [1, 1, 256, 256], Bundled: [1, 3, 512, 512]
    private static func isMulticlassModel(_ model: MLModel) -> Bool {
        guard let imageInput = model.modelDescription.inputDescriptionsByName["image"],
              let constraint = imageInput.multiArrayConstraint else {
            return false
        }
        // shape[1] = number of input channels: 1 = grayscale (server), 3 = RGB (bundled)
        return constraint.shape.count >= 2 && constraint.shape[1].intValue == 1
    }

    // MARK: - Downloaded Server Model (1ch grayscale, 256×256, 10-class)

    private func predictMulticlass(cgImage: CGImage, originalSize: CGSize, size: Int) async throws -> UNetMaskResult {
        // 1. Create grayscale 1-channel input [1, 1, 256, 256]
        guard let inputArray = createGrayscaleInputArray(from: cgImage, size: size) else {
            throw UNetError.invalidInput
        }

        print("[UNet] Running server model inference (1ch, \(size)×\(size), \(Self.numClasses) classes)...")

        let model = models[0]
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": inputArray])
        let output = try model.prediction(from: input)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue
                ?? output.featureNames.lazy.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first else {
            throw UNetError.inferenceError("No logits output from server model")
        }

        // 2. Argmax across classes → class ID per pixel
        //    logits shape: [1, numClasses, H, W]
        let pixelCount = size * size
        var smallMask = [UInt8](repeating: 0, count: pixelCount)
        let numClasses = Self.numClasses

        if logits.dataType == .float16 {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<pixelCount {
                var maxVal = Float(ptr[i])  // class 0
                var maxClass: UInt8 = 0
                for c in 1..<numClasses {
                    let val = Float(ptr[c * pixelCount + i])
                    if val > maxVal {
                        maxVal = val
                        maxClass = UInt8(c)
                    }
                }
                smallMask[i] = maxClass
            }
        } else {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<pixelCount {
                var maxVal = ptr[i]  // class 0
                var maxClass: UInt8 = 0
                for c in 1..<numClasses {
                    let val = ptr[c * pixelCount + i]
                    if val > maxVal {
                        maxVal = val
                        maxClass = UInt8(c)
                    }
                }
                smallMask[i] = maxClass
            }
        }

        // 3. Upscale to original size (nearest neighbor for class IDs)
        let upscaledMask = upscaleMaskNearest(smallMask, fromSize: size, toSize: originalSize)

        print("[UNet] Multiclass prediction complete (\(Int(originalSize.width))x\(Int(originalSize.height)))")
        return UNetMaskResult(mask: upscaledMask, size: originalSize)
    }

    // MARK: - Bundled Ensemble (3ch RGB, 512×512, binary)

    private func predictBundledEnsemble(cgImage: CGImage, originalSize: CGSize, size: Int) async throws -> UNetMaskResult {
        // 1. Resize image to 512x512 and create ImageNet-normalized MLMultiArray
        guard let inputArray = createNormalizedInputArray(from: cgImage, size: size) else {
            throw UNetError.invalidInput
        }

        print("[UNet] Running ensemble inference on \(cgImage.width)x\(cgImage.height) image...")

        // 2. Run all 4 models in parallel
        let logitsArrays = try await withThrowingTaskGroup(of: (Int, MLMultiArray).self) { group in
            for (index, model) in models.enumerated() {
                group.addTask { [inputArray] in
                    let input = try MLDictionaryFeatureProvider(dictionary: [
                        "image": inputArray
                    ])
                    let output = try model.prediction(from: input)
                    let logits: MLMultiArray
                    if let named = output.featureValue(for: "logits")?.multiArrayValue {
                        logits = named
                    } else if let first = output.featureNames.lazy.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first {
                        logits = first
                    } else {
                        throw UNetError.inferenceError("No logits output from fold \(index + 1)")
                    }
                    return (index, logits)
                }
            }

            var results = [(Int, MLMultiArray)]()
            for try await result in group {
                results.append(result)
            }
            return results.sorted(by: { $0.0 < $1.0 }).map(\.1)
        }

        print("[UNet] All folds completed, averaging sigmoid outputs...")

        // 3. Average sigmoid(logits) across 4 folds
        let pixelCount = size * size
        var sigmoidSum = [Float](repeating: 0, count: pixelCount)

        for logits in logitsArrays {
            let count = min(pixelCount, logits.count)
            if logits.dataType == .float16 {
                let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<count {
                    let val = Float(ptr[i])
                    sigmoidSum[i] += 1.0 / (1.0 + exp(-val))
                }
            } else {
                let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<count {
                    sigmoidSum[i] += 1.0 / (1.0 + exp(-ptr[i]))
                }
            }
        }

        // 4. Average + threshold + circular mask → binary mask
        let divisor = Float(models.count)
        let half = Float(size) / 2.0
        let radiusSq = half * half
        var smallMask = [UInt8](repeating: 0, count: pixelCount)
        for y in 0..<size {
            let dy = Float(y) + 0.5 - half
            let dySq = dy * dy
            for x in 0..<size {
                let dx = Float(x) + 0.5 - half
                if dx * dx + dySq > radiusSq {
                    continue
                }
                let i = y * size + x
                smallMask[i] = (sigmoidSum[i] / divisor) > 0.5 ? 1 : 0
            }
        }

        // 5. Upscale to original image size
        let upscaledMask = upscaleMask(smallMask, fromSize: size, toSize: originalSize)

        print("[UNet] Prediction complete (\(Int(originalSize.width))x\(Int(originalSize.height)))")
        return UNetMaskResult(mask: upscaledMask, size: originalSize)
    }

    // MARK: - Private Helpers

    /// Detect optimal compute units based on device GPU capability
    private static func detectOptimalComputeUnits() -> MLComputeUnits {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[UNet] No Metal device, using CPU only")
            return .cpuOnly
        }
        if device.supportsFamily(.apple6) {
            print("[UNet] GPU Apple6+ (\(device.name)) - using CPU + GPU")
            return .cpuAndGPU
        } else {
            print("[UNet] GPU < Apple6 (\(device.name)) - using CPU only")
            return .cpuOnly
        }
    }

    /// Try to find and compile an .mlpackage from the bundle
    private static func compileModelIfNeeded(name: String) -> URL? {
        guard let packageURL = Bundle.main.url(forResource: name, withExtension: "mlpackage") else {
            return nil
        }
        do {
            let compiledURL = try MLModel.compileModel(at: packageURL)
            print("[UNet] Compiled \(name) on first launch")
            return compiledURL
        } catch {
            print("[UNet] Failed to compile \(name): \(error)")
            return nil
        }
    }

    /// Resize CGImage to 512x512 and create ImageNet-normalized MLMultiArray [1, 3, H, W]
    ///
    /// Normalization differs by model source:
    /// - Bundled (ImageType): has baked-in 1/255, so provide (pixel - 255*mean) / std
    /// - Downloaded (TensorType): no internal preprocessing, provide (pixel/255 - mean) / std
    private func createNormalizedInputArray(from image: CGImage, size: Int) -> MLMultiArray? {
        // Draw image into an RGBA byte buffer at target size
        let bytesPerPixel = 4
        let bytesPerRow = size * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: size * size * bytesPerPixel)

        guard let context = CGContext(
            data: &pixelData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue  // RGBX layout
        ) else {
            return nil
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        // Create MLMultiArray with shape [1, 3, 512, 512]
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32) else {
            return nil
        }

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = size * size

        // ImageNet normalization target: (pixel/255 - mean) / std
        //
        // Bundled (ImageType):   model has internal 1/255
        //   → provide (pixel - 255*mean) / std, model divides by 255 → correct
        // Downloaded (TensorType): no internal preprocessing
        //   → provide (pixel/255 - mean) / std directly
        let scale: Float = modelSource == .bundled ? 1.0 : (1.0 / 255.0)
        let meanScale: Float = modelSource == .bundled ? 255.0 : 1.0

        let meanR = meanScale * Self.imagenetMean[0]
        let meanG = meanScale * Self.imagenetMean[1]
        let meanB = meanScale * Self.imagenetMean[2]
        let stdR = Self.imagenetStd[0]
        let stdG = Self.imagenetStd[1]
        let stdB = Self.imagenetStd[2]

        for y in 0..<size {
            for x in 0..<size {
                let pixelIndex = (y * size + x) * bytesPerPixel
                let spatialIndex = y * size + x

                let r = Float(pixelData[pixelIndex]) * scale
                let g = Float(pixelData[pixelIndex + 1]) * scale
                let b = Float(pixelData[pixelIndex + 2]) * scale

                ptr[0 * channelStride + spatialIndex] = (r - meanR) / stdR
                ptr[1 * channelStride + spatialIndex] = (g - meanG) / stdG
                ptr[2 * channelStride + spatialIndex] = (b - meanB) / stdB
            }
        }

        return array
    }

    /// Create grayscale 1-channel MLMultiArray [1, 1, H, W] for downloaded server model
    /// Normalization: (pixel/255 - mean_gray) / std_gray  (ImageNet grayscale approx)
    private func createGrayscaleInputArray(from image: CGImage, size: Int) -> MLMultiArray? {
        // Draw image into RGBA buffer at target size
        let bytesPerPixel = 4
        let bytesPerRow = size * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: size * size * bytesPerPixel)

        guard let context = CGContext(
            data: &pixelData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return nil
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        // Create MLMultiArray with shape [1, 1, H, W]
        guard let array = try? MLMultiArray(
            shape: [1, 1, NSNumber(value: size), NSNumber(value: size)],
            dataType: .float32
        ) else {
            return nil
        }

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)

        // Convert RGB to grayscale luminance, then normalize to [0, 1]
        // Server model uses raw grayscale / 255.0 (no ImageNet normalization for 1ch)
        for y in 0..<size {
            for x in 0..<size {
                let pixelIndex = (y * size + x) * bytesPerPixel
                let r = Float(pixelData[pixelIndex])
                let g = Float(pixelData[pixelIndex + 1])
                let b = Float(pixelData[pixelIndex + 2])
                // Standard luminance weights
                let gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255.0
                ptr[y * size + x] = gray
            }
        }

        return array
    }

    /// Upscale class ID mask using nearest neighbor (preserves discrete class IDs)
    private func upscaleMaskNearest(_ mask: [UInt8], fromSize srcSize: Int, toSize size: CGSize) -> [UInt8] {
        let dstWidth = Int(size.width)
        let dstHeight = Int(size.height)
        var result = [UInt8](repeating: 0, count: dstWidth * dstHeight)

        let scaleX = Float(srcSize) / Float(dstWidth)
        let scaleY = Float(srcSize) / Float(dstHeight)

        for y in 0..<dstHeight {
            for x in 0..<dstWidth {
                let srcX = min(Int(Float(x) * scaleX), srcSize - 1)
                let srcY = min(Int(Float(y) * scaleY), srcSize - 1)
                result[y * dstWidth + x] = mask[srcY * srcSize + srcX]
            }
        }

        return result
    }

    /// Upscale binary mask from square size to target size using bilinear interpolation
    private func upscaleMask(_ mask: [UInt8], fromSize srcSize: Int, toSize size: CGSize) -> [UInt8] {
        let dstWidth = Int(size.width)
        let dstHeight = Int(size.height)
        var result = [UInt8](repeating: 0, count: dstWidth * dstHeight)

        let scaleX = Float(srcSize) / Float(dstWidth)
        let scaleY = Float(srcSize) / Float(dstHeight)

        for y in 0..<dstHeight {
            for x in 0..<dstWidth {
                let srcX = Float(x) * scaleX
                let srcY = Float(y) * scaleY

                let x0 = Int(srcX)
                let y0 = Int(srcY)
                let x1 = min(x0 + 1, srcSize - 1)
                let y1 = min(y0 + 1, srcSize - 1)

                let fx = srcX - Float(x0)
                let fy = srcY - Float(y0)

                let v00 = Float(mask[y0 * srcSize + x0])
                let v10 = Float(mask[y0 * srcSize + x1])
                let v01 = Float(mask[y1 * srcSize + x0])
                let v11 = Float(mask[y1 * srcSize + x1])

                let value = v00 * (1 - fx) * (1 - fy) +
                           v10 * fx * (1 - fy) +
                           v01 * (1 - fx) * fy +
                           v11 * fx * fy

                result[y * dstWidth + x] = value > 0.5 ? 1 : 0
            }
        }

        return result
    }
}

// MARK: - Error Types

enum UNetError: Error, LocalizedError {
    case modelNotLoaded
    case modelNotFound(String)
    case inferenceError(String)
    case invalidInput

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "U-Net models are not loaded"
        case .modelNotFound(let name):
            return "U-Net model not found: \(name)"
        case .inferenceError(let message):
            return "U-Net inference error: \(message)"
        case .invalidInput:
            return "Invalid input for U-Net"
        }
    }
}
