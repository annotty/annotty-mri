import Foundation
import CoreML
import CoreGraphics
import Combine
import Metal

/// Result of U-Net ensemble mask prediction
struct UNetMaskResult {
    /// Binary mask (0 or 1) at original image resolution
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
@MainActor
final class UNetService: ObservableObject {
    // MARK: - Published State

    @Published private(set) var isReady: Bool = false
    @Published private(set) var isProcessing: Bool = false
    @Published private(set) var lastError: String?
    @Published private(set) var modelSource: ModelSource = .bundled

    // MARK: - Models

    private var models: [MLModel] = []

    /// Model input size (must match conversion script)
    static let inputSize: Int = 512

    /// Number of ensemble folds
    static let foldCount: Int = 4

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
        guard isReady else {
            throw UNetError.modelNotLoaded
        }

        isProcessing = true
        defer { isProcessing = false }

        let originalSize = CGSize(width: cgImage.width, height: cgImage.height)

        // 1. Resize image to 512x512 and create ImageNet-normalized MLMultiArray
        guard let inputArray = createNormalizedInputArray(from: cgImage, size: Self.inputSize) else {
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
                    // Try "logits" first, fall back to first MultiArray output
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

        // 3. Average sigmoid(logits) across 4 folds (matching Python reference)
        let pixelCount = Self.inputSize * Self.inputSize
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

        // 4. Average + threshold (0.5) + circular mask → binary mask at 512x512
        //    Training excluded pixels outside the lens circle, so we hard-mask
        //    to a circle inscribed in the square (center, radius = size/2).
        let divisor = Float(models.count)
        let half = Float(Self.inputSize) / 2.0
        let radiusSq = half * half
        var smallMask = [UInt8](repeating: 0, count: pixelCount)
        for y in 0..<Self.inputSize {
            let dy = Float(y) + 0.5 - half
            let dySq = dy * dy
            for x in 0..<Self.inputSize {
                let dx = Float(x) + 0.5 - half
                if dx * dx + dySq > radiusSq {
                    continue  // outside lens circle → leave as 0
                }
                let i = y * Self.inputSize + x
                smallMask[i] = (sigmoidSum[i] / divisor) > 0.5 ? 1 : 0
            }
        }

        // 5. Upscale to original image size using bilinear interpolation
        let upscaledMask = upscaleMask(
            smallMask,
            fromSize: Self.inputSize,
            toSize: originalSize
        )

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
