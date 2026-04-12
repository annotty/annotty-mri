import Foundation
import CoreML
import CoreGraphics
import Combine
import Metal

/// Result of U-Net mask prediction
struct UNetMaskResult {
    /// Mask at original image resolution (0=背景, 1-N=クラスID; バイナリの場合は0/1)
    let mask: [UInt8]
    /// Size of the mask (matches original image)
    let size: CGSize
    /// true when model outputs multi-class (argmax), false for binary (sigmoid)
    let isMultiClass: Bool
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

    /// Auto-detected model configuration (set after loading)
    private var modelConfig: ModelConfig = ModelConfig(inputChannels: 3, inputSize: 512, numClasses: 1)

    /// Number of ensemble folds
    static let foldCount: Int = 4

    // MARK: - ImageNet Normalization Constants
    // ResNet34 encoder was trained with these values
    private static let imagenetMean: [Float] = [0.485, 0.456, 0.406]  // RGB
    private static let imagenetStd: [Float] = [0.229, 0.224, 0.225]   // RGB

    // MARK: - Model Config

    /// Model configuration detected from CoreML model description
    private struct ModelConfig {
        let inputChannels: Int   // 1 or 3
        let inputSize: Int       // 256 or 512
        let numClasses: Int      // 1 or 10
    }

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
            modelConfig = Self.detectModelConfig(from: model)
            isReady = true
            lastError = nil
            print("[UNet] Server model loaded (version: \(await syncManager.currentVersion ?? "unknown")), config: \(modelConfig.inputChannels)ch \(modelConfig.inputSize)x\(modelConfig.inputSize) \(modelConfig.numClasses)cls")
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
        if let firstModel = loadedModels.first {
            modelConfig = Self.detectModelConfig(from: firstModel)
        }
        isReady = true
        lastError = nil
        print("[UNet] All \(Self.foldCount) bundled models loaded, config: \(modelConfig.inputChannels)ch \(modelConfig.inputSize)x\(modelConfig.inputSize) \(modelConfig.numClasses)cls")
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
        let inputSize = modelConfig.inputSize
        let numClasses = modelConfig.numClasses

        // 1. Resize image and create normalized MLMultiArray
        guard let inputArray = createNormalizedInputArray(from: cgImage, size: inputSize) else {
            throw UNetError.invalidInput
        }

        print("[UNet] Running inference on \(cgImage.width)x\(cgImage.height) image (\(modelConfig.inputChannels)ch, \(numClasses)cls)...")

        // 2. Run all models in parallel
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

        let pixelCount = inputSize * inputSize
        let isMultiClass = numClasses > 1

        // Intermediate upscale target: 256→512 for smoother boundaries
        let intermediateSize = 512
        let needsIntermediateUpscale = inputSize < intermediateSize

        var smallMask: [UInt8]

        if isMultiClass {
            // Multi-class: average logits across models [1, N, H, W]
            print("[UNet] Multi-class output (\(numClasses) classes), applying argmax...")

            var avgLogits = [Float](repeating: 0, count: numClasses * pixelCount)
            let divisor = Float(models.count)

            for logits in logitsArrays {
                let totalElements = numClasses * pixelCount
                if logits.dataType == .float16 {
                    let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                    for i in 0..<min(totalElements, logits.count) {
                        avgLogits[i] += Float(ptr[i])
                    }
                } else {
                    let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
                    for i in 0..<min(totalElements, logits.count) {
                        avgLogits[i] += ptr[i]
                    }
                }
            }

            // Divide by model count
            for i in 0..<(numClasses * pixelCount) {
                avgLogits[i] /= divisor
            }

            if needsIntermediateUpscale {
                // Bilinear upscale each class channel from inputSize→512, then argmax at 512
                print("[UNet] Upscaling logits \(inputSize)→\(intermediateSize) for smooth boundaries...")
                let intPixelCount = intermediateSize * intermediateSize
                var upscaledLogits = [Float](repeating: 0, count: numClasses * intPixelCount)
                let scale = Float(inputSize) / Float(intermediateSize)

                for c in 0..<numClasses {
                    let srcOffset = c * pixelCount
                    let dstOffset = c * intPixelCount
                    for y in 0..<intermediateSize {
                        let srcY = Float(y) * scale
                        let y0 = Int(srcY)
                        let y1 = min(y0 + 1, inputSize - 1)
                        let fy = srcY - Float(y0)
                        for x in 0..<intermediateSize {
                            let srcX = Float(x) * scale
                            let x0 = Int(srcX)
                            let x1 = min(x0 + 1, inputSize - 1)
                            let fx = srcX - Float(x0)

                            let v00 = avgLogits[srcOffset + y0 * inputSize + x0]
                            let v10 = avgLogits[srcOffset + y0 * inputSize + x1]
                            let v01 = avgLogits[srcOffset + y1 * inputSize + x0]
                            let v11 = avgLogits[srcOffset + y1 * inputSize + x1]

                            upscaledLogits[dstOffset + y * intermediateSize + x] =
                                v00 * (1 - fx) * (1 - fy) +
                                v10 * fx * (1 - fy) +
                                v01 * (1 - fx) * fy +
                                v11 * fx * fy
                        }
                    }
                }

                // Argmax at 512px
                smallMask = [UInt8](repeating: 0, count: intPixelCount)
                for i in 0..<intPixelCount {
                    var bestClass = 0
                    var bestVal = upscaledLogits[i]  // class 0
                    for c in 1..<numClasses {
                        let val = upscaledLogits[c * intPixelCount + i]
                        if val > bestVal {
                            bestVal = val
                            bestClass = c
                        }
                    }
                    smallMask[i] = UInt8(bestClass)
                }
            } else {
                // Already at 512+, argmax directly
                smallMask = [UInt8](repeating: 0, count: pixelCount)
                for i in 0..<pixelCount {
                    var bestClass = 0
                    var bestVal = avgLogits[i]  // class 0
                    for c in 1..<numClasses {
                        let val = avgLogits[c * pixelCount + i]
                        if val > bestVal {
                            bestVal = val
                            bestClass = c
                        }
                    }
                    smallMask[i] = UInt8(bestClass)
                }
            }
        } else {
            // Binary: sigmoid → average across models
            print("[UNet] Binary output, averaging sigmoid outputs...")
            var sigmoidAvg = [Float](repeating: 0, count: pixelCount)

            for logits in logitsArrays {
                let count = min(pixelCount, logits.count)
                if logits.dataType == .float16 {
                    let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                    for i in 0..<count {
                        let val = Float(ptr[i])
                        sigmoidAvg[i] += 1.0 / (1.0 + exp(-val))
                    }
                } else {
                    let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
                    for i in 0..<count {
                        sigmoidAvg[i] += 1.0 / (1.0 + exp(-ptr[i]))
                    }
                }
            }

            let divisor = Float(models.count)
            for i in 0..<pixelCount {
                sigmoidAvg[i] /= divisor
            }

            // Apply circular mask to probabilities (bundled fundus models only)
            // MRI images are rectangular — no lens circle masking needed
            if modelSource == .bundled {
                let half = Float(inputSize) / 2.0
                let radiusSq = half * half
                for y in 0..<inputSize {
                    let dy = Float(y) + 0.5 - half
                    let dySq = dy * dy
                    for x in 0..<inputSize {
                        let dx = Float(x) + 0.5 - half
                        if dx * dx + dySq > radiusSq {
                            sigmoidAvg[y * inputSize + x] = 0  // outside lens circle
                        }
                    }
                }
            }

            if needsIntermediateUpscale {
                // Bilinear upscale probabilities from inputSize→512, then threshold at 512
                print("[UNet] Upscaling probabilities \(inputSize)→\(intermediateSize) for smooth boundaries...")
                let intPixelCount = intermediateSize * intermediateSize
                let scale = Float(inputSize) / Float(intermediateSize)

                smallMask = [UInt8](repeating: 0, count: intPixelCount)
                for y in 0..<intermediateSize {
                    let srcY = Float(y) * scale
                    let y0 = Int(srcY)
                    let y1 = min(y0 + 1, inputSize - 1)
                    let fy = srcY - Float(y0)
                    for x in 0..<intermediateSize {
                        let srcX = Float(x) * scale
                        let x0 = Int(srcX)
                        let x1 = min(x0 + 1, inputSize - 1)
                        let fx = srcX - Float(x0)

                        let v00 = sigmoidAvg[y0 * inputSize + x0]
                        let v10 = sigmoidAvg[y0 * inputSize + x1]
                        let v01 = sigmoidAvg[y1 * inputSize + x0]
                        let v11 = sigmoidAvg[y1 * inputSize + x1]

                        let prob = v00 * (1 - fx) * (1 - fy) +
                                   v10 * fx * (1 - fy) +
                                   v01 * (1 - fx) * fy +
                                   v11 * fx * fy

                        smallMask[y * intermediateSize + x] = prob > 0.5 ? 1 : 0
                    }
                }
            } else {
                // Already at 512+, threshold directly
                smallMask = [UInt8](repeating: 0, count: pixelCount)
                for i in 0..<pixelCount {
                    smallMask[i] = sigmoidAvg[i] > 0.5 ? 1 : 0
                }
            }
        }

        // Upscale to original image size
        let maskSize = needsIntermediateUpscale ? intermediateSize : inputSize
        let upscaledMask: [UInt8]
        if isMultiClass {
            upscaledMask = upscaleMaskNearestNeighbor(smallMask, fromSize: maskSize, toSize: originalSize)
        } else {
            upscaledMask = upscaleMask(smallMask, fromSize: maskSize, toSize: originalSize)
        }

        print("[UNet] Prediction complete (\(Int(originalSize.width))x\(Int(originalSize.height)), multiClass=\(isMultiClass))")

        return UNetMaskResult(mask: upscaledMask, size: originalSize, isMultiClass: isMultiClass)
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

    /// モデルソースに応じて正規化方式を切り替える
    /// - downloaded (MRI): グレースケール + percentile clip + Z-score + 3ch複製
    /// - bundled (眼底): RGB + ImageNet正規化（従来通り）
    private func createNormalizedInputArray(from image: CGImage, size: Int) -> MLMultiArray? {
        if modelSource == .downloaded {
            return createZScoreInputArray(from: image, size: size)
        } else {
            return createImageNetInputArray(from: image, size: size)
        }
    }

    /// Downloaded (MRI) 用: グレースケール + percentile clip(1-99%) + Z-score + 3ch複製
    /// 学習時の前処理 (TOM NIfTI → percentile clip → Z-score → 3ch duplicate) と同一
    private func createZScoreInputArray(from image: CGImage, size: Int) -> MLMultiArray? {
        // 1. グレースケールで size×size に描画
        var pixelData = [UInt8](repeating: 0, count: size * size)
        guard let context = CGContext(
            data: &pixelData,
            width: size, height: size,
            bitsPerComponent: 8,
            bytesPerRow: size,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        // 2. Float 変換
        var floatData = pixelData.map { Float($0) }
        let n = floatData.count

        // 3. Percentile clip (1-99%)
        let sorted = floatData.sorted()
        let p1  = sorted[max(0,     Int(Float(n) * 0.01))]
        let p99 = sorted[min(n - 1, Int(Float(n) * 0.99))]
        floatData = floatData.map { min(max($0, p1), p99) }

        // 4. Z-score 正規化
        let mean = floatData.reduce(0, +) / Float(n)
        let variance = floatData.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(n)
        let std = variance > 1e-6 ? sqrt(variance) : 1.0
        floatData = floatData.map { ($0 - mean) / std }

        // 5. MLMultiArray [1, 3, size, size] にグレースケールを3ch複製
        guard let array = try? MLMultiArray(
            shape: [1, 3, NSNumber(value: size), NSNumber(value: size)],
            dataType: .float32
        ) else { return nil }

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let ch = size * size
        for i in 0..<ch {
            ptr[0 * ch + i] = floatData[i]   // R ← gray
            ptr[1 * ch + i] = floatData[i]   // G ← gray
            ptr[2 * ch + i] = floatData[i]   // B ← gray
        }
        return array
    }

    /// Bundled (眼底) 用: RGB + ImageNet正規化
    ///
    /// - 1ch model: RGB→grayscale (0.299R + 0.587G + 0.114B), then /255.0 → [1, 1, H, W]
    /// - 3ch model (bundled ImageType): (pixel - 255*mean) / std → [1, 3, H, W]
    /// - 3ch model (downloaded TensorType): (pixel/255 - mean) / std → [1, 3, H, W]
    private func createImageNetInputArray(from image: CGImage, size: Int) -> MLMultiArray? {
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

        let channels = modelConfig.inputChannels

        if channels == 1 {
            // 1ch grayscale: RGB → gray, /255.0
            guard let array = try? MLMultiArray(shape: [1, 1, NSNumber(value: size), NSNumber(value: size)], dataType: .float32) else {
                return nil
            }
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            for y in 0..<size {
                for x in 0..<size {
                    let pixelIndex = (y * size + x) * bytesPerPixel
                    let r = Float(pixelData[pixelIndex])
                    let g = Float(pixelData[pixelIndex + 1])
                    let b = Float(pixelData[pixelIndex + 2])
                    let gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                    ptr[y * size + x] = gray
                }
            }
            return array
        }

        // 3ch RGB with ImageNet normalization
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32) else {
            return nil
        }

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = size * size

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

    /// Upscale multi-class mask using nearest-neighbor (preserves class IDs)
    private func upscaleMaskNearestNeighbor(_ mask: [UInt8], fromSize srcSize: Int, toSize size: CGSize) -> [UInt8] {
        let dstWidth = Int(size.width)
        let dstHeight = Int(size.height)
        var result = [UInt8](repeating: 0, count: dstWidth * dstHeight)

        let scaleX = Float(srcSize) / Float(dstWidth)
        let scaleY = Float(srcSize) / Float(dstHeight)

        for y in 0..<dstHeight {
            let srcY = min(Int(Float(y) * scaleY), srcSize - 1)
            for x in 0..<dstWidth {
                let srcX = min(Int(Float(x) * scaleX), srcSize - 1)
                result[y * dstWidth + x] = mask[srcY * srcSize + srcX]
            }
        }

        return result
    }

    /// Detect model configuration from CoreML model description
    private static func detectModelConfig(from model: MLModel) -> ModelConfig {
        let desc = model.modelDescription

        // Detect input: look for the "image" input and read [1, C, H, W]
        var inputChannels = 3
        var inputSize = 512
        if let imageInput = desc.inputDescriptionsByName["image"],
           let constraint = imageInput.multiArrayConstraint {
            let shape = constraint.shape.map { $0.intValue }
            // Expected [1, C, H, W]
            if shape.count == 4 {
                inputChannels = shape[1]
                inputSize = shape[2]  // assume H == W
            }
        }

        // Detect output: look for "logits" or first MultiArray, read [1, N, H, W]
        var numClasses = 1
        let outputNames = ["logits"] + desc.outputDescriptionsByName.keys.sorted()
        for name in outputNames {
            if let outputDesc = desc.outputDescriptionsByName[name],
               let constraint = outputDesc.multiArrayConstraint {
                let shape = constraint.shape.map { $0.intValue }
                // Expected [1, N, H, W]
                if shape.count == 4 {
                    numClasses = shape[1]
                }
                break
            }
        }

        print("[UNet] Detected config: input=\(inputChannels)ch \(inputSize)x\(inputSize), output=\(numClasses) classes")
        return ModelConfig(inputChannels: inputChannels, inputSize: inputSize, numClasses: numClasses)
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
