import Foundation
import ZIPFoundation

/// Manages downloaded CoreML models in Application Support/HILModels/
/// Handles ZIP extraction, version tracking, and persistent storage.
nonisolated actor ModelSyncManager {
    static let shared = ModelSyncManager()

    // MARK: - Types

    struct SyncMetadata: Codable {
        let version: String
        let downloadedAt: Date
    }

    // MARK: - Paths

    private var modelsDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("HILModels", isDirectory: true)
    }

    private var metadataURL: URL {
        modelsDirectory.appendingPathComponent("metadata.json")
    }

    private var modelPackageURL: URL {
        modelsDirectory.appendingPathComponent("server_model.mlpackage", isDirectory: true)
    }

    private var compiledModelURL: URL {
        modelsDirectory.appendingPathComponent("server_model.mlmodelc", isDirectory: true)
    }

    // MARK: - Public Interface

    /// Whether a downloaded model exists and is ready for use
    var hasDownloadedModel: Bool {
        FileManager.default.fileExists(atPath: modelPackageURL.path)
            || FileManager.default.fileExists(atPath: compiledModelURL.path)
    }

    /// Current downloaded model version (nil if no model downloaded)
    var currentVersion: String? {
        loadMetadata()?.version
    }

    /// URL to the downloaded model (.mlmodelc preferred, falls back to .mlpackage)
    var modelURL: URL? {
        if FileManager.default.fileExists(atPath: compiledModelURL.path) {
            return compiledModelURL
        }
        if FileManager.default.fileExists(atPath: modelPackageURL.path) {
            return modelPackageURL
        }
        return nil
    }

    /// Save a ZIP-compressed model, extract it, and update metadata
    func saveModel(zipData: Data, version: String) throws {
        let fm = FileManager.default

        // Ensure directory exists
        try fm.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        // Clean previous model
        try? fm.removeItem(at: modelPackageURL)
        try? fm.removeItem(at: compiledModelURL)

        // Write ZIP to temp file
        let tempZip = modelsDirectory.appendingPathComponent("temp_model.zip")
        try zipData.write(to: tempZip)
        defer { try? fm.removeItem(at: tempZip) }

        // Extract ZIP
        let extractDir = modelsDirectory.appendingPathComponent("temp_extract", isDirectory: true)
        try? fm.removeItem(at: extractDir)
        try fm.unzipItem(at: tempZip, to: extractDir)

        // Find .mlpackage inside extracted contents
        let extracted = try fm.contentsOfDirectory(at: extractDir, includingPropertiesForKeys: nil)
        guard let mlpackage = extracted.first(where: { $0.pathExtension == "mlpackage" }) else {
            // The ZIP itself might be a flat mlpackage — check for Manifest.json
            let manifestCheck = extractDir.appendingPathComponent("Manifest.json")
            if fm.fileExists(atPath: manifestCheck.path) {
                // Extracted contents ARE the mlpackage
                try fm.moveItem(at: extractDir, to: modelPackageURL)
            } else {
                // Try one level deeper (sometimes ZIP has a single folder)
                let subdirs = extracted.filter { $0.hasDirectoryPath }
                if let subdir = subdirs.first {
                    let subManifest = subdir.appendingPathComponent("Manifest.json")
                    if subManifest.path.hasSuffix("Manifest.json"), fm.fileExists(atPath: subManifest.path) {
                        try fm.moveItem(at: subdir, to: modelPackageURL)
                        try? fm.removeItem(at: extractDir)
                    } else {
                        try? fm.removeItem(at: extractDir)
                        throw ModelSyncError.invalidModelPackage
                    }
                } else {
                    try? fm.removeItem(at: extractDir)
                    throw ModelSyncError.invalidModelPackage
                }
            }
            saveMetadata(SyncMetadata(version: version, downloadedAt: Date()))
            print("[ModelSync] Model saved (version: \(version))")
            return
        }

        // Move .mlpackage to final location
        try fm.moveItem(at: mlpackage, to: modelPackageURL)
        try? fm.removeItem(at: extractDir)

        // Save metadata
        saveMetadata(SyncMetadata(version: version, downloadedAt: Date()))
        print("[ModelSync] Model saved (version: \(version))")
    }

    /// Remove downloaded model and metadata
    func clear() {
        let fm = FileManager.default
        try? fm.removeItem(at: modelPackageURL)
        try? fm.removeItem(at: compiledModelURL)
        try? fm.removeItem(at: metadataURL)
        print("[ModelSync] Cleared downloaded model")
    }

    // MARK: - Private

    private func loadMetadata() -> SyncMetadata? {
        guard let data = try? Data(contentsOf: metadataURL) else { return nil }
        return try? JSONDecoder().decode(SyncMetadata.self, from: data)
    }

    private func saveMetadata(_ metadata: SyncMetadata) {
        guard let data = try? JSONEncoder().encode(metadata) else { return }
        try? data.write(to: metadataURL)
    }
}

// MARK: - Errors

enum ModelSyncError: LocalizedError {
    case invalidModelPackage

    var errorDescription: String? {
        switch self {
        case .invalidModelPackage:
            return "Downloaded file does not contain a valid .mlpackage"
        }
    }
}
