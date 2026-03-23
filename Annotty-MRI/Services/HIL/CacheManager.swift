import Foundation

/// Manages local cache for HIL downloaded images
/// Uses Caches directory which the system can purge under storage pressure
final class CacheManager {
    static let shared = CacheManager()

    private let cacheDir: URL

    private init() {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        cacheDir = caches.appendingPathComponent("HILCache", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    }

    /// Clear all cached files on app launch
    func clearOnLaunch() {
        try? FileManager.default.removeItem(at: cacheDir)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        print("[HILCache] Cleared on launch")
    }

    /// Save image data to cache
    func saveImage(_ data: Data, imageId: String) {
        let url = cacheDir.appendingPathComponent(imageId)
        try? data.write(to: url)
    }

    /// Load cached image data
    func loadImage(imageId: String) -> Data? {
        let url = cacheDir.appendingPathComponent(imageId)
        return try? Data(contentsOf: url)
    }

    /// Check if image is cached
    func hasImage(imageId: String) -> Bool {
        let url = cacheDir.appendingPathComponent(imageId)
        return FileManager.default.fileExists(atPath: url.path)
    }
}
