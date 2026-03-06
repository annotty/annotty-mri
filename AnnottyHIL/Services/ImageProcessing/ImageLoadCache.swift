import Foundation

/// Thread-safe LRU cache for preloaded image + annotation data.
/// Stores the 3 most recent entries (current, previous, next)
/// so adjacent-image navigation can skip file I/O and decode entirely.
actor ImageLoadCache {
    struct Entry {
        let imageData: TextureManager.PreloadedImageData
        let maskData: [UInt8]?
    }

    private var entries: [URL: Entry] = [:]
    private var accessOrder: [URL] = []
    private let maxEntries: Int

    init(maxEntries: Int = 3) {
        self.maxEntries = maxEntries
    }

    func get(for url: URL) -> Entry? {
        guard let entry = entries[url] else { return nil }
        // Move to end of access order (most recently used)
        if let idx = accessOrder.firstIndex(of: url) {
            accessOrder.remove(at: idx)
        }
        accessOrder.append(url)
        return entry
    }

    func put(url: URL, entry: Entry) {
        // Evict oldest if at capacity
        if entries[url] == nil && entries.count >= maxEntries {
            if let oldest = accessOrder.first {
                entries.removeValue(forKey: oldest)
                accessOrder.removeFirst()
            }
        }
        entries[url] = entry
        if let idx = accessOrder.firstIndex(of: url) {
            accessOrder.remove(at: idx)
        }
        accessOrder.append(url)
    }

    func invalidate(for url: URL) {
        entries.removeValue(forKey: url)
        accessOrder.removeAll { $0 == url }
    }

    func clear() {
        entries.removeAll()
        accessOrder.removeAll()
    }
}
