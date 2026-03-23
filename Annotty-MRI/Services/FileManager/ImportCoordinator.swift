import Foundation
import Combine

/// Notification posted when images have been imported into the project
extension Notification.Name {
    static let didImportImages = Notification.Name("annotty.didImportImages")
}

/// Coordinates image imports from AirDrop (onOpenURL) and Drag & Drop.
/// Buffers incoming URLs with a 500ms debounce so multiple AirDrop files
/// are handled as a single batch.
@MainActor
class ImportCoordinator: ObservableObject {
    /// Number of images imported in the last batch (drives toast display)
    @Published var importedCount: Int = 0
    /// Whether an import is currently in progress
    @Published var isImporting: Bool = false
    /// Controls toast visibility
    @Published var showToast: Bool = false

    private var pendingURLs: [URL] = []
    private var debounceTask: Task<Void, Never>?

    // MARK: - Enqueue

    /// Buffer a received URL for batch processing (called from onOpenURL)
    func enqueue(_ url: URL) {
        pendingURLs.append(url)
        print("[Import] Enqueued: \(url.lastPathComponent) (pending: \(pendingURLs.count))")

        // Reset debounce timer
        debounceTask?.cancel()
        debounceTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 500_000_000) // 500ms
            guard !Task.isCancelled else { return }
            await self?.processPendingURLs()
        }
    }

    // MARK: - Process

    private func processPendingURLs() async {
        guard !pendingURLs.isEmpty else { return }

        let urls = pendingURLs
        pendingURLs.removeAll()

        isImporting = true
        var successCount = 0

        for url in urls {
            let didAccess = url.startAccessingSecurityScopedResource()
            defer {
                if didAccess { url.stopAccessingSecurityScopedResource() }
            }

            do {
                _ = try ProjectFileService.shared.copyImageToProject(url)
                successCount += 1

                // Clean up Inbox copy
                cleanInboxFile(url)
            } catch {
                print("[Import] Failed to copy \(url.lastPathComponent): \(error)")
            }
        }

        isImporting = false

        if successCount > 0 {
            importedCount = successCount
            showToast = true
            print("[Import] Batch complete: \(successCount) images imported")

            // Notify CanvasViewModel to reload
            NotificationCenter.default.post(name: .didImportImages, object: nil,
                                            userInfo: ["count": successCount])

            // Auto-hide toast after 3 seconds
            Task {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                showToast = false
            }
        }
    }

    /// Remove the file from Documents/Inbox/ if it resides there
    private func cleanInboxFile(_ url: URL) {
        let path = url.path
        guard path.contains("/Inbox/") else { return }

        try? FileManager.default.removeItem(at: url)
        print("[Import] Cleaned inbox file: \(url.lastPathComponent)")
    }
}
