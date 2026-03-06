import Foundation
import UIKit
import Combine

/// Handles automatic saving of annotations
/// Triggers:
/// - Stroke end (500ms debounce)
/// - Image navigation
/// - App backgrounding
class AutoSaveService: ObservableObject {
    // MARK: - Singleton

    static let shared = AutoSaveService()

    // MARK: - Configuration

    /// Debounce delay for stroke-triggered saves
    let saveDebounceDelay: TimeInterval = 0.5

    // MARK: - State

    @Published private(set) var isSaving = false
    @Published private(set) var lastSaveTime: Date?
    @Published var hasUnsavedChanges = false

    // MARK: - Private

    private var saveWorkItem: DispatchWorkItem?
    private let saveQueue = DispatchQueue(label: "com.annoty.autosave", qos: .utility)
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Callbacks

    /// Called to perform the actual save operation
    var onSave: (() -> Void)?

    // MARK: - Initialization

    private init() {
        setupAppLifecycleObservers()
    }

    // MARK: - App Lifecycle

    private func setupAppLifecycleObservers() {
        NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)
            .sink { [weak self] _ in
                self?.saveImmediately()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { [weak self] _ in
                self?.saveImmediately()
            }
            .store(in: &cancellables)
    }

    // MARK: - Save Operations

    /// Schedule a debounced save (for stroke end)
    func scheduleSave() {
        hasUnsavedChanges = true

        // Cancel any pending save
        saveWorkItem?.cancel()

        // Schedule new save
        saveWorkItem = DispatchWorkItem { [weak self] in
            self?.performSave()
        }

        saveQueue.asyncAfter(
            deadline: .now() + saveDebounceDelay,
            execute: saveWorkItem!
        )
    }

    /// Save immediately (for image navigation, app backgrounding)
    func saveImmediately() {
        saveWorkItem?.cancel()

        if hasUnsavedChanges {
            performSave()
        }
    }

    /// Perform the actual save
    private func performSave() {
        DispatchQueue.main.async { [weak self] in
            self?.isSaving = true
        }

        // Call the save callback
        onSave?()

        DispatchQueue.main.async { [weak self] in
            self?.isSaving = false
            self?.hasUnsavedChanges = false
            self?.lastSaveTime = Date()
        }
    }

    /// Mark that changes have been made
    func markDirty() {
        hasUnsavedChanges = true
    }

    /// Reset state (for new image)
    func reset() {
        saveWorkItem?.cancel()
        hasUnsavedChanges = false
    }
}
