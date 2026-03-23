import Foundation
import Combine

/// Custom undo manager for annotation operations
/// Uses bbox patch method for memory efficiency
class AnnotationUndoManager: ObservableObject {
    // MARK: - Stacks

    private var undoStack: [UndoAction] = []
    private var redoStack: [UndoAction] = []

    // MARK: - Configuration

    /// Maximum number of undo levels
    let maxUndoLevels = 50

    /// Maximum total memory for undo stack (100MB)
    let maxMemoryBytes = 100 * 1024 * 1024

    // MARK: - Published State

    @Published private(set) var canUndo: Bool = false
    @Published private(set) var canRedo: Bool = false

    // MARK: - Memory Tracking

    private var totalMemoryUsage: Int {
        undoStack.reduce(0) { $0 + $1.memorySize }
    }

    // MARK: - Operations

    /// Push a new undo action
    func pushUndo(_ action: UndoAction) {
        undoStack.append(action)

        // Clear redo stack on new action
        redoStack.removeAll()

        // Prune if needed
        pruneIfNeeded()

        updateState()
    }

    /// Perform undo and return the action
    func undo() -> UndoAction? {
        guard let action = undoStack.popLast() else { return nil }

        redoStack.append(action)
        updateState()

        return action
    }

    /// Perform redo and return the action
    func redo() -> UndoAction? {
        guard let action = redoStack.popLast() else { return nil }

        undoStack.append(action)
        updateState()

        return action
    }

    /// Store the current mask state on the last redo action (called during undo)
    func setNewPatchOnLastRedo(_ data: Data) {
        guard !redoStack.isEmpty else { return }
        redoStack[redoStack.count - 1].newPatch = data
    }

    /// Clear all undo/redo history
    func clear() {
        undoStack.removeAll()
        redoStack.removeAll()
        updateState()
    }

    // MARK: - Private Methods

    private func pruneIfNeeded() {
        // Remove oldest entries if exceeding level limit
        while undoStack.count > maxUndoLevels {
            undoStack.removeFirst()
        }

        // Remove oldest entries if exceeding memory limit
        while totalMemoryUsage > maxMemoryBytes && !undoStack.isEmpty {
            undoStack.removeFirst()
        }
    }

    private func updateState() {
        canUndo = !undoStack.isEmpty
        canRedo = !redoStack.isEmpty
    }
}

