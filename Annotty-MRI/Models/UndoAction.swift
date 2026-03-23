import Foundation
import CoreGraphics

/// Represents a single undoable action using the bbox patch method
/// Stores the previous state of the affected region before modification
struct UndoAction {
    /// Class ID of the mask that was modified
    let classID: Int

    /// Bounding box of the affected region in internal mask coordinates
    let bbox: CGRect

    /// Previous mask data for the bbox region (uncompressed UInt8 array)
    /// MVP: Stored uncompressed. Compression may be considered in future versions.
    let previousPatch: Data

    /// New mask data after modification (captured during undo for redo support)
    var newPatch: Data?

    /// Timestamp when this action was created
    let timestamp: Date

    /// Width of the patch in pixels
    var patchWidth: Int {
        Int(bbox.width)
    }

    /// Height of the patch in pixels
    var patchHeight: Int {
        Int(bbox.height)
    }

    /// Estimated memory usage of this action in bytes
    var memorySize: Int {
        previousPatch.count + (newPatch?.count ?? 0) + MemoryLayout<UndoAction>.size
    }

    init(classID: Int, bbox: CGRect, previousPatch: Data) {
        self.classID = classID
        self.bbox = bbox
        self.previousPatch = previousPatch
        self.timestamp = Date()
    }
}
