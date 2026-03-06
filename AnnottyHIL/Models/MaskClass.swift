import SwiftUI

/// Represents a segmentation class with its associated colors and metadata
struct MaskClass: Identifiable, Equatable {
    let id: Int
    /// Original color from loaded annotation (used as class identifier)
    let originalColor: Color
    /// Display color for overlay (can be changed by user, doesn't affect export)
    var displayColor: Color
    /// User-visible name
    var name: String

    /// Maximum number of classes allowed
    static let maxClasses = 8

    init(id: Int, originalColor: Color, displayColor: Color? = nil, name: String? = nil) {
        self.id = id
        self.originalColor = originalColor
        self.displayColor = displayColor ?? originalColor.opacity(0.5)
        self.name = name ?? "Class \(id)"
    }

    /// Create from RGB values (0-255)
    init(id: Int, r: UInt8, g: UInt8, b: UInt8) {
        self.id = id
        let color = Color(
            red: Double(r) / 255.0,
            green: Double(g) / 255.0,
            blue: Double(b) / 255.0
        )
        self.originalColor = color
        self.displayColor = color.opacity(0.5)
        self.name = "Class \(id)"
    }
}
