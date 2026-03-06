import Foundation
import CoreGraphics
import simd

/// Manages pan, zoom, and rotation transformations for the canvas
/// Provides conversion between screen coordinates and internal mask coordinates
struct CanvasTransform {
    /// Translation offset in screen points
    var translation: CGPoint = .zero

    /// Zoom scale factor (1.0 = original size)
    var scale: CGFloat = 1.0

    /// Rotation angle in radians (free rotation, no snap)
    var rotation: CGFloat = 0.0

    /// The mask scale factor (from image to internal mask, typically 2.0 with 4096 max clamp)
    var maskScaleFactor: Float = 2.0

    /// Minimum allowed zoom scale
    static let minScale: CGFloat = 0.1

    /// Maximum allowed zoom scale
    static let maxScale: CGFloat = 10.0

    /// Combined transformation matrix
    var matrix: CGAffineTransform {
        CGAffineTransform.identity
            .translatedBy(x: translation.x, y: translation.y)
            .rotated(by: rotation)
            .scaledBy(x: scale, y: scale)
    }

    /// Inverse of the combined transformation matrix
    /// Returns identity transform if inversion fails (degenerate matrix)
    var inverseMatrix: CGAffineTransform {
        let inverted = matrix.inverted()
        // Check if inversion produced valid values (not NaN or infinite)
        if inverted.a.isFinite && inverted.b.isFinite &&
           inverted.c.isFinite && inverted.d.isFinite &&
           inverted.tx.isFinite && inverted.ty.isFinite {
            return inverted
        }
        // Fallback to identity if matrix is degenerate
        print("[Transform] Degenerate matrix detected, returning identity")
        return .identity
    }

    /// Convert screen point to image coordinates
    func screenToImage(_ screenPoint: CGPoint) -> CGPoint {
        screenPoint.applying(inverseMatrix)
    }

    /// Convert image coordinates to screen point
    func imageToScreen(_ imagePoint: CGPoint) -> CGPoint {
        imagePoint.applying(matrix)
    }

    /// Convert screen point to internal mask coordinates
    /// Uses the dynamic mask scale factor (2x with 4096px max clamp)
    /// Returns safe values if coordinates become invalid
    func screenToMask(_ screenPoint: CGPoint) -> CGPoint {
        let imagePoint = screenToImage(screenPoint)
        let maskPoint = CGPoint(
            x: imagePoint.x * CGFloat(maskScaleFactor),
            y: imagePoint.y * CGFloat(maskScaleFactor)
        )
        // Validate result - return .zero if NaN/infinite
        if maskPoint.x.isFinite && maskPoint.y.isFinite {
            return maskPoint
        }
        return .zero
    }

    /// Convert internal mask coordinates to screen point
    func maskToScreen(_ maskPoint: CGPoint) -> CGPoint {
        let imagePoint = CGPoint(
            x: maskPoint.x / CGFloat(maskScaleFactor),
            y: maskPoint.y / CGFloat(maskScaleFactor)
        )
        return imageToScreen(imagePoint)
    }

    /// Convert screen radius to mask radius
    /// Zoom-independent: radius in mask coordinates stays constant regardless of zoom
    func screenRadiusToMask(_ screenRadius: CGFloat) -> Float {
        // Radius is zoom-independent, so we only apply mask scale factor
        Float(screenRadius) * maskScaleFactor
    }

    /// Apply pan gesture delta
    mutating func applyPan(delta: CGPoint) {
        // Guard against invalid inputs
        guard delta.x.isFinite && delta.y.isFinite else { return }
        translation.x += delta.x
        translation.y += delta.y
    }

    /// Apply pinch gesture scale factor around a center point
    mutating func applyPinch(scaleFactor: CGFloat, center: CGPoint) {
        // Guard against invalid inputs
        guard scaleFactor.isFinite && scaleFactor > 0 &&
              center.x.isFinite && center.y.isFinite else {
            return
        }

        let newScale = (scale * scaleFactor).clamped(to: Self.minScale...Self.maxScale)
        let actualScaleFactor = newScale / scale

        // Adjust translation to keep center point fixed
        let newTransX = center.x - (center.x - translation.x) * actualScaleFactor
        let newTransY = center.y - (center.y - translation.y) * actualScaleFactor

        // Only apply if results are valid
        if newTransX.isFinite && newTransY.isFinite {
            translation.x = newTransX
            translation.y = newTransY
            scale = newScale
        }
    }

    /// Apply rotation gesture angle delta around a center point
    mutating func applyRotation(angleDelta: CGFloat, center: CGPoint) {
        // Guard against invalid inputs
        guard angleDelta.isFinite && center.x.isFinite && center.y.isFinite else { return }

        // Free rotation, no snapping
        rotation += angleDelta

        // Adjust translation to rotate around center point
        let cosA = cos(angleDelta)
        let sinA = sin(angleDelta)
        let dx = translation.x - center.x
        let dy = translation.y - center.y

        let newTransX = center.x + dx * cosA - dy * sinA
        let newTransY = center.y + dx * sinA + dy * cosA

        // Only apply if results are valid
        if newTransX.isFinite && newTransY.isFinite {
            translation.x = newTransX
            translation.y = newTransY
        }
    }

    /// Reset to identity transform
    mutating func reset() {
        translation = .zero
        scale = 1.0
        rotation = 0.0
    }

    /// Fit image to view (aspect fit, centered)
    /// - Parameters:
    ///   - imageSize: Size of the image in pixels
    ///   - viewSize: Size of the view in pixels
    mutating func fitToView(imageSize: CGSize, viewSize: CGSize) {
        guard imageSize.width > 0 && imageSize.height > 0 &&
              viewSize.width > 0 && viewSize.height > 0 else {
            reset()
            return
        }

        // Calculate scale to fit image in view (aspect fit)
        let scaleX = viewSize.width / imageSize.width
        let scaleY = viewSize.height / imageSize.height
        let fitScale = min(scaleX, scaleY)

        // Calculate translation to center the image
        let scaledWidth = imageSize.width * fitScale
        let scaledHeight = imageSize.height * fitScale
        let offsetX = (viewSize.width - scaledWidth) / 2
        let offsetY = (viewSize.height - scaledHeight) / 2

        // Apply fit transform
        scale = fitScale
        translation = CGPoint(x: offsetX, y: offsetY)
        rotation = 0.0
    }

    /// Convert to simd matrix for Metal shaders
    func toSimdMatrix() -> simd_float3x3 {
        let m = matrix
        return simd_float3x3(
            simd_float3(Float(m.a), Float(m.b), 0),
            simd_float3(Float(m.c), Float(m.d), 0),
            simd_float3(Float(m.tx), Float(m.ty), 1)
        )
    }

    /// Inverse simd matrix for Metal shaders
    func toInverseSimdMatrix() -> simd_float3x3 {
        let m = inverseMatrix
        return simd_float3x3(
            simd_float3(Float(m.a), Float(m.b), 0),
            simd_float3(Float(m.c), Float(m.d), 0),
            simd_float3(Float(m.tx), Float(m.ty), 1)
        )
    }
}

// MARK: - CGFloat Extension

extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        if self < range.lowerBound { return range.lowerBound }
        if self > range.upperBound { return range.upperBound }
        return self
    }
}
