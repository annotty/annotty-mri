import SwiftUI

/// Overlay view that shows the smooth stroke path while drawing
/// Displays a semi-transparent stroke to indicate the area being smoothed
struct SmoothStrokeOverlay: View {
    let points: [CGPoint]
    let brushRadius: CGFloat

    /// Stroke color (semi-transparent white for visibility on any background)
    private let strokeColor = Color.white.opacity(0.5)

    var body: some View {
        Canvas { context, size in
            guard points.count >= 2 else { return }

            // Draw the stroke path with brush width
            var path = Path()
            path.move(to: points[0])

            for i in 1..<points.count {
                path.addLine(to: points[i])
            }

            // Draw semi-transparent white stroke (no border)
            context.stroke(
                path,
                with: .color(strokeColor),
                style: StrokeStyle(
                    lineWidth: brushRadius * 2,
                    lineCap: .round,
                    lineJoin: .round
                )
            )
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Preview

#Preview {
    ZStack {
        Color.gray
        SmoothStrokeOverlay(
            points: [
                CGPoint(x: 100, y: 100),
                CGPoint(x: 150, y: 120),
                CGPoint(x: 200, y: 100),
                CGPoint(x: 250, y: 150)
            ],
            brushRadius: 20
        )
    }
    .frame(width: 400, height: 300)
}
