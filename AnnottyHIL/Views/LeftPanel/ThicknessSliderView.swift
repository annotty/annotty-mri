import SwiftUI

/// Vertical slider for brush thickness with logarithmic scale
/// Range: 1-100 pixels (in mask coordinates)
struct ThicknessSliderView: View {
    @Binding var radius: Float

    /// Minimum radius
    static let minRadius: Float = 1

    /// Maximum radius
    static let maxRadius: Float = 100

    /// Convert linear position (0-1) to logarithmic radius
    private func positionToRadius(_ position: Float) -> Float {
        let logMin = log(Self.minRadius)
        let logMax = log(Self.maxRadius)
        let logValue = logMin + position * (logMax - logMin)
        return exp(logValue)
    }

    /// Convert logarithmic radius to linear position (0-1)
    private func radiusToPosition(_ radius: Float) -> Float {
        let logMin = log(Self.minRadius)
        let logMax = log(Self.maxRadius)
        let logValue = log(max(radius, Self.minRadius))
        return (logValue - logMin) / (logMax - logMin)
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                // Track background
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(white: 0.3))
                    .frame(width: 8)

                // Filled portion
                RoundedRectangle(cornerRadius: 4)
                    .fill(
                        LinearGradient(
                            colors: [.blue, .cyan],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    )
                    .frame(width: 8, height: geometry.size.height * CGFloat(radiusToPosition(radius)))

                // Thumb with size preview
                ZStack {
                    // Outer circle (thumb)
                    Circle()
                        .fill(Color.white)
                        .frame(width: 32, height: 32)
                        .shadow(radius: 2)

                    // Inner circle (brush preview)
                    Circle()
                        .fill(Color.blue)
                        .frame(
                            width: min(CGFloat(radius) / 5 + 4, 28),
                            height: min(CGFloat(radius) / 5 + 4, 28)
                        )
                }
                .offset(y: -geometry.size.height * CGFloat(radiusToPosition(radius)) + 16)
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .bottom)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { gestureValue in
                        let position = 1.0 - Float(gestureValue.location.y / geometry.size.height)
                        let clampedPosition = min(max(position, 0), 1)
                        radius = positionToRadius(clampedPosition)
                    }
            )
        }
        .frame(width: 44)
    }
}

// MARK: - Preview

#Preview {
    VStack {
        ThicknessSliderView(radius: .constant(10))
            .frame(width: 44, height: 300)

        ThicknessSliderView(radius: .constant(50))
            .frame(width: 44, height: 300)
    }
    .padding()
    .background(Color(white: 0.1))
}
