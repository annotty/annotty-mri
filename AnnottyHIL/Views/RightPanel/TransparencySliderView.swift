import SwiftUI

/// Vertical slider for adjusting image transparency
struct TransparencySliderView: View {
    @Binding var value: Float

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                // Track background
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(white: 0.3))
                    .frame(width: 8)

                // Filled portion
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.white.opacity(0.6))
                    .frame(width: 8, height: geometry.size.height * CGFloat(value))

                // Thumb
                Circle()
                    .fill(Color.white)
                    .frame(width: 24, height: 24)
                    .shadow(radius: 2)
                    .offset(y: -geometry.size.height * CGFloat(value) + 12)
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .bottom)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { gestureValue in
                        let newValue = 1.0 - Float(gestureValue.location.y / geometry.size.height)
                        value = min(max(newValue, 0), 1)
                    }
            )
        }
        .frame(width: 44)
    }
}

// MARK: - Preview

#Preview {
    TransparencySliderView(value: .constant(0.7))
        .frame(width: 44, height: 200)
        .background(Color(white: 0.1))
}
