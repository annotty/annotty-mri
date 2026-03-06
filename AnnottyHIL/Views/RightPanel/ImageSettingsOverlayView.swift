import SwiftUI

/// Slide-in overlay panel for image and mask settings
/// Contains contrast, brightness, mask opacity, and class name editing
struct ImageSettingsOverlayView: View {
    @Binding var isPresented: Bool
    @Binding var imageContrast: Float
    @Binding var imageBrightness: Float
    @Binding var maskFillAlpha: Float
    @Binding var maskEdgeAlpha: Float
    @Binding var smoothKernelSize: Int
    @Binding var classNames: [String]
    var onClearClassNames: () -> Void
    var onDeleteAllFiles: () -> Void

    /// Alert state for 2-step delete confirmation
    @State private var showDeleteFirstAlert = false
    @State private var showDeleteFinalAlert = false

    /// Preset colors for class editing display (must match MetalRenderer.classColors)
    private let presetColors: [Color] = [
        Color(red: 1, green: 0, blue: 0),        // 1: red
        Color(red: 1, green: 0.5, blue: 0),      // 2: orange
        Color(red: 1, green: 1, blue: 0),        // 3: yellow
        Color(red: 0, green: 1, blue: 0),        // 4: green
        Color(red: 0, green: 1, blue: 1),        // 5: cyan
        Color(red: 0, green: 0, blue: 1),        // 6: blue
        Color(red: 0.5, green: 0, blue: 1),      // 7: purple
        Color(red: 1, green: 0.4, blue: 0.7)     // 8: pink
    ]

    var body: some View {
        HStack(spacing: 0) {
            // Pass-through area (allows drawing on canvas)
            Color.clear
                .allowsHitTesting(false)

            // Settings panel
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    HStack {
                        Text("Settings")
                            .font(.headline)
                            .foregroundColor(.white)
                        Spacer()
                        Button(action: {
                            withAnimation(.easeInOut(duration: 0.25)) {
                                isPresented = false
                            }
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title2)
                                .foregroundColor(.gray)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 16)

                    // Image section
                    VStack(spacing: 12) {
                        Text("Image")
                            .font(.caption)
                            .foregroundColor(.gray)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 16)

                        SettingsSliderView(
                            title: "Contrast",
                            value: $imageContrast,
                            range: 0...2,
                            displayFormatter: { Int($0 * 100) },
                            displaySuffix: "%"
                        )

                        SettingsSliderView(
                            title: "Brightness",
                            value: $imageBrightness,
                            range: -1...1,
                            displayFormatter: { Int($0 * 100) },
                            displaySuffix: ""
                        )

                        SettingsSliderView(
                            title: "Mask Fill",
                            value: $maskFillAlpha,
                            range: 0...1,
                            displayFormatter: { Int($0 * 100) },
                            displaySuffix: "%"
                        )

                        SettingsSliderView(
                            title: "Edge Fill",
                            value: $maskEdgeAlpha,
                            range: 0...1,
                            displayFormatter: { Int($0 * 100) },
                            displaySuffix: "%"
                        )

                        // Reset button
                        Button(action: resetImageSettings) {
                            HStack {
                                Image(systemName: "arrow.counterclockwise")
                                Text("Reset")
                            }
                            .font(.caption)
                            .foregroundColor(.cyan)
                            .padding(.vertical, 6)
                            .padding(.horizontal, 12)
                            .background(Color(white: 0.2))
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                    }

                    Divider()
                        .background(Color.gray.opacity(0.5))
                        .padding(.horizontal, 16)

                    // Smooth settings section
                    VStack(spacing: 12) {
                        Text("Smooth Tool")
                            .font(.caption)
                            .foregroundColor(.gray)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 16)

                        IntSettingsSliderView(
                            title: "Kernel Size",
                            value: $smoothKernelSize,
                            range: 7...31,
                            step: 2  // Ensure odd numbers
                        )

                        Text("Larger = smoother edges")
                            .font(.caption2)
                            .foregroundColor(.gray.opacity(0.7))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 16)
                    }

                    Divider()
                        .background(Color.gray.opacity(0.5))
                        .padding(.horizontal, 16)

                    // Class names section
                    VStack(spacing: 8) {
                        HStack {
                            Text("Class Names")
                                .font(.caption)
                                .foregroundColor(.gray)
                            Spacer()
                            Button(action: onClearClassNames) {
                                HStack(spacing: 4) {
                                    Image(systemName: "trash")
                                        .font(.caption2)
                                    Text("Clear")
                                        .font(.caption2)
                                }
                                .foregroundColor(.red.opacity(0.8))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color(white: 0.2))
                                .cornerRadius(4)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 16)

                        VStack(spacing: 6) {
                            ForEach(0..<8, id: \.self) { index in
                                HStack(spacing: 8) {
                                    // Color indicator
                                    Circle()
                                        .fill(presetColors[index])
                                        .frame(width: 16, height: 16)

                                    // Editable text field
                                    TextField("Class \(index + 1)", text: $classNames[index])
                                        .font(.caption)
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 6)
                                        .background(Color(white: 0.2))
                                        .cornerRadius(6)
                                }
                                .padding(.horizontal, 16)
                            }
                        }
                    }

                    Divider()
                        .background(Color.gray.opacity(0.5))
                        .padding(.horizontal, 16)

                    // Toolbar section
                    Divider()
                        .background(Color.gray.opacity(0.5))
                        .padding(.horizontal, 16)

                    // Danger zone - Delete all files
                    VStack(spacing: 8) {
                        Text("Danger Zone")
                            .font(.caption)
                            .foregroundColor(.red.opacity(0.7))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 16)

                        Button(action: {
                            showDeleteFirstAlert = true
                        }) {
                            HStack {
                                Image(systemName: "trash.fill")
                                Text("Delete All Files")
                            }
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color.red.opacity(0.8))
                            .cornerRadius(8)
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 16)

                        Text("Removes all images, annotations, and labels")
                            .font(.caption2)
                            .foregroundColor(.gray.opacity(0.7))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 16)
                    }

                    Spacer()
                        .frame(height: 20)
                }
            }
            .frame(width: 220)
            .background(Color(white: 0.12))
            .alert("Delete All Files?", isPresented: $showDeleteFirstAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Continue", role: .destructive) {
                    showDeleteFinalAlert = true
                }
            } message: {
                Text("This will delete all images, annotations, and exported labels in this project.")
            }
            .alert("Are you absolutely sure?", isPresented: $showDeleteFinalAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Delete Everything", role: .destructive) {
                    onDeleteAllFiles()
                    withAnimation(.easeInOut(duration: 0.25)) {
                        isPresented = false
                    }
                }
            } message: {
                Text("This action cannot be undone. All files will be permanently removed.")
            }
        }
    }

    private func resetImageSettings() {
        withAnimation(.easeOut(duration: 0.2)) {
            imageContrast = 1.0
            imageBrightness = 0.0
            maskFillAlpha = 0.5
            maskEdgeAlpha = 1.0
        }
    }
}

// MARK: - Settings Slider Component

/// Horizontal slider with label and value display
struct SettingsSliderView: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let displayFormatter: (Float) -> Int
    let displaySuffix: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Title and value
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text("\(displayFormatter(value))\(displaySuffix)")
                    .font(.caption)
                    .foregroundColor(.white)
                    .monospacedDigit()
            }

            // Slider track
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Track background
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(white: 0.3))
                        .frame(height: 8)

                    // Filled portion (handle center position for value)
                    let normalizedValue = (value - range.lowerBound) / (range.upperBound - range.lowerBound)
                    let centerPosition = range.contains(0) ? -range.lowerBound / (range.upperBound - range.lowerBound) : 0

                    if range.contains(0) {
                        // Bidirectional slider (like brightness: -1 to 1)
                        let fillStart = min(normalizedValue, centerPosition)
                        let fillWidth = abs(normalizedValue - centerPosition)
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.cyan.opacity(0.6))
                            .frame(width: geometry.size.width * CGFloat(fillWidth), height: 8)
                            .offset(x: geometry.size.width * CGFloat(fillStart))
                    } else {
                        // Unidirectional slider (like contrast: 0 to 2)
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.cyan.opacity(0.6))
                            .frame(width: geometry.size.width * CGFloat(normalizedValue), height: 8)
                    }

                    // Thumb
                    Circle()
                        .fill(Color.white)
                        .frame(width: 20, height: 20)
                        .shadow(radius: 2)
                        .offset(x: geometry.size.width * CGFloat(normalizedValue) - 10)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { gestureValue in
                            let normalized = Float(gestureValue.location.x / geometry.size.width)
                            let clamped = min(max(normalized, 0), 1)
                            value = range.lowerBound + clamped * (range.upperBound - range.lowerBound)
                        }
                )
            }
            .frame(height: 20)
        }
        .padding(.horizontal, 16)
    }
}

// MARK: - Int Settings Slider Component

/// Horizontal slider for integer values with step control
struct IntSettingsSliderView: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Title and value
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text("\(value)px")
                    .font(.caption)
                    .foregroundColor(.white)
                    .monospacedDigit()
            }

            // Slider track
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Track background
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(white: 0.3))
                        .frame(height: 8)

                    // Filled portion
                    let normalizedValue = Float(value - range.lowerBound) / Float(range.upperBound - range.lowerBound)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.orange.opacity(0.6))
                        .frame(width: geometry.size.width * CGFloat(normalizedValue), height: 8)

                    // Thumb
                    Circle()
                        .fill(Color.white)
                        .frame(width: 20, height: 20)
                        .shadow(radius: 2)
                        .offset(x: geometry.size.width * CGFloat(normalizedValue) - 10)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { gestureValue in
                            let normalized = gestureValue.location.x / geometry.size.width
                            let clamped = min(max(normalized, 0), 1)
                            let rawValue = Float(range.lowerBound) + Float(clamped) * Float(range.upperBound - range.lowerBound)
                            // Snap to step
                            let stepped = Int(round(rawValue / Float(step))) * step
                            value = min(max(stepped, range.lowerBound), range.upperBound)
                        }
                )
            }
            .frame(height: 20)
        }
        .padding(.horizontal, 16)
    }
}

// MARK: - Preview

#Preview {
    ZStack {
        Color(white: 0.15)
        ImageSettingsOverlayView(
            isPresented: .constant(true),
            imageContrast: .constant(1.0),
            imageBrightness: .constant(0.0),
            maskFillAlpha: .constant(0.5),
            maskEdgeAlpha: .constant(1.0),
            smoothKernelSize: .constant(21),
            classNames: .constant(["iris", "eyelid", "sclera", "pupil", "", "", "", ""]),
            onClearClassNames: {},
            onDeleteAllFiles: {}
        )
    }
    .frame(width: 400, height: 600)
}
