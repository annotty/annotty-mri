import SwiftUI

/// Right panel with annotation color, settings button, and HIL controls
struct RightPanelView: View {
    @Binding var annotationColor: Color
    @Binding var isFillMode: Bool
    @Binding var isSmoothMode: Bool
    let isUNetLoading: Bool
    let isUNetProcessing: Bool
    let classNames: [String]
    @Binding var hiddenClassIDs: Set<Int>
    let onSettingsTapped: () -> Void
    // HIL parameters (optional — shown only when HIL is enabled)
    var isHILEnabled: Bool = false
    var isHILSubmitting: Bool = false
    var localImageCount: Int = 0
    var trainingStatus: HILServerClient.TrainingStatus? = nil
    var serverInfo: HILServerClient.ServerInfo? = nil
    var onServerPredictTapped: (() -> Void)? = nil
    var onSubmitTapped: (() -> Void)? = nil
    var onTrainTapped: (() -> Void)? = nil
    var onCancelTrainTapped: (() -> Void)? = nil
    var isSyncingModel: Bool = false
    var modelSource: ModelSource = .bundled
    var onSyncModelTapped: (() -> Void)? = nil
    var hilErrorMessage: String? = nil

    @State private var showTrainAlert = false

    /// Preset colors for annotation (index+1 = classID)
    /// These must match MetalRenderer.classColors exactly
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

    /// Display name for a class (shows class number if unnamed)
    private func displayName(for index: Int) -> String {
        let name = classNames[index]
        return name.isEmpty ? "\(index + 1)" : name
    }

    /// Train alert message with image counts (server labeled/total + local stock)
    private var trainAlertMessage: String {
        let labeled = serverInfo?.labeledImages ?? 0
        let total = serverInfo?.totalImages ?? 0
        return "Images: \(labeled)/\(total) labeled + \(localImageCount) on device\n\nTraining will overwrite the current model. This cannot be undone."
    }

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
        VStack(spacing: 24) {
            // Annotation color section with class names
            VStack(spacing: 6) {
                Text("Classes")
                    .font(.caption)
                    .foregroundColor(.gray)

                // Color + name list (vertical)
                VStack(spacing: 3) {
                    ForEach(Array(presetColors.enumerated()), id: \.offset) { index, color in
                        let classID = index + 1
                        let isHidden = hiddenClassIDs.contains(classID)
                        HStack(spacing: 4) {
                            // Color circle
                            Circle()
                                .fill(color)
                                .frame(width: 18, height: 18)
                                .overlay(
                                    Circle()
                                        .stroke(annotationColor == color ? Color.white : Color.clear, lineWidth: 2)
                                )
                                .opacity(isHidden ? 0.3 : 1.0)

                            // Class name (truncated)
                            Text(displayName(for: index))
                                .font(.caption2)
                                .foregroundColor(annotationColor == color ? .white : .gray)
                                .lineLimit(1)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .opacity(isHidden ? 0.3 : 1.0)

                            // Visibility toggle
                            Image(systemName: isHidden ? "eye.slash" : "eye")
                                .font(.caption2)
                                .foregroundColor(isHidden ? .gray.opacity(0.5) : .gray)
                                .frame(width: 20, height: 20)
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        if hiddenClassIDs.contains(classID) {
                                            hiddenClassIDs.remove(classID)
                                        } else {
                                            hiddenClassIDs.insert(classID)
                                        }
                                    }
                                }
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(annotationColor == color ? Color.white.opacity(0.15) : Color.clear)
                        )
                        .contentShape(Rectangle())
                        .onTapGesture {
                            annotationColor = color
                        }
                    }
                }

                // Fill mode toggle button
                Button(action: {
                    isFillMode.toggle()
                }) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isFillMode ? annotationColor : Color(white: 0.2))
                        .frame(width: 50, height: 50)
                        .overlay(
                            Image(systemName: "drop.fill")
                                .font(.title2)
                                .foregroundColor(isFillMode ? .white : .gray)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(isFillMode ? Color.white : Color.gray.opacity(0.5), lineWidth: isFillMode ? 2 : 1)
                        )
                }
                .buttonStyle(.plain)

                Text(isFillMode ? "Tap to fill" : "Fill")
                    .font(.caption2)
                    .foregroundColor(isFillMode ? .cyan : .gray)

                // Smooth mode toggle button
                Button(action: {
                    isSmoothMode.toggle()
                }) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isSmoothMode ? annotationColor : Color(white: 0.2))
                        .frame(width: 50, height: 50)
                        .overlay(
                            Image(systemName: "waveform.path")
                                .font(.title2)
                                .foregroundColor(isSmoothMode ? .white : .gray)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(isSmoothMode ? Color.white : Color.gray.opacity(0.5), lineWidth: isSmoothMode ? 2 : 1)
                        )
                }
                .buttonStyle(.plain)

                Text(isSmoothMode ? "Trace edge" : "Smooth")
                    .font(.caption2)
                    .foregroundColor(isSmoothMode ? .cyan : .gray)
            }

            Divider()
                .background(Color.gray)

            // Image settings button
            VStack(spacing: 8) {
                Text("Image")
                    .font(.caption)
                    .foregroundColor(.gray)

                Button(action: onSettingsTapped) {
                    VStack(spacing: 4) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.title2)
                        Text("Settings")
                            .font(.caption2)
                    }
                    .foregroundColor(.white)
                    .padding(12)
                    .background(Color(white: 0.25))
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }

            // HIL buttons (shown only when HIL is enabled)
            if isHILEnabled {
                Divider()
                    .background(Color.gray)
                    .padding(.vertical, 4)

                VStack(spacing: 8) {
                    Text("HIL")
                        .font(.caption)
                        .foregroundColor(.orange)

                    // On-device AI Predict button
                    Button(action: { onServerPredictTapped?() }) {
                        VStack(spacing: 4) {
                            if isUNetProcessing {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                                    .frame(width: 24, height: 24)
                            } else {
                                Image(systemName: "brain")
                                    .font(.title3)
                            }
                            Text(isUNetProcessing ? "Predicting" : "AI Predict")
                                .font(.caption2)
                        }
                        .foregroundColor(.white)
                        .padding(10)
                        .frame(maxWidth: .infinity)
                        .background(Color.orange.opacity(0.3))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                    .disabled(isUNetLoading || isUNetProcessing || isHILSubmitting)

                    // Submit & Next button
                    Button(action: { onSubmitTapped?() }) {
                        VStack(spacing: 4) {
                            if isHILSubmitting {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                                    .frame(width: 24, height: 24)
                            } else {
                                Image(systemName: "arrow.right.circle.fill")
                                    .font(.title3)
                            }
                            Text(isHILSubmitting ? "Submitting" : "Submit")
                                .font(.caption2)
                        }
                        .foregroundColor(.white)
                        .padding(10)
                        .frame(maxWidth: .infinity)
                        .background(Color.green.opacity(0.3))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                    .disabled(isUNetProcessing || isHILSubmitting)

                    // Train button (with confirmation alert)
                    Button(action: { showTrainAlert = true }) {
                        VStack(spacing: 4) {
                            Image(systemName: "brain")
                                .font(.title3)
                            Text("Train")
                                .font(.caption2)
                        }
                        .foregroundColor(.white)
                        .padding(10)
                        .frame(maxWidth: .infinity)
                        .background(Color.purple.opacity(0.3))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                    .disabled(trainingStatus?.status == "running")
                    .alert("Start Training?", isPresented: $showTrainAlert) {
                        Button("Cancel", role: .cancel) {}
                        Button("Start", role: .destructive) {
                            onTrainTapped?()
                        }
                    } message: {
                        Text(trainAlertMessage)
                    }

                    // Training status inline
                    TrainingStatusView(status: trainingStatus)

                    // Error message (e.g. GPU busy)
                    if let errorMsg = hilErrorMessage {
                        Text(errorMsg)
                            .font(.caption2)
                            .foregroundColor(.red)
                            .lineLimit(2)
                            .frame(maxWidth: .infinity)
                            .padding(6)
                            .background(Color.red.opacity(0.15))
                            .cornerRadius(6)
                    }

                    // Cancel button (shown only during training)
                    if trainingStatus?.status == "running" {
                        Button(action: { onCancelTrainTapped?() }) {
                            HStack(spacing: 4) {
                                Image(systemName: "xmark.circle")
                                    .font(.caption2)
                                Text("Cancel")
                                    .font(.caption2)
                            }
                            .foregroundColor(.red)
                            .padding(6)
                            .frame(maxWidth: .infinity)
                            .background(Color.red.opacity(0.15))
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                    }

                    Divider()
                        .background(Color.gray)
                        .padding(.vertical, 2)

                    // Model source indicator
                    HStack(spacing: 4) {
                        Circle()
                            .fill(modelSource == .downloaded ? Color.green : Color.orange)
                            .frame(width: 8, height: 8)
                        Text(modelSource.rawValue)
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }

                    // Sync Model button
                    Button(action: { onSyncModelTapped?() }) {
                        VStack(spacing: 4) {
                            if isSyncingModel {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                                    .frame(width: 24, height: 24)
                            } else {
                                Image(systemName: "arrow.down.circle")
                                    .font(.title3)
                            }
                            Text(isSyncingModel ? "Syncing..." : "Sync Model")
                                .font(.caption2)
                        }
                        .foregroundColor(.white)
                        .padding(10)
                        .frame(maxWidth: .infinity)
                        .background(Color.cyan.opacity(0.3))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                    .disabled(isSyncingModel)
                }
            }

            Spacer()
                .frame(height: 20)
        }
        .padding(.top, 20)
        }
        .frame(maxHeight: .infinity)
        .background(Color(white: 0.1))
    }
}

// MARK: - Preview

#Preview {
    RightPanelView(
        annotationColor: .constant(Color(red: 1, green: 0, blue: 0)),
        isFillMode: .constant(false),
        isSmoothMode: .constant(false),
        isUNetLoading: false,
        isUNetProcessing: false,
        classNames: ["iris", "eyelid", "sclera", "pupil", "", "", "", ""],
        hiddenClassIDs: .constant([]),
        onSettingsTapped: {}
    )
    .frame(width: 120, height: 600)
}
