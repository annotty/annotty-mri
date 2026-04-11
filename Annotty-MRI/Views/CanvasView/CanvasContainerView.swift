import SwiftUI

/// Container view that combines MetalCanvasView with UIKit gesture handling
struct CanvasContainerView: View {
    @ObservedObject var viewModel: CanvasViewModel

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Metal canvas with UIKit touch handling
                if let renderer = viewModel.renderer {
                    MetalCanvasView(
                        renderer: renderer,
                        gestureCoordinator: viewModel.gestureCoordinator
                    )
                } else {
                    // Placeholder when no renderer
                    Rectangle()
                        .fill(Color(white: 0.2))
                        .overlay(
                            Text("No image loaded")
                                .foregroundColor(.gray)
                        )
                }

                // Brush preview circle (when drawing)
                if viewModel.isDrawing {
                    Circle()
                        .stroke(viewModel.isPainting ? Color.green : Color.red, lineWidth: 2)
                        .frame(width: viewModel.brushPreviewSize, height: viewModel.brushPreviewSize)
                        .position(viewModel.lastDrawPoint)
                        .allowsHitTesting(false)
                }

                // Fill mode indicator overlay
                if viewModel.isFillMode {
                    Rectangle()
                        .fill(Color.cyan.opacity(0.05))
                        .allowsHitTesting(false)

                    // Fill mode badge
                    VStack {
                        HStack {
                            Spacer()
                            Text("Fill Mode")
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.cyan.opacity(0.8))
                                .foregroundColor(.white)
                                .cornerRadius(4)
                                .padding(8)
                        }
                        Spacer()
                    }
                    .allowsHitTesting(false)
                }

                // Smooth mode indicator overlay
                if viewModel.isSmoothMode {
                    // Smooth mode badge
                    VStack {
                        HStack {
                            Spacer()
                            Text("Smooth: Trace Edge")
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.orange.opacity(0.8))
                                .foregroundColor(.white)
                                .cornerRadius(4)
                                .padding(8)
                        }
                        Spacer()
                    }
                    .allowsHitTesting(false)
                }

                // Smooth stroke path overlay
                if !viewModel.smoothStrokePoints.isEmpty {
                    // Reuse brushPreviewSize so the overlay's lineWidth matches the
                    // actual mask stroke width in SwiftUI points (handles Retina/iPad).
                    SmoothStrokeOverlay(
                        points: viewModel.smoothStrokePoints,
                        brushRadius: viewModel.brushPreviewSize
                    )
                    .allowsHitTesting(false)
                }
            }
            .onAppear {
                viewModel.updateViewSize(geometry.size)
            }
            .onChange(of: geometry.size) { _, newSize in
                viewModel.updateViewSize(newSize)
            }
        }
    }
}

// MARK: - Preview

#Preview {
    CanvasContainerView(viewModel: CanvasViewModel())
}
