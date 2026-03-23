import SwiftUI
import MetalKit

/// UIKit-based canvas view that properly handles Apple Pencil input
/// Wraps MTKView and integrates GestureCoordinator for input separation
class DrawingCanvasUIView: MTKView {
    var gestureCoordinator: GestureCoordinator?
    var renderer: MetalRenderer?

    override init(frame: CGRect, device: MTLDevice?) {
        super.init(frame: frame, device: device)
        setup()
    }

    required init(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        isMultipleTouchEnabled = true
        isUserInteractionEnabled = true
    }

    // MARK: - Touch Handling

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        gestureCoordinator?.touchesBegan(touches, with: event, in: self)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        gestureCoordinator?.touchesMoved(touches, with: event, in: self)
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        gestureCoordinator?.touchesEnded(touches, with: event, in: self)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        gestureCoordinator?.touchesCancelled(touches, with: event, in: self)
    }
}

/// SwiftUI wrapper for DrawingCanvasUIView
struct DrawingCanvasView: UIViewRepresentable {
    @ObservedObject var viewModel: CanvasViewModel

    func makeUIView(context: Context) -> DrawingCanvasUIView {
        guard let renderer = viewModel.renderer else {
            fatalError("Renderer not initialized")
        }

        let view = DrawingCanvasUIView(frame: .zero, device: renderer.device)
        view.delegate = context.coordinator
        view.clearColor = MTLClearColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1.0)
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.preferredFramesPerSecond = 60

        // Setup gesture coordinator
        let coordinator = context.coordinator.gestureCoordinator
        coordinator.setupGestures(for: view)
        view.gestureCoordinator = coordinator
        view.renderer = renderer

        // Wire up callbacks
        coordinator.onStrokeBegin = { [weak viewModel] point in
            viewModel?.beginStroke(at: point)
        }
        coordinator.onStrokeContinue = { [weak viewModel] point in
            viewModel?.continueStroke(to: point)
        }
        coordinator.onStrokeEnd = { [weak viewModel] in
            viewModel?.endStroke()
        }
        coordinator.onPan = { [weak viewModel] delta in
            viewModel?.handlePanDelta(delta)
        }
        coordinator.onPinch = { [weak viewModel] scale, center in
            viewModel?.handlePinchAt(scale: scale, center: center)
        }
        coordinator.onRotation = { [weak viewModel] angle, center in
            viewModel?.handleRotationAt(angle: angle, center: center)
        }
        coordinator.onUndo = { [weak viewModel] in
            viewModel?.undo()
        }
        coordinator.onRedo = { [weak viewModel] in
            viewModel?.redo()
        }

        return view
    }

    func updateUIView(_ uiView: DrawingCanvasUIView, context: Context) {
        uiView.setNeedsDisplay()
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(viewModel: viewModel)
    }

    class Coordinator: NSObject, MTKViewDelegate {
        let viewModel: CanvasViewModel
        let gestureCoordinator = GestureCoordinator()

        init(viewModel: CanvasViewModel) {
            self.viewModel = viewModel
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            viewModel.updateViewSize(size)
        }

        func draw(in view: MTKView) {
            guard let renderer = viewModel.renderer,
                  let drawable = view.currentDrawable,
                  let renderPassDescriptor = view.currentRenderPassDescriptor else {
                return
            }

            renderer.render(to: drawable, with: renderPassDescriptor)
        }
    }
}

// MARK: - CanvasViewModel Extensions

extension CanvasViewModel {
    /// Handle pan gesture delta
    func handlePanDelta(_ delta: CGPoint) {
        renderer?.canvasTransform.applyPan(delta: delta)
    }

    /// Handle pinch gesture at center point
    func handlePinchAt(scale: CGFloat, center: CGPoint) {
        renderer?.canvasTransform.applyPinch(scaleFactor: scale, center: center)
    }

    /// Handle rotation gesture at center point
    func handleRotationAt(angle: CGFloat, center: CGPoint) {
        renderer?.canvasTransform.applyRotation(angleDelta: angle, center: center)
    }
}
