import SwiftUI
import MetalKit

/// Custom MTKView subclass that handles touch events for drawing
class TouchableMTKView: MTKView {
    /// Gesture coordinator for handling drawing vs navigation
    var gestureCoordinator: GestureCoordinator?

    override init(frame frameRect: CGRect, device: MTLDevice?) {
        super.init(frame: frameRect, device: device)
        isMultipleTouchEnabled = true
    }

    required init(coder: NSCoder) {
        super.init(coder: coder)
        isMultipleTouchEnabled = true
    }

    // Prevent iPadOS system 3-finger gestures (undo/redo/edit menu)
    // from intercepting our custom 3-finger tap for redo
    override var undoManager: UndoManager? { nil }

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

/// SwiftUI wrapper for MTKView that renders the canvas with UIKit touch handling
struct MetalCanvasView: UIViewRepresentable {
    @ObservedObject var renderer: MetalRenderer
    let gestureCoordinator: GestureCoordinator

    func makeUIView(context: Context) -> TouchableMTKView {
        let mtkView = TouchableMTKView(frame: .zero, device: renderer.device)
        mtkView.delegate = context.coordinator
        mtkView.clearColor = MTLClearColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1.0)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.framebufferOnly = false
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = false
        mtkView.preferredFramesPerSecond = 60

        // Connect gesture coordinator
        mtkView.gestureCoordinator = gestureCoordinator
        gestureCoordinator.setupGestures(for: mtkView)

        return mtkView
    }

    func updateUIView(_ uiView: TouchableMTKView, context: Context) {
        uiView.setNeedsDisplay()
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer)
    }

    class Coordinator: NSObject, MTKViewDelegate {
        let renderer: MetalRenderer

        init(renderer: MetalRenderer) {
            self.renderer = renderer
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            renderer.updateViewportSize(size, scaleFactor: view.contentScaleFactor)
        }

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let renderPassDescriptor = view.currentRenderPassDescriptor else {
                return
            }

            renderer.render(to: drawable, with: renderPassDescriptor)
        }
    }
}
