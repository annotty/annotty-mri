import UIKit
import Combine

/// Input type classification
enum InputType {
    case pencil
    case finger
}

/// Gesture coordinator that separates Apple Pencil from finger input
/// - Pencil: Drawing immediately (no delay)
/// - 1-finger: Drawing after 32ms delay (cancelable if 2nd finger detected)
/// - 2+ fingers: Navigation (pan, zoom, rotate)
/// - 2-finger tap: Undo (requires navigation gestures to fail first)
/// - 3-finger tap: Redo
class GestureCoordinator: NSObject {
    // MARK: - Configuration

    /// Delay before finger drawing starts (allows time to detect 2nd finger)
    static let fingerDrawingDelay: TimeInterval = 0.032  // 32ms ≈ 2 frames at 60fps

    // MARK: - Callbacks

    /// Called when stroke begins (pencil or finger after delay)
    var onStrokeBegin: ((CGPoint) -> Void)?

    /// Called when stroke continues
    var onStrokeContinue: ((CGPoint) -> Void)?

    /// Called when stroke ends normally
    var onStrokeEnd: (() -> Void)?

    /// Called when stroke is cancelled (2+ fingers detected)
    var onStrokeCancel: (() -> Void)?

    /// Called when pan gesture updates
    var onPan: ((CGPoint) -> Void)?

    /// Called when pinch gesture updates
    var onPinch: ((CGFloat, CGPoint) -> Void)?

    /// Called when rotation gesture updates
    var onRotation: ((CGFloat, CGPoint) -> Void)?

    /// Called when undo triggered (2-finger tap)
    var onUndo: (() -> Void)?

    /// Called when redo triggered (3-finger tap)
    var onRedo: (() -> Void)?

    /// Called when fill tap triggered (single tap in fill mode)
    var onFillTap: ((CGPoint) -> Void)?

    /// Called when smooth stroke begins
    var onSmoothStrokeBegin: ((CGPoint) -> Void)?

    /// Called when smooth stroke continues
    var onSmoothStrokeContinue: ((CGPoint) -> Void)?

    /// Called when smooth stroke ends
    var onSmoothStrokeEnd: (() -> Void)?

    /// Whether fill mode is active
    var isFillMode: Bool = false

    /// Whether smooth mode is active (for boundary smoothing)
    var isSmoothMode: Bool = false

    // MARK: - State

    private var isDrawing = false
    private var isPendingDraw = false  // Waiting for 32ms delay
    private var pendingDrawPoint: CGPoint = .zero
    private var pendingDrawTimer: DispatchWorkItem?
    private var currentTouchCount = 0
    private var lastPinchScale: CGFloat = 1.0
    private var drawingInputType: InputType = .finger
    private var touchStartPoint: CGPoint = .zero  // For fill tap detection

    /// Last time a navigation gesture (pinch/rotation) ended
    private var lastNavigationGestureTime: Date = .distantPast

    /// Cooldown period after navigation gestures before allowing undo/redo
    private let undoCooldownPeriod: TimeInterval = 0.3

    // MARK: - Gesture Recognizers

    private var panGesture: UIPanGestureRecognizer?
    private var pinchGesture: UIPinchGestureRecognizer?
    private var rotationGesture: UIRotationGestureRecognizer?
    private var twoFingerTapGesture: UITapGestureRecognizer?
    private var threeFingerTapGesture: UITapGestureRecognizer?

    // MARK: - Setup

    func setupGestures(for view: UIView) {
        // 2-finger tap (undo) - set up FIRST so other gestures can require it to fail
        let twoFingerTap = UITapGestureRecognizer(target: self, action: #selector(handleTwoFingerTap(_:)))
        twoFingerTap.numberOfTouchesRequired = 2
        twoFingerTap.numberOfTapsRequired = 1
        twoFingerTap.delegate = self
        view.addGestureRecognizer(twoFingerTap)
        twoFingerTapGesture = twoFingerTap

        // 3-finger tap (redo)
        let threeFingerTap = UITapGestureRecognizer(target: self, action: #selector(handleThreeFingerTap(_:)))
        threeFingerTap.numberOfTouchesRequired = 3
        threeFingerTap.numberOfTapsRequired = 1
        threeFingerTap.delegate = self
        view.addGestureRecognizer(threeFingerTap)
        threeFingerTapGesture = threeFingerTap

        // Pan gesture (2 fingers for navigation)
        let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        pan.minimumNumberOfTouches = 2
        pan.maximumNumberOfTouches = 2
        pan.delegate = self
        view.addGestureRecognizer(pan)
        panGesture = pan

        // Pinch gesture (zoom)
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        pinch.delegate = self
        view.addGestureRecognizer(pinch)
        pinchGesture = pinch

        // Rotation gesture (free rotation)
        let rotation = UIRotationGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
        rotation.delegate = self
        view.addGestureRecognizer(rotation)
        rotationGesture = rotation

        // 2-finger tap requires navigation gestures to FAIL before recognizing.
        // This prevents undo from firing during or after pan/pinch/rotation.
        twoFingerTap.require(toFail: pan)
        twoFingerTap.require(toFail: pinch)
        twoFingerTap.require(toFail: rotation)
        // Note: 3-finger tap does NOT require(toFail:) because pinch/rotation
        // can hold "possible" state with 3 fingers, blocking redo indefinitely.
        // 3-finger tap is deliberate enough that cooldown alone is sufficient.
    }

    // MARK: - Touch Handling

    /// Classify touch input type
    func classifyTouch(_ touch: UITouch) -> InputType {
        switch touch.type {
        case .pencil:
            return .pencil
        case .direct, .indirect, .indirectPointer:
            return .finger
        @unknown default:
            return .finger
        }
    }

    /// Handle touch began
    func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?, in view: UIView) {
        // Update total touch count
        currentTouchCount = event?.allTouches?.filter { $0.phase != .ended && $0.phase != .cancelled }.count ?? touches.count

        // If 2+ fingers, cancel any pending/ongoing stroke
        if currentTouchCount >= 2 {
            cancelOngoingStroke()
            return
        }

        guard let touch = touches.first else { return }
        let inputType = classifyTouch(touch)
        let location = touch.location(in: view)

        // Record start point for fill tap detection
        touchStartPoint = location

        // In smooth mode, handle like regular drawing but with different callbacks
        if isSmoothMode {
            drawingInputType = inputType
            isDrawing = true
            onSmoothStrokeBegin?(location)
            return
        }

        // In fill mode, wait for tap (handled in touchesEnded)
        if isFillMode {
            return
        }

        if inputType == .pencil {
            // Pencil: Start drawing immediately
            drawingInputType = .pencil
            isDrawing = true
            onStrokeBegin?(location)
        } else {
            // Finger: Wait 32ms before starting (allows 2nd finger to cancel)
            drawingInputType = .finger
            isPendingDraw = true
            pendingDrawPoint = location

            pendingDrawTimer?.cancel()
            pendingDrawTimer = DispatchWorkItem { [weak self] in
                guard let self = self, self.isPendingDraw else { return }
                // 32ms passed without 2nd finger - start drawing
                self.isPendingDraw = false
                self.isDrawing = true
                self.onStrokeBegin?(self.pendingDrawPoint)
            }
            DispatchQueue.main.asyncAfter(
                deadline: .now() + Self.fingerDrawingDelay,
                execute: pendingDrawTimer!
            )
        }
    }

    /// Handle touch moved
    func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?, in view: UIView) {
        // Update total touch count
        currentTouchCount = event?.allTouches?.filter { $0.phase != .ended && $0.phase != .cancelled }.count ?? touches.count

        // If 2+ fingers detected during drawing, cancel
        if currentTouchCount >= 2 {
            cancelOngoingStroke()
            return
        }

        guard let touch = touches.first else { return }
        let location = touch.location(in: view)

        // Smooth mode: send continue events
        if isSmoothMode && isDrawing {
            onSmoothStrokeContinue?(location)
            return
        }

        if isPendingDraw {
            // Still waiting for 32ms delay - update pending point
            pendingDrawPoint = location
        } else if isDrawing {
            // Already drawing - continue stroke
            onStrokeContinue?(location)
        }
    }

    /// Handle touch ended
    func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?, in view: UIView) {
        // Update total touch count
        let remainingTouches = event?.allTouches?.filter { $0.phase != .ended && $0.phase != .cancelled }.count ?? 0
        currentTouchCount = remainingTouches

        guard let touch = touches.first else { return }
        let location = touch.location(in: view)

        // Smooth mode: end stroke
        if isSmoothMode && isDrawing {
            isDrawing = false
            onSmoothStrokeEnd?()
            return
        }

        // Fill mode: detect tap (small movement from start)
        if isFillMode {
            let distance = hypot(location.x - touchStartPoint.x, location.y - touchStartPoint.y)
            if distance < 20 {  // Tap threshold: 20 points
                onFillTap?(location)
            }
            return
        }

        if isPendingDraw {
            // Touch ended before 32ms delay - cancel pending
            cancelPendingDraw()
        } else if isDrawing {
            // Normal stroke end
            isDrawing = false
            onStrokeEnd?()
        }
    }

    /// Handle touch cancelled
    func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?, in view: UIView) {
        currentTouchCount = 0
        cancelOngoingStroke()
    }

    // MARK: - Private Helpers

    /// Cancel pending draw timer (before 32ms)
    private func cancelPendingDraw() {
        pendingDrawTimer?.cancel()
        pendingDrawTimer = nil
        isPendingDraw = false
    }

    /// Cancel both pending and ongoing strokes (e.g. when 2+ fingers detected)
    private func cancelOngoingStroke() {
        cancelPendingDraw()
        if isDrawing {
            isDrawing = false
            onStrokeCancel?()  // Different from onStrokeEnd - triggers undo restore
        }
    }

    // MARK: - Gesture Handlers

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let view = gesture.view else { return }

        switch gesture.state {
        case .changed:
            let translation = gesture.translation(in: view)
            onPan?(translation)
            gesture.setTranslation(.zero, in: view)
        case .ended, .cancelled:
            lastNavigationGestureTime = Date()
        default:
            break
        }
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard let view = gesture.view else { return }

        switch gesture.state {
        case .began:
            lastPinchScale = 1.0
        case .changed:
            let center = gesture.location(in: view)
            let scale = gesture.scale / lastPinchScale
            onPinch?(scale, center)
            lastPinchScale = gesture.scale
        case .ended, .cancelled:
            lastPinchScale = 1.0
            lastNavigationGestureTime = Date()
        default:
            break
        }
    }

    @objc private func handleRotation(_ gesture: UIRotationGestureRecognizer) {
        guard let view = gesture.view else { return }

        switch gesture.state {
        case .changed:
            let center = gesture.location(in: view)
            // Free rotation, no snapping
            onRotation?(gesture.rotation, center)
            gesture.rotation = 0
        case .ended, .cancelled:
            lastNavigationGestureTime = Date()
        default:
            break
        }
    }

    @objc private func handleTwoFingerTap(_ gesture: UITapGestureRecognizer) {
        if gesture.state == .recognized {
            // Ignore if within cooldown period after pinch/rotation
            guard Date().timeIntervalSince(lastNavigationGestureTime) > undoCooldownPeriod else {
                return
            }
            onUndo?()
        }
    }

    @objc private func handleThreeFingerTap(_ gesture: UITapGestureRecognizer) {
        if gesture.state == .recognized {
            // Ignore if within cooldown period after pinch/rotation
            guard Date().timeIntervalSince(lastNavigationGestureTime) > undoCooldownPeriod else {
                return
            }
            onRedo?()
        }
    }
}

// MARK: - UIGestureRecognizerDelegate

extension GestureCoordinator: UIGestureRecognizerDelegate {
    /// Allow simultaneous gesture recognition for pan, pinch, and rotation
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
    ) -> Bool {
        // Allow pan, pinch, and rotation to work together
        let navigationGestures: [UIGestureRecognizer?] = [
            panGesture, pinchGesture, rotationGesture
        ]

        if navigationGestures.contains(where: { $0 === gestureRecognizer }) &&
           navigationGestures.contains(where: { $0 === otherGestureRecognizer }) {
            return true
        }

        // Allow 3-finger tap (redo) to recognize alongside pinch/rotation,
        // which may also activate with 3 fingers on screen.
        if gestureRecognizer === threeFingerTapGesture ||
           otherGestureRecognizer === threeFingerTapGesture {
            return true
        }

        return false
    }

    /// Ensure gestures only activate with finger touches, not pencil
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldReceive touch: UITouch
    ) -> Bool {
        // Only allow finger touches for navigation gestures
        return classifyTouch(touch) == .finger
    }
}
