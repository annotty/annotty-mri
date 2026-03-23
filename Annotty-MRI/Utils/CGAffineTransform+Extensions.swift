import CoreGraphics

extension CGAffineTransform {
    /// Extract scale factor from transform
    var scaleX: CGFloat {
        sqrt(a * a + c * c)
    }

    var scaleY: CGFloat {
        sqrt(b * b + d * d)
    }

    /// Extract rotation angle from transform (in radians)
    var rotation: CGFloat {
        atan2(b, a)
    }

    /// Extract translation
    var translation: CGPoint {
        CGPoint(x: tx, y: ty)
    }

    /// Create transform with rotation around a point
    static func rotation(angle: CGFloat, around point: CGPoint) -> CGAffineTransform {
        CGAffineTransform.identity
            .translatedBy(x: point.x, y: point.y)
            .rotated(by: angle)
            .translatedBy(x: -point.x, y: -point.y)
    }

    /// Create transform with scale around a point
    static func scale(_ scale: CGFloat, around point: CGPoint) -> CGAffineTransform {
        CGAffineTransform.identity
            .translatedBy(x: point.x, y: point.y)
            .scaledBy(x: scale, y: scale)
            .translatedBy(x: -point.x, y: -point.y)
    }
}

extension CGPoint {
    /// Distance to another point
    func distance(to other: CGPoint) -> CGFloat {
        sqrt(pow(x - other.x, 2) + pow(y - other.y, 2))
    }

    /// Midpoint between two points
    func midpoint(to other: CGPoint) -> CGPoint {
        CGPoint(x: (x + other.x) / 2, y: (y + other.y) / 2)
    }
}

extension CGRect {
    /// Center point of rectangle
    var center: CGPoint {
        CGPoint(x: midX, y: midY)
    }

    /// Create rectangle from center and size
    init(center: CGPoint, size: CGSize) {
        self.init(
            x: center.x - size.width / 2,
            y: center.y - size.height / 2,
            width: size.width,
            height: size.height
        )
    }
}
