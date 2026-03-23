import Foundation
import CoreGraphics

/// Extracts polygon contours from binary masks using marching squares algorithm
class ContourExtractor {
    /// Extract contours from a binary mask
    /// - Parameters:
    ///   - mask: Binary mask data (UInt8, 0 or 1)
    ///   - width: Mask width
    ///   - height: Mask height
    /// - Returns: Array of polygon contours, each contour is an array of points
    static func extractContours(
        from mask: [UInt8],
        width: Int,
        height: Int
    ) -> [[CGPoint]] {
        var contours: [[CGPoint]] = []
        var visited = [Bool](repeating: false, count: width * height)

        // Find all starting points for contours
        for y in 0..<height - 1 {
            for x in 0..<width - 1 {
                let index = y * width + x

                // Skip if already visited
                if visited[index] { continue }

                // Check if this is a boundary pixel (1 adjacent to 0)
                if mask[index] == 1 && hasBoundaryNeighbor(mask: mask, x: x, y: y, width: width, height: height) {
                    // Trace contour starting from this point
                    if let contour = traceContour(
                        mask: mask,
                        startX: x,
                        startY: y,
                        width: width,
                        height: height,
                        visited: &visited
                    ) {
                        if contour.count >= 3 {
                            contours.append(contour)
                        }
                    }
                }
            }
        }

        return contours
    }

    /// Check if pixel has a boundary neighbor (adjacent to background)
    private static func hasBoundaryNeighbor(
        mask: [UInt8],
        x: Int,
        y: Int,
        width: Int,
        height: Int
    ) -> Bool {
        let neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for (dx, dy) in neighbors {
            let nx = x + dx
            let ny = y + dy

            if nx < 0 || nx >= width || ny < 0 || ny >= height {
                return true // Edge of image is boundary
            }

            if mask[ny * width + nx] == 0 {
                return true
            }
        }

        return false
    }

    /// Trace a single contour using boundary following
    private static func traceContour(
        mask: [UInt8],
        startX: Int,
        startY: Int,
        width: Int,
        height: Int,
        visited: inout [Bool]
    ) -> [CGPoint]? {
        var contour: [CGPoint] = []
        var x = startX
        var y = startY
        var direction = 0 // 0: right, 1: down, 2: left, 3: up

        let dx = [1, 0, -1, 0]
        let dy = [0, 1, 0, -1]

        let maxIterations = width * height * 4

        for _ in 0..<maxIterations {
            // Mark as visited
            visited[y * width + x] = true
            contour.append(CGPoint(x: x, y: y))

            // Find next boundary pixel
            var found = false

            for i in 0..<4 {
                let newDir = (direction + 3 + i) % 4 // Turn left first, then check clockwise
                let nx = x + dx[newDir]
                let ny = y + dy[newDir]

                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    if mask[ny * width + nx] == 1 {
                        x = nx
                        y = ny
                        direction = newDir
                        found = true
                        break
                    }
                }
            }

            if !found {
                break
            }

            // Check if we've returned to start
            if x == startX && y == startY && contour.count > 2 {
                break
            }
        }

        return contour
    }

    /// Simplify contour using Douglas-Peucker algorithm
    static func simplifyContour(_ contour: [CGPoint], epsilon: CGFloat = 1.0) -> [CGPoint] {
        guard contour.count > 2 else { return contour }

        // Find the point with maximum distance from line between first and last
        var maxDist: CGFloat = 0
        var maxIndex = 0

        let first = contour[0]
        let last = contour[contour.count - 1]

        for i in 1..<contour.count - 1 {
            let dist = perpendicularDistance(point: contour[i], lineStart: first, lineEnd: last)
            if dist > maxDist {
                maxDist = dist
                maxIndex = i
            }
        }

        // If max distance is greater than epsilon, recursively simplify
        if maxDist > epsilon {
            let left = simplifyContour(Array(contour[0...maxIndex]), epsilon: epsilon)
            let right = simplifyContour(Array(contour[maxIndex...]), epsilon: epsilon)

            return Array(left.dropLast()) + right
        } else {
            return [first, last]
        }
    }

    /// Calculate perpendicular distance from point to line
    private static func perpendicularDistance(
        point: CGPoint,
        lineStart: CGPoint,
        lineEnd: CGPoint
    ) -> CGFloat {
        let dx = lineEnd.x - lineStart.x
        let dy = lineEnd.y - lineStart.y

        let length = sqrt(dx * dx + dy * dy)

        if length == 0 {
            return sqrt(pow(point.x - lineStart.x, 2) + pow(point.y - lineStart.y, 2))
        }

        let t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / (length * length)
        let clampedT = max(0, min(1, t))

        let projX = lineStart.x + clampedT * dx
        let projY = lineStart.y + clampedT * dy

        return sqrt(pow(point.x - projX, 2) + pow(point.y - projY, 2))
    }

    /// Scale contour from mask coordinates to image coordinates
    static func scaleContour(
        _ contour: [CGPoint],
        scaleFactor: Float
    ) -> [CGPoint] {
        let invScale = CGFloat(1.0 / scaleFactor)
        return contour.map { point in
            CGPoint(x: point.x * invScale, y: point.y * invScale)
        }
    }
}
