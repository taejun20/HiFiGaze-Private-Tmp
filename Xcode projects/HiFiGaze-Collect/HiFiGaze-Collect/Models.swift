import Foundation
import CoreGraphics

enum LRResponse {
    case left, right
}

struct TrialTarget {
    let point: CGPoint
    let isGrid: Bool
}

struct CSVRow {
    let subject: String
    let frameID: String
    let screen: String
    let gt_x_px: Int
    let gt_y_px: Int
    let timestamp: Double
}
