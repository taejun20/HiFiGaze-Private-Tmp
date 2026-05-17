import Foundation
import CoreGraphics
import QuartzCore

final class OneEuroFilter {
    private var freq: Double
    private let minCutoff: Double
    private let beta: Double
    private let dCutoff: Double

    private var xPrev: CGPoint?
    private var dxPrev: CGPoint?
    private var lastTime: CFTimeInterval?

    init(
        freq: Double = 20.0,
        minCutoff: Double = 1.0,
        beta: Double = 0.005,
        dCutoff: Double = 1.0
    ) {
        self.freq = freq
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    private func alpha(cutoff: Double) -> Double {
        let tau = 1.0 / (2.0 * .pi * cutoff)
        let te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)
    }

    private func lowpass(
        _ value: CGPoint,
        _ prev: CGPoint,
        _ alpha: Double
    ) -> CGPoint {
        CGPoint(
            x: CGFloat(alpha) * value.x + CGFloat(1 - alpha) * prev.x,
            y: CGFloat(alpha) * value.y + CGFloat(1 - alpha) * prev.y
        )
    }

    func filter(_ x: CGPoint, timestamp: CFTimeInterval = CACurrentMediaTime()) -> CGPoint {

        if lastTime == nil {
            lastTime = timestamp
            xPrev = x
            dxPrev = .zero
            return x
        }

        let dt = timestamp - lastTime!
        lastTime = timestamp

        // Protect against spikes
        if dt <= 0 || dt > 0.2 {
            xPrev = x
            dxPrev = .zero
            return x
        }

        freq = 1.0 / dt

        let dx = CGPoint(
            x: (x.x - xPrev!.x) * CGFloat(freq),
            y: (x.y - xPrev!.y) * CGFloat(freq)
        )

        let dxHat = lowpass(
            dx,
            dxPrev!,
            alpha(cutoff: dCutoff)
        )

        let speed = hypot(dxHat.x, dxHat.y)
        let cutoff = minCutoff + beta * speed

        let xHat = lowpass(
            x,
            xPrev!,
            alpha(cutoff: cutoff)
        )

        xPrev = xHat
        dxPrev = dxHat
        return xHat
    }


}
