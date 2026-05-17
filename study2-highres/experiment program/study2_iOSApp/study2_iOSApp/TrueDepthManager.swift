//
//  TrueDepthManager.swift
//  study2_iOSApp
//
//  Created by TJ on 8/7/25.
//

import ARKit
import Combine

class TrueDepthManager: NSObject, ObservableObject, ARSessionDelegate {
    static let shared = TrueDepthManager()

    @Published var leftEyeDistance: Float = 0.0
    @Published var rightEyeDistance: Float = 0.0

    private var session: ARSession?
    private var oneShotMode = false
    private var hasCapturedOnce = false

    // Normal start (continuous)
    func start() {
        configureAndRun()
        oneShotMode = false
        hasCapturedOnce = false
    }

    // One-time start: stop after first measurement
    func startOnce() {
        configureAndRun()
        oneShotMode = true
        hasCapturedOnce = false
    }

    private func configureAndRun() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("❌ TrueDepth not supported on this device")
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = true

        let s = ARSession()
        s.delegate = self
        s.run(config, options: [.resetTracking, .removeExistingAnchors])
        self.session = s
    }

    func stop() {
        session?.pause()
        session = nil
        print("⏹️ TrueDepth session stopped")
    }

    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let cameraTransform = session.currentFrame?.camera.transform else { return }

        for anchor in anchors {
            guard let faceAnchor = anchor as? ARFaceAnchor else { continue }

            // Eye pose in face -> world
            let leftWorld  = faceAnchor.transform * faceAnchor.leftEyeTransform
            let rightWorld = faceAnchor.transform * faceAnchor.rightEyeTransform

            // World -> camera
            let camInv = simd_inverse(cameraTransform)
            let leftCam4  = camInv * leftWorld.columns.3
            let rightCam4 = camInv * rightWorld.columns.3

            let l = SIMD3(leftCam4.x,  leftCam4.y,  leftCam4.z)
            let r = SIMD3(rightCam4.x, rightCam4.y, rightCam4.z)

            leftEyeDistance  = simd_length(l)
            rightEyeDistance = simd_length(r)

            TCPClient.shared.send(message: "dist:l\(Int(leftEyeDistance * 1000))r\(Int(rightEyeDistance * 1000))\n")

            // Stop right after the first successful measurement if one-shot
            if oneShotMode && !hasCapturedOnce {
                hasCapturedOnce = true
                DispatchQueue.main.async { [weak self] in
                    self?.stop()
                }
            }
            // We processed one face anchor this frame; no need to loop further
            break
        }
    }
}
