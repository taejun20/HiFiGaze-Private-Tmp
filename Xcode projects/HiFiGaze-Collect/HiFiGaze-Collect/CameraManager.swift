import Foundation
import AVFoundation

final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()

    private let queue = DispatchQueue(label: "hifigaze.camera.queue")
    private let output = AVCaptureVideoDataOutput()

    var onSampleBuffer: ((CMSampleBuffer) -> Void)?

    func start() {
        queue.async {
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }

    func configureFrontCameraBestEffort4K() {
        session.beginConfiguration()
        session.sessionPreset = .high

        // Input
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            return
        }
        session.inputs.forEach { session.removeInput($0) }
        session.addInput(input)

        // Try to pick highest resolution format (best effort)
        try? device.lockForConfiguration()
        if let best = device.formats.max(by: { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            let pa = Int(da.width) * Int(da.height)
            let pb = Int(db.width) * Int(db.height)
            return pa < pb
        }) {
            device.activeFormat = best
        }
        device.unlockForConfiguration()

        // Output
        session.outputs.forEach { session.removeOutput($0) }
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        if let conn = output.connection(with: .video) {
            conn.videoOrientation = .portrait
            conn.isVideoMirrored = true
        }

        session.commitConfiguration()
    }

    // MARK: Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        onSampleBuffer?(sampleBuffer)
    }
}
