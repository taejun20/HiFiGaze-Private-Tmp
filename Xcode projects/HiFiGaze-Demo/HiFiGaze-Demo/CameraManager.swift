import AVFoundation
import UIKit

final class CameraManager: NSObject {
    enum CameraError: Error { case configurationFailed, permissionDenied }
    
    private let session = AVCaptureSession()
    var captureSession: AVCaptureSession { session }
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")

    var onFrame: ((CMSampleBuffer) -> Void)?

    func start() throws {
        var auth = AVCaptureDevice.authorizationStatus(for: .video)
        if auth == .notDetermined {
            let sema = DispatchSemaphore(value: 0)
            AVCaptureDevice.requestAccess(for: .video) { _ in sema.signal() }
            _ = sema.wait(timeout: .now() + 10)
            auth = AVCaptureDevice.authorizationStatus(for: .video)
        }
        guard auth == .authorized else { throw CameraError.permissionDenied }
        
        sessionQueue.async {
            self.session.beginConfiguration()
            self.session.sessionPreset = .hd4K3840x2160

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }
            self.session.addInput(input)

            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frames.queue"))
            if self.session.canAddOutput(self.videoOutput) { self.session.addOutput(self.videoOutput) }

            if let conn = self.videoOutput.connection(with: .video), conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
            }

            self.session.commitConfiguration()
            self.session.startRunning()
        }
    }
    
    func stop() {
        sessionQueue.sync { session.stopRunning() }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        onFrame?(sampleBuffer)
    }
}
