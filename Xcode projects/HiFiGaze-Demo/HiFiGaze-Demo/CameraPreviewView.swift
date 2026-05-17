import SwiftUI
import AVFoundation

// UIView subclass whose backing layer *is* an AVCaptureVideoPreviewLayer
final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black  // just for debugging

        let previewLayer = view.videoPreviewLayer
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspectFill

        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.videoPreviewLayer.session = session
        // No need to set frame; layer automatically matches view.bounds
    }
}
