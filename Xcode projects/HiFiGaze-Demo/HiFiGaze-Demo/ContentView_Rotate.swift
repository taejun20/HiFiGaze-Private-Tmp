//import SwiftUI
//import AVFoundation
//import CoreImage
//import MediaPipeTasksVision
//
//let L_LEFT_CORNER  = 33
//let L_RIGHT_CORNER = 133
//let L_IRIS         = 468
//
//struct ContentView: View {
//    @StateObject private var vm = EyeCropViewModel()
//
//    @State private var currentImageIndex = 1
//    @State private var showCamera = true
//
//    private let numberOfImages = 9
//
//    // Display box variables
//    private let displayWidth: CGFloat = 300
//    private let displayHeight: CGFloat = 150
//
//    // (0,0) = screen center
//    private let displayCenterX: CGFloat = 0
//    private let displayCenterY: CGFloat = 0
//
//    var body: some View {
//        GeometryReader { geometry in
//            let posX = geometry.size.width / 2 + displayCenterX
//            let posY = geometry.size.height / 2 + displayCenterY
//
//            ZStack(alignment: .top) {
//                if showCamera {
//                    CameraPreviewView(session: vm.captureSession)
//                        .frame(width: geometry.size.width, height: geometry.size.height)
//                        .clipped()
//                } else {
//                    Image("img\(currentImageIndex)")
//                        .resizable()
//                        .aspectRatio(contentMode: .fill)
//                        .frame(width: geometry.size.width)
//                        .clipped()
//                }
//
//                if let image = vm.eyeImage {
//                    Image(uiImage: image)
//                        .resizable()
//                        .interpolation(.high)
//                        .scaledToFit()
//                        .frame(width: displayWidth, height: displayHeight)
//                        .background(Color.black)
//                        .clipShape(RoundedRectangle(cornerRadius: 12))
//                        .position(x: posX, y: posY)
//                } else {
//                    Text("Waiting for left eye…")
//                        .foregroundColor(.white)
//                        .padding(10)
//                        .background(Color.black.opacity(0.4))
//                        .clipShape(RoundedRectangle(cornerRadius: 8))
//                        .position(x: posX, y: posY)
//                }
//            }
//            .contentShape(Rectangle())
//            .onTapGesture {
//                withAnimation(.easeInOut(duration: 0.2)) {
//                    if showCamera {
//                        showCamera = false
//                        currentImageIndex = 1
//                    } else {
//                        if currentImageIndex < numberOfImages {
//                            currentImageIndex += 1
//                        } else {
//                            showCamera = true
//                            currentImageIndex = 1
//                        }
//                    }
//                }
//            }
//        }
//        .onAppear {
//            vm.start()
//        }
//        .onDisappear {
//            vm.stop()
//        }
//    }
//}
//
//final class EyeCropViewModel: ObservableObject {
//    @Published var eyeImage: UIImage?
//
//    private let eyeCropWidth: CGFloat = 260
//    private let eyeCropHeight: CGFloat = 130
//
//    private let camera = CameraManager()
//    var captureSession: AVCaptureSession { camera.captureSession }
//
//    private let ciContext = CIContext()
//    private var faceLandmarker: FaceLandmarker?
//    private var isRunning = false
//
//    func start() {
//        camera.onFrame = { [weak self] sampleBuffer in
//            self?.process(sampleBuffer: sampleBuffer)
//        }
//        try? camera.start()
//    }
//
//    func stop() {
//        camera.stop()
//    }
//
//    private func ensureLandmarker() {
//        if faceLandmarker != nil { return }
//
//        let base = BaseOptions()
//        base.modelAssetPath = Bundle.main.path(
//            forResource: "face_landmarker",
//            ofType: "task"
//        )!
//
//        let opts = FaceLandmarkerOptions()
//        opts.baseOptions = base
//        opts.numFaces = 1
//        opts.runningMode = .image
//
//        faceLandmarker = try? FaceLandmarker(options: opts)
//    }
//
//    private func process(sampleBuffer: CMSampleBuffer) {
//        guard !isRunning else { return }
//        isRunning = true
//        defer { isRunning = false }
//
//        guard
//            let sbufCopy = copySampleBuffer(sampleBuffer),
//            let pixelBuffer = CMSampleBufferGetImageBuffer(sbufCopy)
//        else { return }
//
//        ensureLandmarker()
//        guard let landmarker = faceLandmarker else { return }
//
//        do {
//            let mpImage = try MPImage(pixelBuffer: pixelBuffer, orientation: .up)
//            let result = try landmarker.detect(image: mpImage)
//
//            guard let lms = result.faceLandmarks.first else { return }
//
//            guard let leftEyeCI = cropEyeCI(
//                pixelBuffer: pixelBuffer,
//                lms: lms,
//                left: L_LEFT_CORNER,
//                right: L_RIGHT_CORNER,
//                center: L_IRIS,
//                cropWidth: eyeCropWidth,
//                cropHeight: eyeCropHeight
//            ) else {
//                return
//            }
//
//            let normalized = leftEyeCI.transformed(
//                by: CGAffineTransform(
//                    translationX: -leftEyeCI.extent.origin.x,
//                    y: -leftEyeCI.extent.origin.y
//                )
//            )
//
//            let flipped = normalized.transformed(
//                by: CGAffineTransform(scaleX: -1, y: 1)
//                    .translatedBy(x: -normalized.extent.width, y: 0)
//            )
//
//            guard let cgImage = ciContext.createCGImage(flipped, from: flipped.extent) else {
//                return
//            }
//
//            let uiImage = UIImage(cgImage: cgImage)
//
//            DispatchQueue.main.async {
//                self.eyeImage = uiImage
//            }
//        } catch {
//            print("❌ MediaPipe error:", error)
//        }
//    }
//}
//
//func copySampleBuffer(_ sbuf: CMSampleBuffer) -> CMSampleBuffer? {
//    var copy: CMSampleBuffer?
//    let status = CMSampleBufferCreateCopy(
//        allocator: kCFAllocatorDefault,
//        sampleBuffer: sbuf,
//        sampleBufferOut: &copy
//    )
//    return status == noErr ? copy : nil
//}
//
//func lmkPx(_ lms: [NormalizedLandmark], _ idx: Int, _ w: CGFloat, _ h: CGFloat) -> CGPoint {
//    CGPoint(
//        x: CGFloat(lms[idx].x) * w,
//        y: CGFloat(lms[idx].y) * h
//    )
//}
//
//func cropEyeCI(
//    pixelBuffer: CVPixelBuffer,
//    lms: [NormalizedLandmark],
//    left: Int,
//    right: Int,
//    center: Int,
//    cropWidth: CGFloat,
//    cropHeight: CGFloat
//) -> CIImage? {
//    let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
//    let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
//
//    let L0 = lmkPx(lms, left,   w, h)
//    let R0 = lmkPx(lms, right,  w, h)
//    let C0 = lmkPx(lms, center, w, h)
//
//    let L = CGPoint(x: L0.x, y: h - L0.y)
//    let R = CGPoint(x: R0.x, y: h - R0.y)
//    let C = CGPoint(x: C0.x, y: h - C0.y)
//
//    let dx = R.x - L.x
//    let dy = R.y - L.y
//    let eyeWidth = hypot(dx, dy)
//
//    guard eyeWidth > 1 else { return nil }
//
//    let angle = -atan2(dy, dx)
//    let scale = 250.0 / eyeWidth
//
//    let src = CIImage(cvPixelBuffer: pixelBuffer)
//
//    var t = CGAffineTransform.identity
//    t = t.scaledBy(x: scale, y: scale)
//    t = t.rotated(by: angle)
//    t = t.translatedBy(x: -C.x, y: -C.y)
//
//    let transformed = src.transformed(by: t)
//    let cropped = transformed.cropped(
//        to: CGRect(
//            x: -cropWidth / 2,
//            y: -cropHeight / 2,
//            width: cropWidth,
//            height: cropHeight
//        )
//    )
//
//    return cropped
//}
