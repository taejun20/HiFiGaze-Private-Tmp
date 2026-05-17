//import UIKit
//import MediaPipeTasksVision
//import SwiftUI
//import AVFoundation
//import CoreML
//import CoreImage
//import Combine
//
//final class MainViewModelSocket_Debug: ObservableObject {
//    private let camera = CameraManager()
//    private let ciContext = CIContext()
//
//    var captureSession: AVCaptureSession { camera.captureSession }
//
//    @Published var leftEyeImage: CGImage?
//    @Published var rightEyeImage: CGImage?
//
//    @Published var rgbGazeCursor: CGPoint = .zero          // red
//    @Published var socketGazeCursor: CGPoint = .zero       // blue
//
//    private var templateArray: MLMultiArray?
//
//    private let inferenceQueue = DispatchQueue(label: "gaze.inference.queue")
//    private var isRunningInference = false
//
//    private let socketClient = SocketClient(host: "192.168.0.138", port: 9999) // <-- change to your Python machine IP/port
//
//    private var faceLandmarker: FaceLandmarker?
//
//    private let frameLockQueue = DispatchQueue(label: "frame.lock.queue")
//    private var isFrameInFlight = false
//    private var processedIndex: Int = 0
//
//    private lazy var rgbModel: HiFiGaze_RGB? = {
//        return try? HiFiGaze_RGB(configuration: MLModelConfiguration())
//    }()
//
//    func start() {
//        socketClient.connect()
//        camera.onFrame = { [weak self] sbuf in self?.process(sampleBuffer: sbuf) }
//        do { try camera.start() } catch { print("Camera error: \(error)") }
//    }
//
//    func stop() {
//        camera.stop()
//        socketClient.disconnect()
//    }
//
//    func updateScreenThumbnail(index: Int) {
//        let name = "img\(index)_thumbnail"
//        guard let ui = UIImage(named: name) else {
//            DispatchQueue.main.async { self.templateArray = nil }
//            return
//        }
//
//        let cgImage: CGImage?
//        if let cg = ui.cgImage { cgImage = cg }
//        else if let ci = ui.ciImage { cgImage = ciContext.createCGImage(ci, from: ci.extent) }
//        else { cgImage = nil }
//
//        guard let thumbCG = cgImage else {
//            DispatchQueue.main.async { self.templateArray = nil }
//            print("⚠️ Could not create CGImage for \(name)")
//            return
//        }
//
//        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
//            guard let self else { return }
//            let tmpl = cgImageToMLMultiArray(thumbCG, targetSize: CGSize(width: 50, height: 101))
//            DispatchQueue.main.async { self.templateArray = tmpl }
//        }
//    }
//
//    private func ensureLandmarker() {
//        if faceLandmarker != nil { return }
//        do {
//            guard let path = Bundle.main.path(forResource: "face_landmarker", ofType: "task") else {
//                print("⚠️ Missing face_landmarker.task in bundle")
//                return
//            }
//            let base = BaseOptions()
//            base.modelAssetPath = path
//
//            let opts = FaceLandmarkerOptions()
//            opts.baseOptions = base
//            opts.numFaces = 1
//            opts.minFaceDetectionConfidence = 0.5
//            opts.minFacePresenceConfidence = 0.5
//            opts.minTrackingConfidence = 0.5
//            opts.runningMode = .image
//
//            opts.outputFaceBlendshapes = false
//            opts.outputFacialTransformationMatrixes = false
//
//            faceLandmarker = try FaceLandmarker(options: opts)
//        } catch {
//            print("FaceLandmarker init failed: \(error)")
//        }
//    }
//
//    private func cgImageToJPEGBase64(_ cg: CGImage, quality: CGFloat = 0.85) -> String? {
//        let ui = UIImage(cgImage: cg)
//        guard let data = ui.jpegData(compressionQuality: quality) else { return nil }
//        return data.base64EncodedString()
//    }
//
//    func mlMultiArrayToData(_ arr: MLMultiArray) -> Data {
//        let count = arr.count
//        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))
//        return Data(bytes: ptr, count: count * MemoryLayout<Float32>.size)
//    }
//    
//    func debugPrintFloatArrayTorchStyle(
//        _ arr: [Float],
//        shape: [Int],
//        name: String
//    ) {
//        var minVal = Float.greatestFiniteMagnitude
//        var maxVal = -Float.greatestFiniteMagnitude
//
//        for v in arr {
//            minVal = min(minVal, v)
//            maxVal = max(maxVal, v)
//        }
//
//        print(
//            "\(name): torch.Size(\(shape)) " +
//            "(dtype: torch.float32, min: \(minVal), max: \(maxVal))"
//        )
//    }
//
//    
//    func debugPrintRawTensorTorchStyle(
//        rawData: Data,
//        shape: [Int],
//        name: String
//    ) {
//        let count = shape.reduce(1, *)
//        let expectedBytes = count * MemoryLayout<Float32>.size
//
//        guard rawData.count == expectedBytes else {
//            print("❌ \(name): byte count mismatch (got \(rawData.count), expected \(expectedBytes))")
//            return
//        }
//
//        rawData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
//            let fptr = ptr.bindMemory(to: Float32.self)
//
//            var minVal = Float.greatestFiniteMagnitude
//            var maxVal = -Float.greatestFiniteMagnitude
//
//            for i in 0..<count {
//                let v = fptr[i]
//                minVal = min(minVal, v)
//                maxVal = max(maxVal, v)
//            }
//
//            print(
//                "\(name): torch.Size(\(shape)) " +
//                "(dtype: torch.float32, min: \(minVal), max: \(maxVal))"
//            )
//        }
//    }
//
//    
//    private func process(sampleBuffer: CMSampleBuffer) {
//        var shouldProcess = false
//        frameLockQueue.sync {
//            if !isFrameInFlight {
//                isFrameInFlight = true
//                shouldProcess = true
//            }
//        }
//        if !shouldProcess {
//            return
//        }
//        
//        ensureLandmarker()
//        guard let landmarker = faceLandmarker else { return }
//        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
//
//        let ci = CIImage(cvPixelBuffer: pb)
//
//        let mpImage: MPImage
//        do {
//            mpImage = try MPImage(sampleBuffer: sampleBuffer, orientation: .up)
//        } catch {
//            print("MPImage creation failed: \(error)")
//            return
//        }
//
//        let result: FaceLandmarkerResult
//        do {
//            result = try landmarker.detect(image: mpImage)
//        } catch {
//            print("Face detection failed: \(error)")
//            return
//        }
//        guard let lms = result.faceLandmarks.first, !lms.isEmpty else { return }
//
//        let imgW = ci.extent.width
//        let imgH = ci.extent.height
//
//        let LEFT_EYE_OUTER = 263
//        let LEFT_EYELID_UPPEROUTER = 386
//        let LEFT_EYELID_UPPERINNER = 385
//        let LEFT_EYE_INNER = 362
//        let LEFT_EYELID_LOWERINNER = 374
//        let LEFT_EYELID_LOWEROUTER = 373
//
//        let RIGHT_EYE_OUTER = 33
//        let RIGHT_EYELID_UPPEROUTER = 159
//        let RIGHT_EYELID_UPPERINNER = 158
//        let RIGHT_EYE_INNER = 133
//        let RIGHT_EYELID_LOWERINNER = 145
//        let RIGHT_EYELID_LOWEROUTER = 144
//
//        guard
//            let leftRect  = rectForEye(ci: ci, lms: lms, imgW: imgW, imgH: imgH,
//                                       outer: LEFT_EYE_OUTER, upperouter: LEFT_EYELID_UPPEROUTER, upperinner: LEFT_EYELID_UPPERINNER,
//                                       inner: LEFT_EYE_INNER, lowerinner: LEFT_EYELID_LOWERINNER, lowerouter: LEFT_EYELID_LOWEROUTER),
//            let rightRect = rectForEye(ci: ci, lms: lms, imgW: imgW, imgH: imgH,
//                                       outer: RIGHT_EYE_OUTER, upperouter: RIGHT_EYELID_UPPEROUTER, upperinner: RIGHT_EYELID_UPPERINNER,
//                                       inner: RIGHT_EYE_INNER, lowerinner: RIGHT_EYELID_LOWERINNER, lowerouter: RIGHT_EYELID_LOWEROUTER)
//        else { return }
//
//        guard
//            let leftCG  = cropCG(ciContext: ciContext, ci: ci, rect: leftRect),
//            let rightCG = cropCG(ciContext: ciContext, ci: ci, rect: rightRect)
//        else { return }
//
//        DispatchQueue.main.async {
//            self.leftEyeImage  = leftCG
//            self.rightEyeImage = rightCG
//        }
//
//        let eyeLandmarks = prepareEyeLandmarks(
//            lms: lms,
//            leftOuter: LEFT_EYE_OUTER, leftUpper: LEFT_EYELID_UPPEROUTER,
//            leftInner: LEFT_EYE_INNER, leftLower: LEFT_EYELID_LOWEROUTER,
//            rightOuter: RIGHT_EYE_OUTER, rightUpper: RIGHT_EYELID_UPPEROUTER,
//            rightInner: RIGHT_EYE_INNER, rightLower: RIGHT_EYELID_LOWEROUTER
//        )
//
//        // CoreML inputs (on-device)
//        guard let leftEyeArray = cgImageToMLMultiArray(leftCG, targetSize: CGSize(width: 500, height: 250)),
//              let rightEyeArray = cgImageToMLMultiArray(rightCG, targetSize: CGSize(width: 500, height: 250)) else { return }
//
//        guard let eye_landmarksArray = try? MLMultiArray(shape: [1, 8], dataType: .float32) else { return }
//        for i in 0..<8 { eye_landmarksArray[i] = NSNumber(value: eyeLandmarks[i]) }
//
//        guard let face_landmarkArray = makeFaceLandmarksArray(from: lms) else { return }
//        guard self.templateArray != nil else { return } // you can include template too later if needed
//
//        // ---- Throttle on-device inference ----
//        if isRunningInference { return }
//        isRunningInference = true
//
//        inferenceQueue.async { [weak self] in
//            guard let self else { return }
//            defer { self.isRunningInference = false }
//
//            // 1) On-device CoreML -> red dot
//            if let rgbModel = self.rgbModel {
//                do {
//                    let pred = try rgbModel.prediction(
//                        left_eye: leftEyeArray,
//                        right_eye: rightEyeArray,
//                        eye_landmarks: eye_landmarksArray,
//                        face_landmarks: face_landmarkArray
//                    )
//                    let gazeX = pred.gaze[0].floatValue
//                    let gazeY = pred.gaze[1].floatValue
//                   
//                    print(
//                        String(
//                            format: "coreml_pred: (x: %.6f, y: %.6f)",
//                            gazeX, gazeY
//                        )
//                    )
//
//                    DispatchQueue.main.async {
//                        self.rgbGazeCursor = CGPoint(
//                            x: CGFloat(gazeX) * 1000,
//                            y: CGFloat(gazeY) * 1000
//                        )
//                    }
//                } catch {
//                    print("RGB Model inference failed: \(error)")
//                }
//            }
//
//            let leftEyeData  = mlMultiArrayToData(leftEyeArray)
//            let rightEyeData = mlMultiArrayToData(rightEyeArray)
//            
//            let faceCount = face_landmarkArray.count
//            var faceFloats = [Float](repeating: 0, count: faceCount)
//            for i in 0..<faceCount {
//                faceFloats[i] = face_landmarkArray[i].floatValue
//            }
//            self.processedIndex += 1
//            let idx = self.processedIndex
//            
//            print("=========== FRAME \(idx) ===========")
//
//            debugPrintRawTensorTorchStyle(
//                rawData: leftEyeData,
//                shape: [1, 3, 250, 500],
//                name: "left"
//            )
//
//            debugPrintRawTensorTorchStyle(
//                rawData: rightEyeData,
//                shape: [1, 3, 250, 500],
//                name: "right"
//            )
//            
//            debugPrintFloatArrayTorchStyle(
//                eyeLandmarks,
//                shape: [1, 8],
//                name: "eye_landmarks"
//            )
//
//            debugPrintFloatArrayTorchStyle(
//                faceFloats,
//                shape: [1, faceFloats.count],
//                name: "face_landmarks"
//            )
//            
//          
//            let req: [String: Any] = [
//                "index": idx,
//                "left_eye_raw": leftEyeData.base64EncodedString(),
//                "right_eye_raw": rightEyeData.base64EncodedString(),
//                "eye_landmarks": eyeLandmarks,
//                "face_landmarks": faceFloats
//            ]
//
//            self.socketClient.sendRequest(req) { [weak self] result in
//                guard let self else { return }
//
//                defer {
//                    self.frameLockQueue.async {
//                        self.isFrameInFlight = false
//                    }
//                }
//
//                switch result {
//                case .success(let dict):
//                    let x = (dict["x"] as? NSNumber)?.doubleValue ?? 0.0
//                    let y = (dict["y"] as? NSNumber)?.doubleValue ?? 0.0
//
//                    DispatchQueue.main.async {
//                        self.socketGazeCursor = CGPoint(
//                            x: CGFloat(x) * 1000,
//                            y: CGFloat(y) * 1000
//                        )
//                    }
//
//                case .failure(let err):
//                    print("❌ Socket prediction failed:", err)
//                }
//            }
//        }
//    }
//}
//
//struct MainViewSocket_Debug: View {
//    @StateObject private var vm = MainViewModelSocket_Debug()
//    
//    let showRGB: Bool
//    let showRGBT: Bool
//    
//    @State private var currentImageIndex = 1   // start at img1
//    @State private var showCamera = true       // 🔽 start with camera
//    @State private var tapsSinceCamera = 0     // count images since last camera
//
//    private func anchorDots(size: CGSize) -> some View {
//        let w = size.width
//        let h = size.height
//        let inset: CGFloat = 90  // how far from the edges
//        
//        // 8 anchor positions: 4 corners + midpoints of each edge
//        let points: [CGPoint] = [
//            CGPoint(x: inset,        y: inset),        // top-left
//            CGPoint(x: w - inset,    y: inset),        // top-right
//            CGPoint(x: inset,        y: h - inset),    // bottom-left
//            CGPoint(x: w - inset,    y: h - inset),    // bottom-right
//            CGPoint(x: w / 2,        y: inset),        // top-center
//            CGPoint(x: w / 2,        y: h - inset),    // bottom-center
//            CGPoint(x: inset,        y: h / 2),        // left-center
//            CGPoint(x: w - inset,    y: h / 2)         // right-center
//        ]
//        
//        return ZStack {
//            ForEach(0..<points.count, id: \.self) { i in
//                Circle()
//                    .fill(Color.black.opacity(0.9))
//                    .frame(width: 10, height: 10)
//                    .overlay(
//                        Circle()
//                            .stroke(Color.black.opacity(0.8), lineWidth: 1)
//                    )
//                    .position(points[i])
//            }
//        }
//    }
//
//    var body: some View {
//        GeometryReader { geometry in
//            ZStack(alignment: .top) {
//                
//                // 🔽 Background: either camera preview or static image
//                if showCamera {
//                    CameraPreviewView(session: vm.captureSession)
//                        .frame(width: geometry.size.width, height: geometry.size.height)
//                        .ignoresSafeArea()
//                } else {
//                    Image("img\(currentImageIndex)")
//                        .resizable()
//                        .aspectRatio(contentMode: .fill)
//                        .frame(width: geometry.size.width)
//                        .clipped()
//                }
//
//                anchorDots(size: geometry.size)
//
//                if showRGB {
//                    Circle()
//                        .fill(Color.red)
//                        .frame(width: 12, height: 12)
//                        .position(vm.rgbGazeCursor)
//                }
//                
//                if showRGBT {
//                    Circle()
//                        .fill(Color.blue)
//                        .frame(width: 12, height: 12)
//                        .position(vm.socketGazeCursor)
//                }
//            }
//            .onAppear {
//                vm.start()
//                vm.updateScreenThumbnail(index: currentImageIndex)
//            }
//            .onDisappear { vm.stop() }
//            .contentShape(Rectangle())
//            .onTapGesture {
//                withAnimation(.easeInOut(duration: 0.25)) {
//                    if showCamera {
//                        // ✅ First tap after camera → show img1
//                        showCamera = false
//                        tapsSinceCamera = 1      // this is "1st image"
//                        currentImageIndex = 1
//                        vm.updateScreenThumbnail(index: currentImageIndex)
//                    } else {
//                        tapsSinceCamera += 1
//
//                        if tapsSinceCamera > 5 {
//                            // ✅ After 5 images → show camera again
//                            showCamera = true
//                            tapsSinceCamera = 0
//                            // (Keep last template, or you could clear it in VM if you want)
//                        } else {
//                            // ✅ Move to next image
//                            currentImageIndex = currentImageIndex % 20 + 1
//                            vm.updateScreenThumbnail(index: currentImageIndex)
//                        }
//                    }
//                }
//            }
//        }
//    }
//}
//
//
//
