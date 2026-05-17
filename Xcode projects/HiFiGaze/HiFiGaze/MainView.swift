import Foundation
import AVFoundation
import CoreML
import CoreImage
import MediaPipeTasksVision
import SwiftUI

let L_LEFT_CORNER  = 33
let L_RIGHT_CORNER = 133
let R_LEFT_CORNER  = 362
let R_RIGHT_CORNER = 263
let L_IRIS         = 468
let R_IRIS         = 473

struct MainView: View {
    @StateObject private var vm = MainViewModel()
    @State private var screenBrightness: CGFloat = UIScreen.main.brightness
    
    let showRGB: Bool
    let showRGBT: Bool
    let gazeSamples: [GazeSample]
    let onRecalibrate: () -> Void

    @State private var currentImageIndex = 1   // start at img1
    @State private var showCamera = false       // 🔽 start with camera
    @State private var tapsSinceCamera = 0     // count images since last camera

//    var calibrator: PolynomialGazeCalibrator?
    @State private var calibrator: FirstOrderPolynomialGazeCalibrator?
//    @State private var calibrator: SupportVectorGazeCalibrator?

    // debug start
    private func anchorDots(in size: CGSize) -> some View {
        let numRows = 5
        let numCols = 3
        let inset: CGFloat = 60

        return ZStack {
            ForEach(0..<numRows, id: \.self) { row in
                ForEach(0..<numCols, id: \.self) { col in
                    Circle()
                        .fill(Color.black)
                        .frame(width: 14, height: 14)
                        .position(
                            x: inset + CGFloat(col) * ((size.width - inset * 2) / CGFloat(numCols - 1)),
                            y: inset + CGFloat(row) * ((size.height - inset * 2) / CGFloat(numRows - 1))
                        )
                }
            }
        }
    }
    // debug end

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .top) {

                // 🔽 Background: either camera preview or static image
                if showCamera {
                    CameraPreviewView(session: vm.captureSession)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .ignoresSafeArea()
                } else {
                    Image("img\(currentImageIndex)")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: geometry.size.width)
                        .clipped()
                }

                // Debug Start
//                anchorDots(in: geometry.size)
                // Debug End
                
                if screenBrightness < 0.25 {
                    Text("Raise screen brightness")
                        .font(.system(size: 30, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(14)
                        .background(Color.black.opacity(1.0))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .multilineTextAlignment(.center)
                }
                
                if showRGB && screenBrightness >= 0.25 {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 12, height: 12)
                        .position(vm.rgbGazeCursor)
                }

                if showRGBT && screenBrightness >= 0.25 {
                    Circle()
                        .fill(Color.black.opacity(0.05))               // 90% transparent fill
                        .overlay(
                            Circle().stroke(Color.red.opacity(0.4), lineWidth: 0.5) // black boundary
                        )
                        .frame(width: 100, height: 100)                  // 40x40
                        .position(vm.rgbtGazeCursor)
                }
                
                VStack {
                    Spacer()

                    Button(action: {
                        vm.stop()
                        onRecalibrate()
                    }) {
                        Text("Recalibrate")
                            .font(.system(size: 18, weight: .heavy))
                            .foregroundColor(.white)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 10)
                            .background(Color.black.opacity(0.40))
                            .cornerRadius(12)
                    }
                    .padding(.bottom, 20) // adjust height from bottom
                }
                .frame(width: geometry.size.width, height: geometry.size.height)
            }
            .onAppear {
                // debug start
                printAllGazeSamples()
                // debug end
//                calibrator = PolynomialGazeCalibrator(samples: gazeSamples)
                calibrator = FirstOrderPolynomialGazeCalibrator(samples: gazeSamples)
//                calibrator = SupportVectorGazeCalibrator(samples: gazeSamples)
                
                vm.showRGB = showRGB
                vm.showRGBT = showRGBT
                vm.calibrator = calibrator
                vm.viewSize = geometry.size
                
                vm.start()
                vm.updateScreenThumbnail(index: currentImageIndex)
            }
            .onDisappear { vm.stop() }
            .onReceive(NotificationCenter.default.publisher(
                for: UIScreen.brightnessDidChangeNotification
            )) { _ in
                screenBrightness = UIScreen.main.brightness
                vm.isPausedForLowBrightness = screenBrightness < 0.25
            }
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.25)) {
                    if showCamera {
                        // ✅ First tap after camera → show img1
                        showCamera = false
                        tapsSinceCamera = 1      // this is "1st image"
                        currentImageIndex = 1
                        vm.updateScreenThumbnail(index: currentImageIndex)
                    } else {
                        tapsSinceCamera += 1

                        if tapsSinceCamera > 5 {
                            // ✅ After 5 images → show camera again
                            showCamera = true
                            tapsSinceCamera = 0
                            // (Keep last template, or you could clear it in VM if you want)
                        } else {
                            // ✅ Move to next image
                            //currentImageIndex = currentImageIndex % 3 + 1
                            currentImageIndex = 1
                            vm.updateScreenThumbnail(index: currentImageIndex)
                        }
                    }
                }
            }
        }
    }
    
    func printAllGazeSamples() {
        print("\n===== FINAL GAZE SAMPLES =====")
        
        for s in gazeSamples {
            print("[currentIndex: \(s.currentIndex), timeAfterShrinking: \(String(format: "%.2f", s.timeAfterShrinking)), predX: \(s.predX), predY: \(s.predY), gtX: \(s.gtX), gtY: \(s.gtY)]")
        }
        
        print("===== END =====\n")
    }
    
}

final class MainViewModel: ObservableObject {
    @Published var showRGB: Bool = false
    @Published var showRGBT: Bool = false
    var isPausedForLowBrightness = false
    
    //    var calibrator: PolynomialGazeCalibrator?
    var calibrator: FirstOrderPolynomialGazeCalibrator?
//    var calibrator: SupportVectorGazeCalibrator?

    @Published var rgbGazeCursor: CGPoint = .zero
    @Published var rgbtGazeCursor: CGPoint = .zero
    private var templateArray: MLMultiArray?

    var viewSize: CGSize = .zero
    
    private let rgbtFilter = OneEuroFilter()
    
    private let camera = CameraManager()
    private let ciContext = CIContext()
    private var faceLandmarker: FaceLandmarker?

    var captureSession: AVCaptureSession { camera.captureSession }

    private lazy var rgbModel: HiFiGaze_RGB? = {
        let cfg = MLModelConfiguration()
        return try? HiFiGaze_RGB(configuration: cfg)
    }()
    private lazy var rgbtModel: HiFiGaze_RGBT? = {
        let cfg = MLModelConfiguration()
        return try? HiFiGaze_RGBT(configuration: cfg)
    }()

    private var isRunning = false
    private var frameID = 0

    func start() {
        camera.onFrame = { [weak self] sbuf in
            self?.process(sampleBuffer: sbuf)
        }
        try? camera.start()
    }

    func stop() {
        camera.stop()
    }
    
    func updateScreenThumbnail(index: Int) {
        let name = "img\(index)_thumbnail"
        guard let ui = UIImage(named: name) else {
            DispatchQueue.main.async {
                self.templateArray = nil
            }
            return
        }

        let cgImage: CGImage?
        if let cg = ui.cgImage {
            cgImage = cg
        } else if let ci = ui.ciImage {
            cgImage = ciContext.createCGImage(ci, from: ci.extent)
        } else {
            cgImage = nil
        }

        guard let thumbCG = cgImage else {
            DispatchQueue.main.async {
                self.templateArray = nil
            }
            print("⚠️ Could not create CGImage for \(name)")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            let tmpl = cgImageToMLMultiArray_Screen(thumbCG)
            DispatchQueue.main.async {
                self.templateArray = tmpl
            }
        }
    }
     
    private func ensureLandmarker() {
        if faceLandmarker != nil { return }

        let base = BaseOptions()
        base.modelAssetPath = Bundle.main.path(
            forResource: "face_landmarker",
            ofType: "task"
        )!

        let opts = FaceLandmarkerOptions()
        opts.baseOptions = base
        opts.numFaces = 1
        opts.runningMode = .image

        faceLandmarker = try? FaceLandmarker(options: opts)
    }
    
    // debug start
//    private var fpsCounter = 0
//    private var lastFPSUpdateTime: CFTimeInterval = CACurrentMediaTime()
    // debug end
    
    
    private func process(sampleBuffer: CMSampleBuffer) {
        // debug start
        // fps check
//        fpsCounter += 1
//
//        let now2 = CACurrentMediaTime()
//        let dt = now - lastFPSUpdateTime
//
//        if dt >= 1.0 {
//            let fps = Double(fpsCounter) / dt
//            print(String(format: "🔥 process() FPS: %.1f", fps))
//
//            fpsCounter = 0
//            lastFPSUpdateTime = now2
//        }
        // debug end
        
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }
        
        guard !isPausedForLowBrightness else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        ensureLandmarker()
        guard let landmarker = faceLandmarker else { return }

        do {
            let mpImage = try MPImage(pixelBuffer: pixelBuffer, orientation: .up)
            let result = try landmarker.detect(image: mpImage)
            guard let lms = result.faceLandmarks.first else { return }
            
            guard
                let eyeTensor = makeEyeTensor(pixelBuffer: pixelBuffer, lms: lms, frameID: frameID),
                let eyeLmk = makeEyeCornerArray(lms: lms)
            else {
                return
            }
                    
            // debug start
//            frameID += 1
//            savePixelBufferWithLandmarks(pixelBuffer: pixelBuffer, lms: lms, frameID: frameID)
//            saveEyeTensorAsPNG(eyeTensor, frameID: frameID)
//            saveEyeLmkAsJSON(eyeLmk, frameID: frameID)
            // debug end
            
//            if showRGB, let rgbModel = rgbModel {
//                let rgbPred = try rgbModel.prediction(
//                    eye: eyeTensor,
//                    eye_lmk: eyeLmk
//                )
//
//                let rx = rgbPred.output[0].floatValue * 1000
//                let ry = rgbPred.output[1].floatValue * 1000
//
//                DispatchQueue.main.async {
//                    self.rgbGazeCursor = CGPoint(
//                        x: CGFloat(rx),
//                        y: CGFloat(ry)
//                    )
//                    let corrected = self.calibrator?.correct(
//                        predX: Double(rx),
//                        predY: Double(ry)
//                    ) ?? CGPoint(x: CGFloat(rx), y: CGFloat(ry))
//
//                    self.rgbGazeCursor = corrected
//                }
//            }
            
            if showRGBT, let rgbtModel = rgbtModel,
               let screen = templateArray {

                let rgbtPred = try rgbtModel.prediction(
                    eye: eyeTensor,
                    screen: screen,
                    eye_lmk: eyeLmk
                )
                let raw = CGPoint(
                    x: CGFloat(rgbtPred.output[0].floatValue),
                    y: CGFloat(rgbtPred.output[1].floatValue)
                )
                let filtered = rgbtFilter.filter(raw)
                
                DispatchQueue.main.async {
//                    self.rgbtGazeCursor = CGPoint(x: filtered.x * 1000, y: filtered.y * 1000)
                    let px = Double(filtered.x * 1000)
                    let py = Double(filtered.y * 1000)

                    let corrected = self.calibrator?.correct(
                        predX: px,
                        predY: py
                    ) ?? CGPoint(x: CGFloat(px), y: CGFloat(py))

//                    self.rgbtGazeCursor = corrected
                    let w = self.viewSize.width
                    let h = self.viewSize.height

                    let clippedX = min(max(corrected.x, 5), w-5)
                    let clippedY = min(max(corrected.y, 5), h-5)

                    self.rgbtGazeCursor = CGPoint(x: clippedX, y: clippedY)
                }
            }
        } catch {
            print("❌ Frame \(frameID) error:", error)
        }
    }
}

func lmkPx(_ lms: [NormalizedLandmark], _ idx: Int, _ w: CGFloat, _ h: CGFloat) -> CGPoint {
    CGPoint(
        x: CGFloat(lms[idx].x) * w,
        y: CGFloat(lms[idx].y) * h
    )
}

func cropEyeCI(pixelBuffer: CVPixelBuffer, lms: [NormalizedLandmark], left: Int, right: Int, center: Int, tag: String, frameID: Int) -> CIImage? {
    let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
    let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
   
    let L0 = lmkPx(lms, left,   w, h)
    let R0 = lmkPx(lms, right,  w, h)
    let C0 = lmkPx(lms, center, w, h)

    let L = CGPoint(x: L0.x, y: h - L0.y)
    let R = CGPoint(x: R0.x, y: h - R0.y)
    let C = CGPoint(x: C0.x, y: h - C0.y)

    let dx = R.x - L.x
    let dy = R.y - L.y
    let eyeWidth = hypot(dx, dy)

    let angle = -atan2(dy, dx)
    let scale = 250.0 / eyeWidth

    let src = CIImage(cvPixelBuffer: pixelBuffer)
    
    var t = CGAffineTransform.identity
    t = t.scaledBy(x: scale, y: scale)
    t = t.rotated(by: angle)
    t = t.translatedBy(x: -C.x, y: -C.y)

    let transformed = src.transformed(by: t)
    let cropped = transformed.cropped(
        to: CGRect(
            x: -150,
            y: -150,
            width: 300,
            height: 300
        )
    )

    return cropped
}

func makeEyeTensor(pixelBuffer: CVPixelBuffer, lms: [NormalizedLandmark], frameID: Int) -> MLMultiArray? {
    guard
        let leftCI = cropEyeCI(
            pixelBuffer: pixelBuffer,
            lms: lms,
            left: L_LEFT_CORNER,
            right: L_RIGHT_CORNER,
            center: L_IRIS,
            tag: "L",
            frameID: frameID
        ),
        let rightCI = cropEyeCI(
            pixelBuffer: pixelBuffer,
            lms: lms,
            left: R_LEFT_CORNER,
            right: R_RIGHT_CORNER,
            center: R_IRIS,
            tag: "R",
            frameID: frameID
        )
    else { return nil }

    let leftN = leftCI.transformed(
        by: CGAffineTransform(
            translationX: -leftCI.extent.origin.x,
            y: -leftCI.extent.origin.y
        )
    )

    let rightN = rightCI.transformed(
        by: CGAffineTransform(
            translationX: -rightCI.extent.origin.x,
            y: -rightCI.extent.origin.y
        )
    )

    let canvas = CIImage(color: .black)
        .cropped(to: CGRect(x: 0, y: 0, width: 600, height: 300))

    var combined = leftN
        .composited(over: canvas)
    combined = rightN
        .transformed(by: CGAffineTransform(translationX: 300, y: 0))
        .composited(over: combined)

    guard let cg = ciImageToCGImage(combined) else {
        print("❌ CIImage → CGImage failed")
        return nil
    }

    return cgImageToMLMultiArrayNoResize_Eye(cg)
}

let sharedCIContext = CIContext(options: nil)

func ciImageToCGImage(_ ciImage: CIImage) -> CGImage? {
    sharedCIContext.createCGImage(ciImage, from: ciImage.extent)
}

func makeEyeCornerArray(lms: [NormalizedLandmark]) -> MLMultiArray? {
    guard let out = try? MLMultiArray(
        shape: [1, 8],
        dataType: .float32
    ) else { return nil }

    let idx = [
        R_LEFT_CORNER, R_RIGHT_CORNER,
        L_RIGHT_CORNER, L_LEFT_CORNER
    ]

    for i in 0..<4 {
        out[i * 2 + 0] = NSNumber(value: lms[idx[i]].x)
        out[i * 2 + 1] = NSNumber(value: lms[idx[i]].y)
    }

    return out
}

func cgImageToMLMultiArrayNoResize_Eye(
    _ cgImage: CGImage
) -> MLMultiArray? {

    let width = cgImage.width
    let height = cgImage.height

    // Expect 600x300
    guard width == 600, height == 300 else {
        print("❌ Unexpected CGImage size:", width, height)
        return nil
    }

    guard let arr = try? MLMultiArray(
        shape: [1, 3, 300, 600],
        dataType: .float32
    ) else {
        return nil
    }

    let strideC = arr.strides[1].intValue
    let strideH = arr.strides[2].intValue
    let strideW = arr.strides[3].intValue

    var rgba = [UInt8](repeating: 0, count: width * height * 4)

    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: &rgba,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: cs,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        return nil
    }

    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)

    for y in 0..<height {
        for x in 0..<width {
            let idx = y * width + x
            let base = idx * 4

            let r = Float(rgba[base + 0]) / 255.0
            let g = Float(rgba[base + 1]) / 255.0
            let b = Float(rgba[base + 2]) / 255.0

            ptr[0 * strideC + y * strideH + x * strideW] = r
            ptr[1 * strideC + y * strideH + x * strideW] = g
            ptr[2 * strideC + y * strideH + x * strideW] = b
        }
    }

    return arr
}

func cgImageToMLMultiArray_Screen(_ cgImage: CGImage) -> MLMultiArray? {
    let width  = cgImage.width    // should be 64
    let height = cgImage.height   // should be 128

    // Safety check — fail loudly if asset is wrong
    assert(width == 64 && height == 128,
           "Screen thumbnail must be 64x128, got \(width)x\(height)")

    let pixelCount = width * height

    guard let array = try? MLMultiArray(
        shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
        dataType: .float32
    ) else { return nil }

    guard let data = cgImage.dataProvider?.data,
          let ptr = CFDataGetBytePtr(data) else { return nil }

    let bytesPerPixel = cgImage.bitsPerPixel / 8
    let bytesPerRow   = cgImage.bytesPerRow

    for y in 0..<height {
        for x in 0..<width {
            let i = y * width + x
            let p = y * bytesPerRow + x * bytesPerPixel

            let r = Float(ptr[p + 0]) / 255
            let g = Float(ptr[p + 1]) / 255
            let b = Float(ptr[p + 2]) / 255

            array[i]                    = NSNumber(value: r)
            array[pixelCount + i]       = NSNumber(value: g)
            array[2 * pixelCount + i]   = NSNumber(value: b)
        }
    }
    return array
}

struct PolynomialGazeCalibrator {
    // gtX = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
    private let coeffX: [Double]
    private let coeffY: [Double]

    init?(samples: [GazeSample]) {
        guard samples.count >= 6 else { return nil }

        let features = samples.map {
            PolynomialGazeCalibrator.phi(x: $0.predX, y: $0.predY)
        }

        let targetX = samples.map { $0.gtX }
        let targetY = samples.map { $0.gtY }

        guard
            let cx = Self.solveLeastSquares(A: features, b: targetX),
            let cy = Self.solveLeastSquares(A: features, b: targetY)
        else {
            return nil
        }

        self.coeffX = cx
        self.coeffY = cy
        
        Self.printFormula(name: "gtX", coeffs: cx)
        Self.printFormula(name: "gtY", coeffs: cy)
    }

    private static func printFormula(name: String, coeffs: [Double]) {
        guard coeffs.count == 6 else { return }

        print("""
        \(name) = \(coeffs[0]) \
        + \(coeffs[1]) * predX \
        + \(coeffs[2]) * predY \
        + \(coeffs[3]) * predX^2 \
        + \(coeffs[4]) * predX * predY \
        + \(coeffs[5]) * predY^2
        """)
    }
    
    func correct(predX x: Double, predY y: Double) -> CGPoint {
        let f = Self.phi(x: x, y: y)

        let gx = zip(coeffX, f).map(*).reduce(0, +)
        let gy = zip(coeffY, f).map(*).reduce(0, +)

        return CGPoint(x: gx, y: gy)
    }

    private static func phi(x: Double, y: Double) -> [Double] {
        [
            1.0,
            x,
            y,
            x * x,
            x * y,
            y * y
        ]
    }

    private static func solveLeastSquares(A: [[Double]], b: [Double]) -> [Double]? {
        let m = A.count
        let n = A[0].count

        // Compute AtA and Atb
        var AtA = Array(
            repeating: Array(repeating: 0.0, count: n),
            count: n
        )
        var Atb = Array(repeating: 0.0, count: n)

        for i in 0..<m {
            for r in 0..<n {
                Atb[r] += A[i][r] * b[i]
                for c in 0..<n {
                    AtA[r][c] += A[i][r] * A[i][c]
                }
            }
        }

        // Small ridge regularization for stability
        let lambda = 1e-6
        for i in 0..<n {
            AtA[i][i] += lambda
        }

        return gaussianSolve(AtA, Atb)
    }

    private static func gaussianSolve(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        var A = A
        var b = b
        let n = b.count

        for i in 0..<n {
            var maxRow = i
            var maxVal = abs(A[i][i])

            for r in (i + 1)..<n {
                let v = abs(A[r][i])
                if v > maxVal {
                    maxVal = v
                    maxRow = r
                }
            }

            if maxVal < 1e-12 {
                return nil
            }

            if maxRow != i {
                A.swapAt(i, maxRow)
                b.swapAt(i, maxRow)
            }

            let pivot = A[i][i]

            for c in i..<n {
                A[i][c] /= pivot
            }
            b[i] /= pivot

            for r in 0..<n {
                if r == i { continue }

                let factor = A[r][i]

                for c in i..<n {
                    A[r][c] -= factor * A[i][c]
                }
                b[r] -= factor * b[i]
            }
        }

        return b
    }
}

struct FirstOrderPolynomialGazeCalibrator {
    // gtX = c0 + c1*predX + c2*predY
    // gtY = c0 + c1*predX + c2*predY
    private let coeffX: [Double]
    private let coeffY: [Double]

    init?(samples: [GazeSample]) {
        guard samples.count >= 3 else { return nil }

        let features = samples.map {
            FirstOrderPolynomialGazeCalibrator.phi(x: $0.predX, y: $0.predY)
        }

        let targetX = samples.map { $0.gtX }
        let targetY = samples.map { $0.gtY }

        guard
            let cx = Self.solveLeastSquares(A: features, b: targetX),
            let cy = Self.solveLeastSquares(A: features, b: targetY)
        else {
            return nil
        }

        self.coeffX = cx
        self.coeffY = cy

        Self.printFormula(name: "gtX", coeffs: cx)
        Self.printFormula(name: "gtY", coeffs: cy)
    }

    func correct(predX x: Double, predY y: Double) -> CGPoint {
        let f = Self.phi(x: x, y: y)

        let gx = zip(coeffX, f).map(*).reduce(0, +)
        let gy = zip(coeffY, f).map(*).reduce(0, +)

        return CGPoint(x: gx, y: gy)
    }

    private static func phi(x: Double, y: Double) -> [Double] {
        [
            1.0,
            x,
            y
        ]
    }

    private static func printFormula(name: String, coeffs: [Double]) {
        guard coeffs.count == 3 else { return }

        print("""
        \(name) = \(coeffs[0]) \
        + \(coeffs[1]) * predX \
        + \(coeffs[2]) * predY
        """)
    }

    private static func solveLeastSquares(A: [[Double]], b: [Double]) -> [Double]? {
        let m = A.count
        let n = A[0].count

        var AtA = Array(
            repeating: Array(repeating: 0.0, count: n),
            count: n
        )
        var Atb = Array(repeating: 0.0, count: n)

        for i in 0..<m {
            for r in 0..<n {
                Atb[r] += A[i][r] * b[i]

                for c in 0..<n {
                    AtA[r][c] += A[i][r] * A[i][c]
                }
            }
        }

        let lambda = 1e-6
        for i in 0..<n {
            AtA[i][i] += lambda
        }

        return gaussianSolve(AtA, Atb)
    }

    private static func gaussianSolve(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        var A = A
        var b = b
        let n = b.count

        for i in 0..<n {
            var maxRow = i
            var maxVal = abs(A[i][i])

            for r in (i + 1)..<n {
                let v = abs(A[r][i])
                if v > maxVal {
                    maxVal = v
                    maxRow = r
                }
            }

            if maxVal < 1e-12 {
                return nil
            }

            if maxRow != i {
                A.swapAt(i, maxRow)
                b.swapAt(i, maxRow)
            }

            let pivot = A[i][i]

            for c in i..<n {
                A[i][c] /= pivot
            }
            b[i] /= pivot

            for r in 0..<n {
                if r == i { continue }

                let factor = A[r][i]

                for c in i..<n {
                    A[r][c] -= factor * A[i][c]
                }
                b[r] -= factor * b[i]
            }
        }

        return b
    }
}

struct SupportVectorGazeCalibrator {
    private let trainX: [[Double]]
    private let alphaX: [Double]
    private let alphaY: [Double]
    private let biasX: Double
    private let biasY: Double

    private let meanPredX: Double
    private let meanPredY: Double
    private let stdPredX: Double
    private let stdPredY: Double

    private let gamma: Double = 0.8
    private let epsilon: Double = 8.0
    private let learningRate: Double = 0.01
    private let lambda: Double = 0.001
    private let epochs: Int = 2000

    init?(samples: [GazeSample]) {
        guard samples.count >= 3 else { return nil }

        let predXs = samples.map { $0.predX }
        let predYs = samples.map { $0.predY }

        let localMeanPredX = predXs.reduce(0, +) / Double(predXs.count)
        let localMeanPredY = predYs.reduce(0, +) / Double(predYs.count)

        let localStdPredX = max(Self.std(predXs, mean: localMeanPredX), 1e-6)
        let localStdPredY = max(Self.std(predYs, mean: localMeanPredY), 1e-6)

        let localTrainX = samples.map {
            [
                ($0.predX - localMeanPredX) / localStdPredX,
                ($0.predY - localMeanPredY) / localStdPredY
            ]
        }

        let targetX = samples.map { $0.gtX }
        let targetY = samples.map { $0.gtY }

        let localBiasX = targetX.reduce(0, +) / Double(targetX.count)
        let localBiasY = targetY.reduce(0, +) / Double(targetY.count)

        let localAlphaX = Self.trainSVR(
            inputs: localTrainX,
            targets: targetX.map { $0 - localBiasX },
            gamma: gamma,
            epsilon: epsilon,
            learningRate: learningRate,
            lambda: lambda,
            epochs: epochs
        )

        let localAlphaY = Self.trainSVR(
            inputs: localTrainX,
            targets: targetY.map { $0 - localBiasY },
            gamma: gamma,
            epsilon: epsilon,
            learningRate: learningRate,
            lambda: lambda,
            epochs: epochs
        )

        self.meanPredX = localMeanPredX
        self.meanPredY = localMeanPredY
        self.stdPredX = localStdPredX
        self.stdPredY = localStdPredY
        self.trainX = localTrainX
        self.biasX = localBiasX
        self.biasY = localBiasY
        self.alphaX = localAlphaX
        self.alphaY = localAlphaY

        print("SVR calibrator trained")
        print("gtX = SVR_RBF(predX, predY)")
        print("gtY = SVR_RBF(predX, predY)")
        print("num support vectors:", samples.count)
        print("gamma:", gamma, "epsilon:", epsilon)
    }

    func correct(predX x: Double, predY y: Double) -> CGPoint {
        let input = [
            (x - meanPredX) / stdPredX,
            (y - meanPredY) / stdPredY
        ]

        let gx = biasX + Self.predict(
            input: input,
            trainX: trainX,
            alpha: alphaX,
            gamma: gamma
        )

        let gy = biasY + Self.predict(
            input: input,
            trainX: trainX,
            alpha: alphaY,
            gamma: gamma
        )

        return CGPoint(x: gx, y: gy)
    }

    private static func trainSVR(
        inputs: [[Double]],
        targets: [Double],
        gamma: Double,
        epsilon: Double,
        learningRate: Double,
        lambda: Double,
        epochs: Int
    ) -> [Double] {
        let n = inputs.count
        var alpha = Array(repeating: 0.0, count: n)

        for _ in 0..<epochs {
            for i in 0..<n {
                let pred = predict(
                    input: inputs[i],
                    trainX: inputs,
                    alpha: alpha,
                    gamma: gamma
                )

                let error = pred - targets[i]

                if abs(error) > epsilon {
                    let direction = error > 0 ? 1.0 : -1.0

                    for j in 0..<n {
                        let k = rbf(inputs[j], inputs[i], gamma: gamma)
                        alpha[j] -= learningRate * direction * k
                        alpha[j] *= (1.0 - learningRate * lambda)
                    }
                }
            }
        }

        return alpha
    }

    private static func predict(
        input: [Double],
        trainX: [[Double]],
        alpha: [Double],
        gamma: Double
    ) -> Double {
        var result = 0.0

        for i in 0..<trainX.count {
            result += alpha[i] * rbf(trainX[i], input, gamma: gamma)
        }

        return result
    }

    private static func rbf(_ a: [Double], _ b: [Double], gamma: Double) -> Double {
        let dx = a[0] - b[0]
        let dy = a[1] - b[1]
        return exp(-gamma * (dx * dx + dy * dy))
    }

    private static func std(_ values: [Double], mean: Double) -> Double {
        let variance = values
            .map { pow($0 - mean, 2) }
            .reduce(0, +) / Double(values.count)

        return sqrt(variance)
    }
}


// debug start
//func savePixelBufferWithLandmarks(pixelBuffer: CVPixelBuffer, lms: [NormalizedLandmark], frameID: Int) {
//    let width  = CVPixelBufferGetWidth(pixelBuffer)
//    let height = CVPixelBufferGetHeight(pixelBuffer)
//
//    // --- Convert pixelBuffer → CIImage ---
//    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
//
//    let ciContext = CIContext(options: nil)
//
//    guard let cgImage = ciContext.createCGImage(
//        ciImage,
//        from: CGRect(x: 0, y: 0, width: width, height: height)
//    ) else {
//        print("❌ Failed to create CGImage")
//        return
//    }
//
//    // --- Create UIKit graphics context ---
//    UIGraphicsBeginImageContextWithOptions(
//        CGSize(width: width, height: height),
//        false,
//        1.0
//    )
//    defer { UIGraphicsEndImageContext() }
//
//    guard let ctx = UIGraphicsGetCurrentContext() else { return }
//
//    ctx.translateBy(x: 0, y: CGFloat(height))
//    ctx.scaleBy(x: 1, y: -1)
//
//    // --- Draw camera image (now upright) ---
//    ctx.draw(
//        cgImage,
//        in: CGRect(x: 0, y: 0, width: width, height: height)
//    )
//
//    // --- Draw landmarks ---
//    ctx.setFillColor(UIColor.red.cgColor)
//    let r: CGFloat = 10.0
//
//    for lm in lms {
//        let x = CGFloat(lm.x) * CGFloat(width)
//        let y = (1.0 - CGFloat(lm.y)) * CGFloat(height)
//
//        ctx.fillEllipse(
//            in: CGRect(
//                x: x - r,
//                y: y - r,
//                width: r * 2,
//                height: r * 2
//            )
//        )
//    }
//
//    // --- Export PNG ---
//    guard let outImage = UIGraphicsGetImageFromCurrentImageContext(),
//          let data = outImage.pngData()
//    else { return }
//
//    let url = FileManager.default
//        .urls(for: .documentDirectory, in: .userDomainMask)[0]
//        .appendingPathComponent(
//            String(format: "frame_wLMKS_%d.png", frameID)
//        )
//
//    do {
//        try data.write(to: url)
//        print("🟥 Saved landmark debug:", url.lastPathComponent)
//    } catch {
//        print("❌ Failed to write landmark debug:", error)
//    }
//}
//
//func saveEyeTensorAsPNG(_ eye: MLMultiArray, frameID: Int) {
//    guard eye.dataType == .float32 else {
//        print("❌ eye dtype not float32")
//        return
//    }
//
//    let shape = eye.shape.map { Int(truncating: $0) }
//    guard shape.count == 4,
//          shape[0] == 1,
//          shape[1] == 3
//    else {
//        print("❌ Unexpected eye shape:", shape)
//        return
//    }
//
//    let height = shape[2]   // 300
//    let width  = shape[3]   // 600
//
//    let strideC = eye.strides[1].intValue
//    let strideH = eye.strides[2].intValue
//    let strideW = eye.strides[3].intValue
//
//    let ptr = eye.dataPointer.assumingMemoryBound(to: Float.self)
//
//    var rgba = [UInt8](repeating: 0, count: width * height * 4)
//
//    for y in 0..<height {
//        for x in 0..<width {
//            let idx = y * width + x
//            let base = idx * 4
//
//            let r = ptr[0 * strideC + y * strideH + x * strideW]
//            let g = ptr[1 * strideC + y * strideH + x * strideW]
//            let b = ptr[2 * strideC + y * strideH + x * strideW]
//
//            rgba[base + 0] = UInt8(clamping: Int(r * 255))
//            rgba[base + 1] = UInt8(clamping: Int(g * 255))
//            rgba[base + 2] = UInt8(clamping: Int(b * 255))
//            rgba[base + 3] = 255
//        }
//    }
//
//    let cs = CGColorSpaceCreateDeviceRGB()
//    guard let ctx = CGContext(
//        data: &rgba,
//        width: width,
//        height: height,
//        bitsPerComponent: 8,
//        bytesPerRow: width * 4,
//        space: cs,
//        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
//    ),
//    let cgImage = ctx.makeImage()
//    else {
//        print("❌ CGContext failed")
//        return
//    }
//
//    let uiImage = UIImage(cgImage: cgImage)
//
//    let url = FileManager.default
//        .urls(for: .documentDirectory, in: .userDomainMask)[0]
//        .appendingPathComponent("eye_\(frameID).png")
//
//    do {
//        try uiImage.pngData()?.write(to: url)
//        print("✅ Saved eye:", url.lastPathComponent)
//    } catch {
//        print("❌ Failed to save eye PNG:", error)
//    }
//}
//
//func saveEyeLmkAsJSON(_ eyeLmk: MLMultiArray, frameID: Int) {
//    guard eyeLmk.dataType == .float32 else {
//        print("❌ eye_lmk dtype not float32")
//        return
//    }
//
//    guard eyeLmk.count == 8 else {
//        print("❌ eye_lmk count != 8:", eyeLmk.count)
//        return
//    }
//
//    let stride = eyeLmk.strides.last?.intValue ?? 1
//    let ptr = eyeLmk.dataPointer.assumingMemoryBound(to: Float.self)
//
//    var arr: [Float] = []
//    arr.reserveCapacity(8)
//
//    for i in 0..<8 {
//        arr.append(ptr[i * stride])
//    }
//
//    let url = FileManager.default
//        .urls(for: .documentDirectory, in: .userDomainMask)[0]
//        .appendingPathComponent("lmk_\(frameID).json")
//
//    do {
//        let data = try JSONSerialization.data(
//            withJSONObject: arr,
//            options: [.prettyPrinted]
//        )
//        try data.write(to: url)
//        print("✅ Saved eye_lmk:", url.lastPathComponent)
//    } catch {
//        print("❌ Failed to save eye_lmk JSON:", error)
//    }
//}
// debug end
