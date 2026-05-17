import SwiftUI
import AVFoundation
import CoreML
import CoreImage
import MediaPipeTasksVision
import UIKit

struct GazeSample: Identifiable {
    let id = UUID()
    let currentIndex: Int
    let timeAfterShrinking: Double
    let gtX: Double
    let gtY: Double
    let predX: Double
    let predY: Double
}

struct CalibrationView: View {
    @StateObject private var vm = CalibrationViewModel()

    let showRGB: Bool
    let showRGBT: Bool
    let calibrationPointCount: Int
    let shrinkDuration: Double
    let onCalibrationFinished: ([GazeSample]) -> Void
    @State private var positions: [CGPoint] = []
    
    @State private var remainingSeconds = 5
    @State private var countdownComplete = false
    @State private var sequenceStarted = false

    @State private var currentIndex = 0
    @State private var shrinkInnerCircle = false
    
    // debug start
    @State private var shrinkStartTime: CFTimeInterval = 0
    // debug end
    
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    let marginPadding = 60.0
    private let moveDuration = 0.6
    
    private var innerScale: CGFloat {
        shrinkInnerCircle ? 14.0 / 50.0 : 1.0
    }
    
    // start with black. then alternating pink. and purple.
    private var dotColors: [Color] {
        var colors: [Color] = [.black]

        for i in 1..<calibrationPointCount+1 {
            colors.append(i % 2 == 1 ? .pink : .purple)
        }

        return colors
    }
    
    var body: some View {
        GeometryReader { geo in
            let currentPosition = positions.isEmpty ? .zero : positions[currentIndex]
            let dotColor = dotColors[currentIndex]

            ZStack {
                Image("img1")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geo.size.width)
                    .clipped()
                
//                Image("img\(currentImageIndex)")
//                    .resizable()
//                    .aspectRatio(contentMode: .fill)
//                    .frame(width: geometry.size.width)
//                    .clipped()

                if !countdownComplete {
                    Text("Follow the dot with your eyes as it moves around the screen.")
                        .font(.system(size: 30))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .padding(20) // padding INSIDE the rounded box
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color.black)
                        )
                        .padding(.horizontal, 40)
                        .position(x: geo.size.width / 2,
                                  y: geo.size.height / 2 - 120)
                }

                ZStack {
                    if countdownComplete {
                        Circle()
                            .fill(dotColor.opacity(0.3))
                            .frame(width: 50, height: 50)

                        Circle()
                            .fill(dotColor.opacity(0.8))
                            .frame(width: 50, height: 50)
                            .scaleEffect(innerScale, anchor: .center)
                    } else {
                        Circle()
                            .fill(Color.black)
                            .frame(width: 50, height: 50)

                        Text("\(remainingSeconds)")
                            .font(.system(size: 30, weight: .regular))
                            .foregroundColor(.white)
                    }
                }
                .frame(width: 50, height: 50)
                .position(currentPosition)
                
//                if countdownComplete && showRGB {
//                    Circle()
//                        .fill(Color.red)
//                        .frame(width: 12, height: 12)
//                        .position(vm.rgbGazeCursor)
//                }

//                if countdownComplete && showRGBT {
//                    Circle()
//                        .fill(Color.red)
//                        .frame(width: 12, height: 12)
//                        .position(vm.rgbtGazeCursor)
//                }
            }
           
            .onAppear {
                if positions.isEmpty {
                    positions = makePositions(in: geo.size)
                    vm.setPositions(positions)
                }
                print("width: \(geo.size.width), height: \(geo.size.height)")
                vm.showRGB = showRGB
                vm.showRGBT = showRGBT
                vm.start()
                vm.updateScreenThumbnail(index: 1)
            }
            .onDisappear {
               vm.stop()
            }
        }
        .onReceive(timer) { _ in
            guard !sequenceStarted else { return }

            if remainingSeconds > 1 {
                remainingSeconds -= 1
            } else {
                sequenceStarted = true
                countdownComplete = true
                startCalibrationAnimation()
            }
        }
    }
    
    private func makePositions(in size: CGSize) -> [CGPoint] {
        let centerX = size.width / 2
        let centerY = size.height / 2
        print("calibrationPointCount: \(calibrationPointCount)")
        
        if calibrationPointCount == 5 {
            return [
                CGPoint(x: centerX, y: centerY),
                CGPoint(x: marginPadding, y: marginPadding),
                CGPoint(x: size.width - marginPadding, y: marginPadding),
                CGPoint(x: size.width - marginPadding, y: size.height - marginPadding),
                CGPoint(x: marginPadding, y: size.height - marginPadding),
                CGPoint(x: centerX, y: centerY)
            ]
        } else {
            return [
                CGPoint(x: centerX, y: centerY),
                CGPoint(x: marginPadding, y: marginPadding),
                CGPoint(x: centerX, y: marginPadding),
                CGPoint(x: size.width - marginPadding, y: marginPadding),
                CGPoint(x: size.width - marginPadding, y: centerY),
                CGPoint(x: size.width - marginPadding, y: size.height - marginPadding),
                CGPoint(x: centerX, y: size.height - marginPadding),
                CGPoint(x: marginPadding, y: size.height - marginPadding),
                CGPoint(x: marginPadding, y: centerY),
                CGPoint(x: centerX, y: centerY)
            ]
        }
    }
    
    private func startCalibrationAnimation() {
        guard !positions.isEmpty else { return }
        vm.setPredictionEnabled(true)
        runStep(targetIndex: 1)
    }
    
    private func runStep(targetIndex: Int) {
        withAnimation(.easeInOut(duration: moveDuration)) {
            currentIndex = targetIndex
            shrinkInnerCircle = false
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + moveDuration) {
            // ✅ mark shrink start time
            shrinkStartTime = CACurrentMediaTime()

            // ✅ also tell ViewModel (we'll use it there)
            vm.updateTiming(
                currentIndex: targetIndex,
                shrinkStartTime: shrinkStartTime,
                shrinkDuration: shrinkDuration
            )
        
            withAnimation(.linear(duration: shrinkDuration)) {
                shrinkInnerCircle = true
            }
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + moveDuration + shrinkDuration) {
            let nextIndex = targetIndex + 1
            
            if nextIndex < positions.count {
                runStep(targetIndex: nextIndex)
            } else {
                vm.setPredictionEnabled(false)
                //vm.printAllGazeSamples()
                onCalibrationFinished(vm.getGazeSamples())
                return
            }
        }
    }
}

final class CalibrationViewModel: ObservableObject {
    @Published var showRGB: Bool = false
    @Published var showRGBT: Bool = false

    @Published var rgbGazeCursor: CGPoint = .zero
    @Published var rgbtGazeCursor: CGPoint = .zero

    private var templateArray: MLMultiArray?
    private let rgbtFilter = OneEuroFilter()
    
    private let camera = CameraManager()
    private let ciContext = CIContext()
    private var faceLandmarker: FaceLandmarker?
    
    // debug start
    private var currentIndex: Int = 0
    private var shrinkStartTime: CFTimeInterval = 0
    private var shrinkDuration: CFTimeInterval = 1.8
    
    func updateTiming(currentIndex: Int,
                      shrinkStartTime: CFTimeInterval,
                      shrinkDuration: CFTimeInterval) {
        self.currentIndex = currentIndex
        self.shrinkStartTime = shrinkStartTime
        self.shrinkDuration = shrinkDuration
    }
    // debug end
    private var positions: [CGPoint] = []

    func setPositions(_ positions: [CGPoint]) {
        self.positions = positions
    }
    
  
    
    private var gazeSamples: [GazeSample] = []
    func getGazeSamples() -> [GazeSample] {
        return gazeSamples
    }


    var captureSession: AVCaptureSession {
        camera.captureSession
    }

    private lazy var rgbModel: HiFiGaze_RGB? = {
        let cfg = MLModelConfiguration()
        return try? HiFiGaze_RGB(configuration: cfg)
    }()

    private lazy var rgbtModel: HiFiGaze_RGBT? = {
        let cfg = MLModelConfiguration()
        return try? HiFiGaze_RGBT(configuration: cfg)
    }()

    private var isRunning = false
    private var predictionEnabled = false
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

    func setPredictionEnabled(_ enabled: Bool) {
        predictionEnabled = enabled
    }
    
    func updateScreenThumbnail(index: Int) {
        let name = "img\(index)_thumbnail"

        guard let ui = UIImage(named: name) else {
            DispatchQueue.main.async {
                self.templateArray = nil
            }
            print("⚠️ Could not load \(name)")
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

    private func process(sampleBuffer: CMSampleBuffer) {
        guard predictionEnabled else { return }
        guard !isRunning else { return }

        isRunning = true
        defer { isRunning = false }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        ensureLandmarker()
        guard let landmarker = faceLandmarker else { return }

        do {
            let mpImage = try MPImage(pixelBuffer: pixelBuffer, orientation: .up)
            let result = try landmarker.detect(image: mpImage)

            guard let lms = result.faceLandmarks.first else { return }

            guard
                let eyeTensor = makeEyeTensor(
                    pixelBuffer: pixelBuffer,
                    lms: lms,
                    frameID: frameID
                ),
                let eyeLmk = makeEyeCornerArray(lms: lms)
            else { return }

            frameID += 1

            if showRGB, let rgbModel {
                let pred = try rgbModel.prediction(
                    eye: eyeTensor,
                    eye_lmk: eyeLmk
                )

                let x = CGFloat(pred.output[0].floatValue * 1000)
                let y = CGFloat(pred.output[1].floatValue * 1000)

                DispatchQueue.main.async {
                    self.rgbGazeCursor = CGPoint(x: x, y: y)
                }
            }

            if showRGBT,
               let rgbtModel,
               let screen = templateArray {

                let pred = try rgbtModel.prediction(
                    eye: eyeTensor,
                    screen: screen,
                    eye_lmk: eyeLmk
                )

                let raw = CGPoint(
                    x: CGFloat(pred.output[0].floatValue),
                    y: CGFloat(pred.output[1].floatValue)
                )
                
                let now = CACurrentMediaTime()
                let elapsed = now - shrinkStartTime
                let clamped = max(0, min(elapsed, shrinkDuration))
                    
                let sampleStartTime: CFTimeInterval = 1.0
                let sampleEndTime: CFTimeInterval = shrinkDuration - 0.3

                if clamped >= sampleStartTime && clamped <= sampleEndTime {
                    let sample = GazeSample(
                        currentIndex: currentIndex,
                        timeAfterShrinking: Double(clamped),
                        gtX: Double(positions[currentIndex].x),
                        gtY: Double(positions[currentIndex].y),
                        predX: Double(raw.x * 1000),
                        predY: Double(raw.y * 1000)
                    )
                    gazeSamples.append(sample)
                }

                DispatchQueue.main.async {
                    self.rgbtGazeCursor = CGPoint(
                        x: raw.x * 1000,
                        y: raw.y * 1000
                    )
                }
            }
        } catch {
            print("❌ Calibration frame error:", error)
        }
    }
}
