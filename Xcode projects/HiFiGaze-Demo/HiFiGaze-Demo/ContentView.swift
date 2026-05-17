import SwiftUI
import AVFoundation
import CoreImage
import CoreGraphics
import UIKit
import MediaPipeTasksVision

let L_LEFT_CORNER  = 33
let L_RIGHT_CORNER = 133
let L_IRIS         = 468

struct ContentView: View {
    @StateObject private var vm = EyeCropViewModel()

    @State private var currentImageIndex = 1
    @State private var showCamera = true
    @State private var completedImageRounds = 0

    // iPhone-style toggle
    @State private var showAnnotation = false

    private let numberOfImages = 1

    private let displayWidth: CGFloat = 180
    private let displayHeight: CGFloat = 90

    private let displayCenterX: CGFloat = 0
    private let displayCenterY: CGFloat = 0

    var body: some View {
        GeometryReader { geometry in
            let posX = geometry.size.width / 2 + displayCenterX
            let posY = geometry.size.height / 2 + displayCenterY

            ZStack(alignment: .topLeading) {

                if showCamera {
                    CameraPreviewView(session: vm.captureSession)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .clipped()
                } else {
                    Image("img\(currentImageIndex)")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: geometry.size.width)
                        .clipped()
                }

                VStack(spacing: 10) {
                    Toggle("", isOn: $showAnnotation)
                        .labelsHidden()
                        .toggleStyle(SwitchToggleStyle())
                        .frame(width: 51, height: 31)
                    
                    if vm.screenBrightness < 0.25 {
                        Text("Raise screen brightness")
                            .foregroundColor(.white)
                            .padding(10)
                            .background(Color.black)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .frame(width: displayWidth*3, height: displayHeight)

                    } else if let image = vm.eyeImage {
                        ZStack(alignment: .topLeading) {
                            Image(uiImage: image)
                                .resizable()
                                .interpolation(.high)
                                .scaledToFit()
                                .frame(width: displayWidth, height: displayHeight)
                                .background(Color.black)
                                .clipShape(RoundedRectangle(cornerRadius: 12))

                            // debug start
//                            if vm.debugTopScores.count >= 3 && vm.debugTopLocations.count >= 3 {
//                                VStack(alignment: .leading, spacing: 2) {
//                                    Text(String(format: "1st match: %.0f (%.0f, %.0f)",
//                                                vm.debugTopScores[0],
//                                                vm.debugTopLocations[0].x,
//                                                vm.debugTopLocations[0].y))
//                                        .padding(.top, -30)
//
//                                    Text(String(format: "2nd match: %.0f (%.0f, %.0f)",
//                                                vm.debugTopScores[1],
//                                                vm.debugTopLocations[1].x,
//                                                vm.debugTopLocations[1].y))
//                                        .padding(.top, -15)
//
//                                    Text(String(format: "3rd match: %.0f (%.0f, %.0f)",
//                                                vm.debugTopScores[2],
//                                                vm.debugTopLocations[2].x,
//                                                vm.debugTopLocations[2].y))
//                                        .padding(.top, 15)
//
//                                    if let avg = vm.debugAvgScore {
//                                        Text(String(format: "last %d AVG: %.0f",
//                                                    vm.rollingSuccessCount, avg))
//                                        .padding(.top, 0)
//
//                                    }
//                                    
//                                    Text(String(format: "brightness: %.2f", vm.screenBrightness))
//                                        .padding(.top, 0)
//                                }
//                                .font(.system(size: 14, weight: .bold))
//                                .foregroundColor(.gray)
//                                .fixedSize(horizontal: true, vertical: false)
//                                .offset(x: 0, y: displayHeight + 30)
//                            }
                            // debug end
                        }
                        .frame(width: displayWidth, height: displayHeight, alignment: .topLeading)
                        
                    }
                    else {
                        Text("Waiting for eyes…")
                            .foregroundColor(.white)
                            .padding(10)
                            .background(Color.black)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .frame(width: displayWidth, height: displayHeight)
                    }
                }
                .position(x: posX, y: posY - 150)
            }
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.2)) {
                    if showCamera {
                        showCamera = false
                        currentImageIndex = 1
                    } else {
                        if currentImageIndex < numberOfImages {
                            currentImageIndex += 1
                        } else {
                            completedImageRounds += 1
                            showCamera = true
                            currentImageIndex = 1
                        }
                    }
                    syncTemplateName()
                }
            }
        }
        .onAppear {
            syncTemplateName()
            vm.start()
        }
        .onDisappear {
            vm.stop()
        }
        .onChange(of: showCamera) { _ in
            syncTemplateName()
        }
        .onChange(of: currentImageIndex) { _ in
            syncTemplateName()
        }
        .onChange(of: showAnnotation) { _ in
            syncTemplateName()
        }
    }

    private func syncTemplateName() {
        if showCamera {
            vm.setTemplateImageName(nil, shouldAnnotate: false)
        } else {
            vm.setTemplateImageName(
                "img\(currentImageIndex)",
                shouldAnnotate: showAnnotation
            )
        }
    }
}

final class EyeCropViewModel: ObservableObject {
    @Published var eyeImage: UIImage?
    // debug start
    @Published var debugTopScores: [Double] = []
    @Published var debugTopLocations: [CGPoint] = []
    @Published var debugAvgScore: Double? = nil

    // debug end
    @Published var screenBrightness: CGFloat = UIScreen.main.brightness
    private var lastBrightness: CGFloat = UIScreen.main.brightness
    
    // hyperparameters start
    private let baseSuccessThreshold = -50000.0
    let rollingSuccessCount = 1
    private let allowedScoreDrop = 10000.0
    private var recentSucceededScores: [Double] = []
    // hyperparameters end
    
    private let eyeCropWidth: CGFloat = 260
    private let eyeCropHeight: CGFloat = 130
    //private let templateTargetWidth: Int = 12

    private let debugBoxColor: UIColor = .green
    private let debugBoxThickness: CGFloat = 2
    private var debugSaveCounter: Int = 0

    private let camera = CameraManager()
    var captureSession: AVCaptureSession { camera.captureSession }

    private let ciContext = CIContext()
    private var faceLandmarker: FaceLandmarker?
    private var isRunning = false

    private var templateImageName: String?
    private var shouldAnnotateCurrentFrame: Bool = false
    private var templateCache: [String: GrayImage] = [:]
    

    func setTemplateImageName(_ name: String?, shouldAnnotate: Bool) {
        templateImageName = name
        shouldAnnotateCurrentFrame = shouldAnnotate
    }

    func start() {
        camera.onFrame = { [weak self] sampleBuffer in
            self?.process(sampleBuffer: sampleBuffer)
        }
        try? camera.start()
    }

    func stop() {
        camera.stop()
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
        let currentBrightness = UIScreen.main.brightness

        DispatchQueue.main.async {
            // Detect meaningful change
            if abs(currentBrightness - self.lastBrightness) > 0.01 {
                self.recentSucceededScores.removeAll()
                self.lastBrightness = currentBrightness
            }
            
            self.screenBrightness = currentBrightness
        }
        
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }

        guard
            let sbufCopy = copySampleBuffer(sampleBuffer),
            let pixelBuffer = CMSampleBufferGetImageBuffer(sbufCopy)
        else { return }

        ensureLandmarker()
        guard let landmarker = faceLandmarker else { return }

        do {
            let mpImage = try MPImage(pixelBuffer: pixelBuffer, orientation: .up)
            let result = try landmarker.detect(image: mpImage)

            guard let lms = result.faceLandmarks.first else {
                DispatchQueue.main.async {
                    self.eyeImage = nil
                }
                return
            }

            guard let cropResult = cropEyeCI(
                pixelBuffer: pixelBuffer,
                lms: lms,
                left: L_LEFT_CORNER,
                right: L_RIGHT_CORNER,
                center: L_IRIS,
                cropWidth: eyeCropWidth,
                cropHeight: eyeCropHeight
            ) else {
                return
            }

            //debugSaveCounter += 1
            let normalized = cropResult.image.transformed(
                by: CGAffineTransform(
                    translationX: -cropResult.image.extent.origin.x,
                    y: -cropResult.image.extent.origin.y
                )
            )

            guard let normalizedCG = ciContext.createCGImage(normalized, from: normalized.extent) else {
                return
            }

            var annotatedUIImage = UIImage(cgImage: normalizedCG)

            if shouldAnnotateCurrentFrame,
               screenBrightness >= 0.25,
               let match = findBestTemplateMatch(
                    in: normalized,
                    irisPoint: cropResult.irisCenterPoint
               ) {
                if isSucceededMatch(match.score),
                   let annotated = drawRectangle(
                    on: normalizedCG,
                    rect: match.rect,
                    point: match.topLeftPoint,
                    irisCenterPoint: cropResult.irisCenterPoint,
                    color: debugBoxColor,
                    lineWidth: debugBoxThickness
                ) {
                    // debug start
                    DispatchQueue.main.async {
                        self.debugTopScores = match.topScores
                        self.debugTopLocations = match.topLocations
                    }
                    // debug end
                    
                    annotatedUIImage = annotated
                }
            }
            
            // debug start
            if !shouldAnnotateCurrentFrame {
                DispatchQueue.main.async {
                    self.debugTopScores = []
                }
            }
            // debug end
            
            //saveUIImage_helper2(annotatedUIImage, name: "\(debugSaveCounter)_annotated.jpg")
            let finalUIImage = flipUIImageHorizontally(annotatedUIImage) ?? annotatedUIImage
            
            DispatchQueue.main.async {
                self.eyeImage = finalUIImage
            }
        } catch {
            print("❌ MediaPipe error:", error)
        }
    }
    
    private func isSucceededMatch(_ score: Double) -> Bool {
        // First condition: original threshold
        guard score > baseSuccessThreshold else {
            return false
        }

        // If we don't have history yet, accept it
        guard !recentSucceededScores.isEmpty else {
            recentSucceededScores.append(score)
            return true
        }

        let avgScore = recentSucceededScores.reduce(0, +) / Double(recentSucceededScores.count)
        DispatchQueue.main.async {
            self.debugAvgScore = avgScore
        }
        
        let dynamicThreshold = avgScore - allowedScoreDrop

        guard score >= dynamicThreshold else {
            return false
        }

        recentSucceededScores.append(score)

        if recentSucceededScores.count > rollingSuccessCount {
            recentSucceededScores.removeFirst()
        }

        return true
    }

    private func findBestTemplateMatch(in eyeCI: CIImage, irisPoint: CGPoint) -> MatchDebugResult? {
        //guard let baseName = templateImageName else { return nil }
        let brightness = screenBrightness
        let templateImageFileName: String
        if brightness < 0.25 {
            return nil
        } else if brightness < 0.35 {
            templateImageFileName = "template_brightness25_subtract100"
        } else if brightness < 0.65 {
            templateImageFileName = "template_brightness50_subtract80"
        } else if brightness < 0.85 {
            templateImageFileName = "template_brightness75_subtract60"
        } else {
            templateImageFileName = "template_brightness100_subtract40"
        }

        let templateGray: GrayImage
        if let cached = templateCache[templateImageFileName] {
            templateGray = cached
        } else {
            guard
                let templateUIImage = UIImage(named: templateImageFileName),
                let templateCG = templateUIImage.cgImage,
                let loadedTemplateGray = makeGrayImage(from: templateCG)
            else {
                return nil
            }
            
            templateCache[templateImageFileName] = loadedTemplateGray
            templateGray = loadedTemplateGray
        }

        guard
            let eyeCG = ciContext.createCGImage(eyeCI, from: eyeCI.extent),
            let eyeGray = makeGrayImage(from: eyeCG)
        else {
            return nil
        }

        let tplW = templateGray.width
        let tplH = templateGray.height

        guard eyeGray.width >= tplW, eyeGray.height >= tplH else { return nil }

        let centerX = Int(round(irisPoint.x))
        let centerY = Int(round(irisPoint.y))

        let searchRangeX = 4
        let searchRangeY = 6

        let minX = max(0, centerX - searchRangeX - tplW / 2)
        let maxX = min(eyeGray.width - tplW, centerX + searchRangeX - tplW / 2)
        let minY = max(0, centerY - searchRangeY - tplH / 2)
        let maxY = min(eyeGray.height - tplH, centerY + searchRangeY - tplH / 2)

        guard minX <= maxX, minY <= maxY else { return nil }

        var topScoresAndLocations:[(score: Double, location: CGPoint)] = []    // debug, for top3 score
      
        var bestScore = -Double.infinity
        var bestX = minX
        var bestY = minY

        for y in minY...maxY {
            for x in minX...maxX {
                let score = doTemplateMatch(
                    image: eyeGray,
                    template: templateGray,
                    originX: x,
                    originY: y
                )
                topScoresAndLocations.append((score, CGPoint(x: x, y: y)))
                
                if score > bestScore {
                    bestScore = score
                    bestX = x
                    bestY = y
                }
            }
        }
        
        let sortedTop3 = topScoresAndLocations
            .sorted { $0.score > $1.score }
            .prefix(3)
        
        let annotationRect = CGRect(
            x: CGFloat(bestX),
            y: CGFloat(bestY),
            width: CGFloat(tplW),
            height: CGFloat(tplH)
        )

        return MatchDebugResult(
            rect: annotationRect,
            topLeftPoint: CGPoint(x: CGFloat(bestX), y: CGFloat(bestY)),
            score: bestScore,
            eyeGray: eyeGray,
            templateGray: templateGray,
            topScores: sortedTop3.map { $0.score },
            topLocations: sortedTop3.map { $0.location }
        )
    }

//    private func resizeCGImageKeepingAspect(_ cg: CGImage, targetWidth: Int) -> CGImage? {
//        let srcW = CGFloat(cg.width)
//        let srcH = CGFloat(cg.height)
//        guard srcW > 0, srcH > 0 else { return nil }
//
//        let scale = CGFloat(targetWidth) / srcW
//        let targetHeight = max(1, Int(round(srcH * scale)))
//
//        let colorSpace = CGColorSpaceCreateDeviceGray()
//        guard let ctx = CGContext(
//            data: nil,
//            width: targetWidth,
//            height: targetHeight,
//            bitsPerComponent: 8,
//            bytesPerRow: targetWidth,
//            space: colorSpace,
//            bitmapInfo: CGImageAlphaInfo.none.rawValue
//        ) else {
//            return nil
//        }
//
//        ctx.interpolationQuality = .high
//        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
//        return ctx.makeImage()
//    }

    private func makeGrayImage(from cgImage: CGImage) -> GrayImage? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        let bytesPerRow = width
        var data = [UInt8](repeating: 0, count: width * height)

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }

        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return GrayImage(width: width, height: height, pixels: data)
    }

    private func doTemplateMatch(
        image: GrayImage,
        template: GrayImage,
        originX: Int,
        originY: Int
    ) -> Double {
        let w = template.width
        let h = template.height
        let n = Double(w * h)

        var sum = 0.0

        for j in 0..<h {
            let imageRow = (originY + j) * image.width
            let templateRow = j * template.width
            for i in 0..<w {
                let iv = Double(image.pixels[imageRow + originX + i])
                let tv = Double(template.pixels[templateRow + i])
                let d = iv - tv
                sum += d * d
            }
        }

        let mse = sum / n
        return -mse
    }

    private func flipUIImageHorizontally(_ image: UIImage) -> UIImage? {
        let baseRenderer = UIGraphicsImageRenderer(size: image.size)
        let normalizedImage = baseRenderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }

        guard let cg = normalizedImage.cgImage else { return nil }
        let mirrored = UIImage(cgImage: cg, scale: normalizedImage.scale, orientation: .upMirrored)

        let renderer = UIGraphicsImageRenderer(size: normalizedImage.size)
        return renderer.image { _ in
            mirrored.draw(in: CGRect(origin: .zero, size: normalizedImage.size))
        }
    }

    private func drawRectangle(
        on cgImage: CGImage,
        rect: CGRect,
        point: CGPoint,
        irisCenterPoint: CGPoint,
        color: UIColor,
        lineWidth: CGFloat
    ) -> UIImage? {
        let width = cgImage.width
        let height = cgImage.height

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        return renderer.image { ctx in
            UIImage(cgImage: cgImage).draw(in: CGRect(x: 0, y: 0, width: width, height: height))
            ctx.cgContext.setStrokeColor(color.cgColor)
            ctx.cgContext.setLineWidth(lineWidth)
            ctx.cgContext.stroke(rect)
        }
    }

    private func saveUIImage_helper2(_ image: UIImage, name: String) {
        guard
            let dir = documentsDirectoryURL(),
            let data = image.jpegData(compressionQuality: 0.95)
        else { return }

        let url = dir.appendingPathComponent(name)
        try? data.write(to: url, options: .atomic)
    }

    private func documentsDirectoryURL() -> URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    }
}

struct EyeCropResult {
    let image: CIImage
    let irisCenterPoint: CGPoint
}

struct GrayImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]
}

struct MatchDebugResult {
    let rect: CGRect
    let topLeftPoint: CGPoint
    let score: Double
    let eyeGray: GrayImage
    let templateGray: GrayImage
    
    // debug start (top 3 scores)
    let topScores: [Double]
    let topLocations: [CGPoint]
    // debug end
}

func copySampleBuffer(_ sbuf: CMSampleBuffer) -> CMSampleBuffer? {
    var copy: CMSampleBuffer?
    let status = CMSampleBufferCreateCopy(
        allocator: kCFAllocatorDefault,
        sampleBuffer: sbuf,
        sampleBufferOut: &copy
    )
    return status == noErr ? copy : nil
}

func lmkPx(_ lms: [NormalizedLandmark], _ idx: Int, _ w: CGFloat, _ h: CGFloat) -> CGPoint {
    CGPoint(
        x: CGFloat(lms[idx].x) * w,
        y: CGFloat(lms[idx].y) * h
    )
}

func cropEyeCI(
    pixelBuffer: CVPixelBuffer,
    lms: [NormalizedLandmark],
    left: Int,
    right: Int,
    center: Int,
    cropWidth: CGFloat,
    cropHeight: CGFloat
) -> EyeCropResult? {
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

    guard eyeWidth > 1 else { return nil }

    let scale = 250.0 / eyeWidth
    let src = CIImage(cvPixelBuffer: pixelBuffer)

    let scaled = src.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    let extent = scaled.extent

    let scaledCenter = CGPoint(
        x: C.x * scale,
        y: C.y * scale
    )

    var cropX = scaledCenter.x - cropWidth / 2
    var cropY = scaledCenter.y - cropHeight / 2

    cropX = max(extent.minX, min(cropX, extent.maxX - cropWidth))
    cropY = max(extent.minY, min(cropY, extent.maxY - cropHeight))

    let cropRect = CGRect(
        x: cropX.rounded(.down),
        y: cropY.rounded(.down),
        width: cropWidth.rounded(.down),
        height: cropHeight.rounded(.down)
    )

    let cropped = scaled.cropped(to: cropRect)

    let irisInCrop = CGPoint(
        x: scaledCenter.x - cropRect.minX,
        y: scaledCenter.y - cropRect.minY
    )

    return EyeCropResult(
        image: cropped,
        irisCenterPoint: irisInCrop
    )
}
