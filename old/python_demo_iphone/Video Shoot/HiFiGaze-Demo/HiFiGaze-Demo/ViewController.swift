import UIKit
import AVFoundation
import CoreImage

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    // You can remove this outlet from storyboard later; it's unused now.
    @IBOutlet weak var imageView: UIImageView!

    private let startButton: UIButton = {
        let b = UIButton(type: .system)
        b.setTitle("Start", for: .normal)
        b.titleLabel?.font = .systemFont(ofSize: 22, weight: .bold)
        b.translatesAutoresizingMaskIntoConstraints = false
        b.backgroundColor = .systemBlue
        b.setTitleColor(.white, for: .normal)
        b.layer.cornerRadius = 14
        b.contentEdgeInsets = UIEdgeInsets(top: 12, left: 28, bottom: 12, right: 28)
        return b
    }()

    private let finishedLabel: UILabel = {
        let l = UILabel()
        l.text = "Finished"
        l.font = .systemFont(ofSize: 28, weight: .bold)
        l.textAlignment = .center
        l.textColor = .white
        l.backgroundColor = .black
        l.layer.cornerRadius = 12
        l.clipsToBounds = true
        l.translatesAutoresizingMaskIntoConstraints = false
        l.isHidden = true
        return l
    }()
    // Capture
    private let captureSession = AVCaptureSession()
    private var videoOutput = AVCaptureVideoDataOutput()
    private let captureQueue = DispatchQueue(label: "camera.capture.queue")
    private let ciContext = CIContext(options: nil)

    // Recording control
    private var isRecording = false
    private var startTime: CFAbsoluteTime = 0
    private var frameIndex = 0
    private var csvLines: [String] = ["frame_num,time"]
    var captureDurationSeconds: TimeInterval = 20.0

    // Files
    private var sessionDirURL: URL?
    private var csvURL: URL? { sessionDirURL?.appendingPathComponent("frames.csv") }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        configureSession() // sets up camera, but doesn't start it yet
    }

    private func setupUI() {
        view.addSubview(startButton)
        view.addSubview(finishedLabel)

        NSLayoutConstraint.activate([
            startButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            startButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            finishedLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            finishedLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
        ])

        startButton.addTarget(self, action: #selector(startTapped), for: .touchUpInside)
    }

    // MARK: Session setup (no preview layer)
    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .inputPriority // ✅ honor activeFormat

        // Front camera
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .front) else {
            print("No front camera.")
            captureSession.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { captureSession.addInput(input) }

            try device.lockForConfiguration()

            // Pick a VIDEO format that is exactly (or at least) 3840×2160 and supports >= 30 fps
            let targetW = 3840
            let targetH = 2160

            let best4K: AVCaptureDevice.Format? = device.formats
                .filter { format in
                    // Use VIDEO dimensions, not still image dimensions
                    let desc = format.formatDescription
                    let dims = CMVideoFormatDescriptionGetDimensions(desc)
                    let w = Int(dims.width), h = Int(dims.height)

                    // Some devices report rotated dims; normalize by sorting (min/max)
                    let minSide = min(w, h), maxSide = max(w, h)
                    let minTarget = min(targetW, targetH), maxTarget = max(targetW, targetH)

                    // Accept exact 3840x2160 (in either orientation)
                    guard maxSide == maxTarget && minSide == minTarget else { return false }

                    // Ensure it can do >= 30 fps
                    guard let range = format.videoSupportedFrameRateRanges.first,
                          range.maxFrameRate >= 30 else { return false }

                    return true
                }
                // Prefer highest max fps
                .sorted {
                    let r0 = $0.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 0
                    let r1 = $1.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 0
                    return r0 > r1
                }
                .first

            if let f = best4K {
                device.activeFormat = f
                // Lock to 30 fps; adjust if you prefer 60 and the format supports it
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            } else {
                print("⚠️ No 4K 3840x2160 video format found for front camera; falling back.")
                // As a fallback, try the 4K preset if supported
                if captureSession.canSetSessionPreset(.hd4K3840x2160) {
                    captureSession.sessionPreset = .hd4K3840x2160
                } else if captureSession.canSetSessionPreset(.high) {
                    captureSession.sessionPreset = .high
                }
            }

            device.unlockForConfiguration()
        } catch {
            print("Device config error: \(error)")
            captureSession.commitConfiguration()
            return
        }

        // Data output
        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: captureQueue)

        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }

        // Orientation/mirroring
        if let conn = videoOutput.connection(with: .video) {
            conn.videoOrientation = .portrait
            conn.isVideoMirrored = true
        }

        captureSession.commitConfiguration()
    }



    // MARK: Start/stop
    @objc private func startTapped() {
        requestPermissionIfNeeded { granted in
            guard granted else { return }
            DispatchQueue.main.async {
                self.finishedLabel.isHidden = true
                self.startButton.isHidden = true
                self.startButton.setTitle("Start Again", for: .normal)
            }
            self.startRecording()
        }
    }

    private func startRecording() {
        sessionDirURL = makeNewSessionDirectory()
        frameIndex = 0
        csvLines = ["frame_num,time"]
        isRecording = true

        if !captureSession.isRunning {
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        }

        startTime = CFAbsoluteTimeGetCurrent()

        DispatchQueue.global().asyncAfter(deadline: .now() + captureDurationSeconds) { [weak self] in
            self?.stopRecordingAndFinalize()
        }
    }

    private func stopRecordingAndFinalize() {
        isRecording = false

            if captureSession.isRunning { captureSession.stopRunning() }

            if let csvURL = csvURL {
                let text = csvLines.joined(separator: "\n")
                try? text.data(using: .utf8)?.write(to: csvURL, options: .atomic)
            }

            DispatchQueue.main.async {
                // Show only the Finished label
                self.finishedLabel.isHidden = false
                self.startButton.isHidden = true          // <- keep this hidden at finish
                self.startButton.isUserInteractionEnabled = false
                self.startButton.alpha = 0                // extra safety so it can’t flash
            }

            if let dir = sessionDirURL {
                print("Saved session in: \(dir.path)")
            }
    }

    // MARK: Frame capture
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard isRecording else { return }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }

        let filename = "\(frameIndex).jpg"
        if let dir = sessionDirURL {
            let url = dir.appendingPathComponent(filename)
            if let data = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.9) {
                do { try data.write(to: url, options: .atomic) }
                catch { print("Write failed \(filename): \(error)") }
            }
        }

        csvLines.append("\(frameIndex),\(String(format: "%.3f", elapsed))")
        frameIndex += 1
    }

    // MARK: Helpers
    private func makeNewSessionDirectory() -> URL? {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        let dir = docs.appendingPathComponent("session_\(fmt.string(from: Date()))", isDirectory: true)
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        } catch {
            print("Create dir error: \(error)")
            return nil
        }
    }

    private func requestPermissionIfNeeded(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in completion(granted) }
        case .denied, .restricted:
            DispatchQueue.main.async {
                let alert = UIAlertController(
                    title: "Camera Access Needed",
                    message: "Enable camera access in Settings to record frames.",
                    preferredStyle: .alert
                )
                alert.addAction(UIAlertAction(title: "OK", style: .default))
                self.present(alert, animated: true)
            }
            completion(false)
        @unknown default: completion(false)
        }
    }
}
