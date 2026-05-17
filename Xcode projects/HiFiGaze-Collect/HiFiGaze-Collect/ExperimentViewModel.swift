import Foundation
import SwiftUI

@MainActor
final class ExperimentViewModel: ObservableObject {
    // UI state
    @Published var dotPosition: CGPoint? = nil
    @Published var oscillating: Bool = false
    @Published var showLetter: Bool = false
    @Published var letterToShow: String = ""
    @Published var showResponseButtons: Bool = false
    @Published var progressText: String = ""

    // “mm to points” approximation: iOS is points, not mm.
    // This is a reasonable constant; adjust if you want device-specific calibration.
    let baseOuterRadiusPt: CGFloat = 10   // ~10mm-ish feel
    let maxOuterRadiusPt: CGFloat = 20    // ~30mm-ish feel
    let innerRadiusPt: CGFloat = 4        // ~5mm-ish feel

    // Config
    private var subject: String = ""
    private var screenName: String = ""
    private var canvasSize: CGSize = .zero
    private var randomTrialCount: Int = 100

    // Logic
    private var targets: [TrialTarget] = []
    private var currentIndex: Int = 0
    private var expected: String = "L"
    private var isRunning: Bool = false
    private var finishCallback: (() -> Void)?

    // Camera + recording
    private let camera = CameraManager()
    private let recorder = DataRecorder()

    func configure(subject: String, screenName: String, canvasSize: CGSize, randomTrialCount: Int) {
        self.subject = subject
        self.screenName = screenName
        self.canvasSize = canvasSize
        self.randomTrialCount = randomTrialCount

        targets = makeTargets(size: canvasSize, randomCount: randomTrialCount)
        currentIndex = 0
        dotPosition = CGPoint(x: canvasSize.width/2, y: canvasSize.height/2)
        progressText = "0 / \(targets.count)"
    }

    func startIfNeeded(onDone: @escaping () -> Void) {
        guard !isRunning else { return }
        isRunning = true
        finishCallback = onDone

        // camera
        camera.configureFrontCameraBestEffort4K()
        camera.onSampleBuffer = { [weak self] sbuf in
            guard let self else { return }
            self.recorder.handleSampleBuffer(sbuf)
        }
        camera.start()

        recorder.configure(subject: subject, screenName: screenName)
        recorder.startSessionClock()

        Task {
            await runTrials()
        }
    }

    func stopEverything() {
        isRunning = false
        camera.stop()
        recorder.isRecording = false
        oscillating = false
        showLetter = false
        showResponseButtons = false
    }

    func submitResponse(_ r: LRResponse) {
        guard showResponseButtons else { return }

        switch r {
        case .left:
            handleAnswer("L")

        case .right:
            handleAnswer("R")
        }
    }

    // MARK: - Core flow

    private func runTrials() async {
        while isRunning, currentIndex < targets.count {
            progressText = "\(currentIndex) / \(targets.count)"
            await runOneTrial()
            // Wait here until user answers (or redo triggers)
            // runOneTrial sets UI to response mode; we park using continuation.
            let ok = await waitForAnswer()
            if !ok {
                // redo requested inside wait loop
                continue
            }
            currentIndex += 1
        }

        // Done
        recorder.finalizeCSV()
        camera.stop()
        oscillating = false
        showLetter = false
        showResponseButtons = false
        progressText = "\(targets.count) / \(targets.count)"

        finishCallback?()
    }

    private var answerContinuation: CheckedContinuation<Bool, Never>? = nil
    private var pendingRedo: Bool = false
    private var lastAnswerCorrect: Bool = false

    private func waitForAnswer() async -> Bool {
        // Wait until submitResponse resolves it
        return await withCheckedContinuation { cont in
            answerContinuation = cont
        }
    }

    private func handleAnswer(_ answer: String) {
        guard showResponseButtons else { return }
        showResponseButtons = false

        if answer == expected {
            // commit trial
            recorder.commitTrial()
            lastAnswerCorrect = true
            answerContinuation?.resume(returning: true)
            answerContinuation = nil
        } else {
            // discard + redo
            pendingRedo = true
            Task { await redoCurrentTrial() }
        }
    }

    private func redoCurrentTrial() async {
        // discard captured frames/rows and rerun same target
        recorder.discardTrial()
        pendingRedo = false
        lastAnswerCorrect = false

        // Run same trial again
        await runOneTrial()
        // Wait for answer again
        let ok = await waitForAnswer()
        if ok {
            // will advance by caller loop only when ok and currentIndex increments
            return
        }
    }

    private func runOneTrial() async {
        guard isRunning else { return }

        // Setup per-trial context
        recorder.beginTrialRecordingContext()

        // Move dot: center -> target
        let target = targets[currentIndex].point

        oscillating = false
        showLetter = false
        showResponseButtons = false

        // short settle
        try? await Task.sleep(nanoseconds: 120_000_000)

        // animate to target
        withAnimation(.easeInOut(duration: 0.25)) {
            dotPosition = target
        }
        // “arrival” delay
        try? await Task.sleep(nanoseconds: 260_000_000)

        // Start oscillation for 2 seconds
        oscillating = true

        // Recording: start at +0.5s, end at +2.0s
        recorder.currentGT = target

        // schedule recording window
        Task.detached { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: 500_000_000)
            await MainActor.run { self.recorder.isRecording = true }
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            await MainActor.run { self.recorder.isRecording = false }
        }

        // dwell total 2.0s
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        oscillating = false

        // Show L/R for 0.05s
        expected = Bool.random() ? "L" : "R"
        letterToShow = expected
        showLetter = true
        try? await Task.sleep(nanoseconds: 50_000_000)
        showLetter = false

        // Show response buttons
        showResponseButtons = true
        progressText = "\(currentIndex + 1) / \(targets.count)"
    }

    // MARK: - Targets

    private func makeTargets(size: CGSize, randomCount: Int) -> [TrialTarget] {
        let margin: CGFloat = 20
        let cols = 10
        let rows = 5

        let xs: [CGFloat] = (0..<cols).map { i in
            if cols == 1 { return size.width/2 }
            return margin + CGFloat(i) * (size.width - 2*margin) / CGFloat(cols - 1)
        }
        let ys: [CGFloat] = (0..<rows).map { j in
            if rows == 1 { return size.height/2 }
            return margin + CGFloat(j) * (size.height - 2*margin) / CGFloat(rows - 1)
        }

        var grid: [TrialTarget] = []
        for y in ys {
            for x in xs {
                grid.append(TrialTarget(point: CGPoint(x: x, y: y), isGrid: true))
            }
        }
        // grid is exactly 50 points (10*5)
        grid.shuffle()

        var randoms: [TrialTarget] = []
        for _ in 0..<randomCount {
            let x = CGFloat.random(in: margin...(size.width - margin))
            let y = CGFloat.random(in: margin...(size.height - margin))
            randoms.append(TrialTarget(point: CGPoint(x: x, y: y), isGrid: false))
        }

        return grid + randoms
    }
}
