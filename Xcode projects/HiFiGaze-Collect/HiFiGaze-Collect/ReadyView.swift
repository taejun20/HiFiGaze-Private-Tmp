import SwiftUI

struct ReadyView: View {
    let subject: String
    let onAllDone: () -> Void

    @State private var backgroundIndex: Int = 0
    @State private var isCollecting: Bool = false

    // Background asset names (add A/B/C images to Assets)
    private let backgrounds: [String] = ["screen1", "screen2", "screen3"]

    var body: some View {
        let bgName = backgrounds[backgroundIndex]

        if isCollecting {
            CollectionView(
                subject: subject,
                screenName: bgName,
                randomTrialCount: 100
            ) {
                // finished this background
                isCollecting = false
                if backgroundIndex < backgrounds.count - 1 {
                    backgroundIndex += 1
                } else {
                    onAllDone()
                }
            }
        } else {
            VStack(spacing: 24) {
                Text("Background: \(bgName)")
                    .font(.title.bold())

                Button("Start (~10 mins)") {
                    isCollecting = true
                }
                .font(.title.bold())
                .padding(.horizontal, 30)
                .padding(.vertical, 14)
                .background(Color.white.opacity(0.9))
                .foregroundColor(.black)
                .cornerRadius(14)

                Button("Change Subject") {
                    // just pop back by resetting state in ContentView
                    // easiest is to dismiss view: handled by parent,
                    // so we provide a navigation back option:
                }
                .hidden()
            }
            .padding()
        }
    }
}
