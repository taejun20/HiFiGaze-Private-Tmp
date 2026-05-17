import SwiftUI

struct ContentView: View {
    @State private var subject: String? = nil
    @State private var doneAll: Bool = false

    var body: some View {
        NavigationStack {
            if doneAll {
                VStack(spacing: 20) {
                    Text("Collection Done")
                        .font(.largeTitle.bold())
                    Button("Back to Start") {
                        subject = nil
                        doneAll = false
                    }
                    .font(.title2)
                }
            } else if let subject {
                ReadyView(subject: subject) {
                    // finished all backgrounds
                    doneAll = true
                }
            } else {
                SubjectSelectionView { selected in
                    subject = selected
                }
            }
        }
    }
}
