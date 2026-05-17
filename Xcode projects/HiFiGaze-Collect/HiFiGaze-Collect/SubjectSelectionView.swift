import SwiftUI

struct SubjectSelectionView: View {
    let onSelect: (String) -> Void

    var body: some View {
        ZStack {
            // 🔲 Black background
            Color.black
                .ignoresSafeArea()

            VStack(spacing: 40) {

                // 🔤 Title
                Text("Select Subject")
                    .font(.largeTitle.bold())
                    .foregroundColor(.white)

                VStack(spacing: 30) {

                    // Row 1
                    HStack(spacing: 40) {
                        subjectButton("1")
                        subjectButton("2")
                    }

                    // Row 2
                    HStack(spacing: 40) {
                        subjectButton("3")
                        subjectButton("4")
                    }

                    // Row 3
                    HStack(spacing: 40) {
                        subjectButton("5")
                        subjectButton("6")
                    }
                }
            }
        }
    }

    // MARK: - Button View
    private func subjectButton(_ id: String) -> some View {
        Button {
            onSelect(id)
        } label: {
            Text("p\(id)")
                .font(.largeTitle.bold())
                .foregroundColor(.black)
                .frame(width: 140, height: 90)
                .background(Color.white)
                .cornerRadius(16)
        }
    }
}
