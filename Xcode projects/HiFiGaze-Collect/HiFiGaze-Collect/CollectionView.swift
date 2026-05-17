import SwiftUI

struct CollectionView: View {
    let subject: String
    let screenName: String
    let randomTrialCount: Int
    let onDone: () -> Void

    @StateObject private var vm = ExperimentViewModel()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Background image
                Image(screenName)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geo.size.width)
                    .clipped()

                // Dot
                DotView(
                    showLetter: vm.showLetter,
                    letter: vm.letterToShow,
                    oscillating: vm.oscillating,
                    baseOuterRadiusPt: vm.baseOuterRadiusPt,
                    maxOuterRadiusPt: vm.maxOuterRadiusPt,
                    innerRadiusPt: vm.innerRadiusPt
                )
                .position(vm.dotPosition ?? CGPoint(x: geo.size.width/2, y: geo.size.height/2))
                .animation(.easeInOut(duration: 0.25), value: vm.dotPosition)

                // Response buttons
                if vm.showResponseButtons {
                    HStack(spacing: 0) {
                        // LEFT HALF
                        Color.clear
                            .contentShape(Rectangle())
                            .onTapGesture {
                                vm.submitResponse(.left)
                            }

                        // RIGHT HALF
                        Color.clear
                            .contentShape(Rectangle())
                            .onTapGesture {
                                vm.submitResponse(.right)
                            }
                    }
                    .ignoresSafeArea()
                }
            }
            .onAppear {
                vm.configure(
                    subject: subject,
                    screenName: screenName,
                    canvasSize: geo.size,
                    randomTrialCount: randomTrialCount
                )
                vm.startIfNeeded {
                    onDone()
                }
            }
            .onDisappear {
                vm.stopEverything()
            }
        }
    }
}
