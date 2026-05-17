import SwiftUI

struct DotView: View {
    let showLetter: Bool
    let letter: String
    let oscillating: Bool

    let baseOuterRadiusPt: CGFloat
    let maxOuterRadiusPt: CGFloat
    let innerRadiusPt: CGFloat

    @State private var expanded: Bool = false

    var body: some View {
        ZStack {
            // ✅ Outer BLACK filled circle
            Circle()
                .fill(Color.black)
                .frame(width: outerDiameter, height: outerDiameter)

            // ✅ Inner WHITE filled circle (on top)
            Circle()
                .fill(Color.white)
                .frame(width: innerRadiusPt * 2, height: innerRadiusPt * 2)

            if showLetter {
                Text(letter)
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(.red)
            }
        }
        .onChange(of: oscillating) { _, isOn in
            expanded = isOn
        }
        .animation(
            oscillating
            ? .easeInOut(duration: 0.45).repeatForever(autoreverses: true)
            : .easeInOut(duration: 0.2),
            value: expanded
        )
    }

    private var outerDiameter: CGFloat {
        let r = expanded ? maxOuterRadiusPt : baseOuterRadiusPt
        return r * 2
    }
}
