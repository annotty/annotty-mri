import SwiftUI

/// Inline training status display for the right panel
struct TrainingStatusView: View {
    let status: HILServerClient.TrainingStatus?

    var body: some View {
        if let status = status {
            VStack(spacing: 4) {
                HStack(spacing: 6) {
                    statusIcon(status.status)
                    Text(statusLabel(status.status))
                        .font(.caption2)
                        .foregroundColor(.white)
                }

                if status.status == "running", let epoch = status.epoch, let maxEpochs = status.maxEpochs {
                    ProgressView(value: Double(epoch), total: Double(maxEpochs))
                        .progressViewStyle(LinearProgressViewStyle(tint: .yellow))
                    Text("Epoch \(epoch)/\(maxEpochs)")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }

                if status.status == "completed", let dice = status.bestDice {
                    Text(String(format: "Dice: %.3f", dice))
                        .font(.caption2)
                        .foregroundColor(.green)
                }

                if status.status == "error" {
                    Text("Training error")
                        .font(.caption2)
                        .foregroundColor(.red)
                        .lineLimit(2)
                }
            }
            .padding(8)
            .background(Color(white: 0.15))
            .cornerRadius(6)
        }
    }

    @ViewBuilder
    private func statusIcon(_ status: String) -> some View {
        switch status {
        case "idle":
            Image(systemName: "circle.fill").foregroundColor(.blue).font(.caption2)
        case "running":
            ProgressView().scaleEffect(0.5)
        case "completed":
            Image(systemName: "checkmark.circle.fill").foregroundColor(.green).font(.caption2)
        case "error":
            Image(systemName: "exclamationmark.circle.fill").foregroundColor(.red).font(.caption2)
        default:
            Image(systemName: "questionmark.circle").foregroundColor(.gray).font(.caption2)
        }
    }

    private func statusLabel(_ status: String) -> String {
        switch status {
        case "idle": return "Idle"
        case "running": return "Training..."
        case "completed": return "Completed"
        case "error": return "Error"
        default: return status
        }
    }
}
