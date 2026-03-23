import SwiftUI

/// Export sheet for selecting export formats and triggering export
struct ExportSheetView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var viewModel: CanvasViewModel

    @State private var exportPNG = true
    @State private var exportCOCO = true
    @State private var exportYOLO = true
    @State private var isExporting = false
    @State private var exportComplete = false
    @State private var showingShareSheet = false
    @State private var exportedURLs: [URL] = []

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Format selection
                VStack(alignment: .leading, spacing: 16) {
                    Text("Export Formats")
                        .font(.headline)

                    Toggle(isOn: $exportPNG) {
                        HStack {
                            Image(systemName: "photo")
                            Text("PNG Mask")
                        }
                    }

                    Toggle(isOn: $exportCOCO) {
                        HStack {
                            Image(systemName: "doc.text")
                            Text("COCO JSON")
                        }
                    }

                    Toggle(isOn: $exportYOLO) {
                        HStack {
                            Image(systemName: "list.bullet.rectangle")
                            Text("YOLO-seg TXT")
                        }
                    }
                }
                .padding()
                .background(Color(UIColor.secondarySystemBackground))
                .cornerRadius(12)

                // Export button
                Button(action: performExport) {
                    HStack {
                        if isExporting {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Image(systemName: "square.and.arrow.up")
                        }
                        Text(isExporting ? "Exporting..." : "Export")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(canExport ? Color.green : Color.gray)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .disabled(!canExport || isExporting)

                if exportComplete {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("Export complete!")
                    }

                    Button("Share Files") {
                        showingShareSheet = true
                    }
                }

                Spacer()

                // Info text
                Text("Files will be saved to the labels/ folder")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .padding(24)
            .navigationTitle("Export Annotation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $showingShareSheet) {
                if !exportedURLs.isEmpty {
                    ShareSheet(items: exportedURLs)
                }
            }
        }
    }

    private var canExport: Bool {
        exportPNG || exportCOCO || exportYOLO
    }

    private func performExport() {
        isExporting = true

        // Simulate export (actual implementation in Phase 5)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            isExporting = false
            exportComplete = true

            // Create dummy URLs for now
            let tempDir = FileManager.default.temporaryDirectory
            exportedURLs = []

            if exportPNG {
                exportedURLs.append(tempDir.appendingPathComponent("mask.png"))
            }
            if exportCOCO {
                exportedURLs.append(tempDir.appendingPathComponent("annotation.json"))
            }
            if exportYOLO {
                exportedURLs.append(tempDir.appendingPathComponent("annotation.txt"))
            }
        }
    }
}

/// UIKit share sheet wrapper
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Preview

#Preview {
    ExportSheetView(viewModel: CanvasViewModel())
}
